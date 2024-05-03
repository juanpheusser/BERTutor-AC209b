import torch, torchvision
import matplotlib.pyplot as plt
import json
import cv2
import numpy as np
from copy import deepcopy
from PIL import Image
from io import BytesIO
import gc
from tqdm import tqdm
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures.image_list import ImageList
from detectron2.data import transforms as T
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.structures.boxes import Boxes
from detectron2.layers import nms
from detectron2 import model_zoo
from detectron2.config import get_cfg


class GetVisualEmbeddings:
    MIN_BOXES = 10
    MAX_BOXES = 100

    def __init__(self, cfg_path, cuda=True):
        self.cfg = self.load_config_and_model_weights(cfg_path, cuda)
        self.model = self.get_model(self.cfg)


    def get_visual_embeddings(self, df, batch_size):

        df['batch'] = np.arange(0, len(df)) // batch_size
        output_visual_embeddings = []

        for _, batch in tqdm(df.groupby('batch')):
            
            img_list = self.bytes_to_images(batch)
            del batch
            gc.collect()

            images, batched_inputs = self.prepare_image_inputs(self.cfg, img_list)
            del img_list
            gc.collect()

            features = self.get_features(images)
            proposals = self.get_proposals(images, features)
            box_features, features_list = self.get_box_features(features, proposals)
            pred_class_logits, pred_proposal_deltas = self.get_prediction_logits(features_list, proposals)
            boxes, scores, image_shapes = self.get_box_scores(pred_class_logits, pred_proposal_deltas, proposals)

            del image_shapes, features_list, pred_class_logits, pred_proposal_deltas        
            gc.collect()

            output_boxes = [self.get_output_boxes(boxes[i], batched_inputs[i], proposals[i].image_size) for i in range(len(proposals))]
            temp = [self.select_boxes(self.cfg, output_boxes[i], scores[i]) for i in range(len(scores))]

            keep_boxes, max_conf = [],[]
            for keep_box, mx_conf in temp:
                keep_boxes.append(keep_box)
                max_conf.append(mx_conf)

            keep_boxes = [self.filter_boxes(keep_box, mx_conf, self.MIN_BOXES, self.MAX_BOXES) for keep_box, mx_conf in zip(keep_boxes, max_conf)]
            visual_embeds = [self.get_visual_embeds(box_feature, keep_box) for box_feature, keep_box in zip(box_features, keep_boxes)]

            output_visual_embeddings.extend(visual_embeds)
            
            del boxes, scores, output_boxes, keep_boxes, max_conf, visual_embeds, temp, proposals, features, images, batched_inputs, box_features
            gc.collect()
            torch.cuda.empty_cache()

        return output_visual_embeddings

    def load_config_and_model_weights(self, cfg_path, cuda=True):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(cfg_path))

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        if not cuda:
            cfg['MODEL']['DEVICE']='cpu'

        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)

        return cfg
    
    def get_model(self, cfg):
        model = build_model(cfg)
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        model.eval()

        return model
    
    def bytes_to_images(self, df):

        min_height = float('inf')
        min_width = float('inf')

        img_list = []

        for image_data in df['image'].tolist():
            if image_data is not None:
                image = np.array(Image.open(BytesIO(image_data['bytes'])), dtype=np.uint8)
                img_list.append(image)

                min_height = min(min_height, image.shape[0])
                min_width = min(min_width, image.shape[1])

            else:
                img_list.append(None)

        img_list = [img if img is not None else np.zeros((min_height, min_width, 3), dtype=np.uint8) for img in img_list]

        return img_list

    
    def prepare_image_inputs(self, cfg, img_list):
        # Resizing the image according to the configuration
        transform_gen = T.ResizeShortestEdge(
                    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
                )
        img_list = [transform_gen.get_transform(img).apply_image(img) for img in img_list]

        # Convert to C,H,W format
        convert_to_tensor = lambda x: torch.Tensor(x.astype("float32").transpose(2, 0, 1))

        batched_inputs = [{"image":convert_to_tensor(img), "height": img.shape[0], "width": img.shape[1]} for img in img_list]

        # Normalizing the image
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1)
        normalizer = lambda x: (x - pixel_mean) / pixel_std
        images = [normalizer(x["image"]) for x in batched_inputs]

        # Convert to ImageList
        images =  ImageList.from_tensors(images, self.model.backbone.size_divisibility)

        return images, batched_inputs
    
    def get_features(self, images):
        images.tensor = images.tensor.to(self.model.device)
        features = self.model.backbone(images.tensor)

        return features
    
    def plot_features_random_img(self, images, features):

        image_index = np.random.randint(0, len(images))
        image = images[image_index]

        plt.imshow(cv2.resize(image, (images.tensor.shape[-2:][::-1])))
        plt.show()
        for key in features.keys():
            print(features[key].shape)
            plt.imshow(features[key][0,0,:,:].squeeze().to(self.model.device).detach().numpy(), cmap='jet')
            plt.show()

    def get_proposals(self, images, features):
        proposals, _ = self.model.proposal_generator(images, features)
        return proposals
    
    def get_box_features(self, features, proposals):
        features_list = [features[f] for f in ['p2', 'p3', 'p4', 'p5']]
        box_features = self.model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        box_features = self.model.roi_heads.box_head.flatten(box_features)
        box_features = self.model.roi_heads.box_head.fc1(box_features)
        box_features = self.model.roi_heads.box_head.fc_relu1(box_features)
        box_features = self.model.roi_heads.box_head.fc2(box_features)
        box_features = box_features.reshape(-1, 1000, 512)

        return box_features, features_list
    
    def get_prediction_logits(self, features_list, proposals):
        cls_features = self.model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        cls_features = self.model.roi_heads.box_head(cls_features)
        pred_class_logits, pred_proposal_deltas = self.model.roi_heads.box_predictor(cls_features)

        return pred_class_logits, pred_proposal_deltas
    
    def get_box_scores(self, pred_class_logits, pred_proposal_deltas, proposals):
        box2box_transform = Box2BoxTransform(weights=self.cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        smooth_l1_beta = self.cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA

        outputs = FastRCNNOutputs(
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta,
        )

        boxes = outputs.predict_boxes()
        scores = outputs.predict_probs()
        image_shapes = outputs.image_shapes

        return boxes, scores, image_shapes
    

    def get_output_boxes(self, boxes, batched_inputs, image_size):
        proposal_boxes = boxes.reshape(-1, 4).clone()
        scale_x, scale_y = (batched_inputs["width"] / image_size[1], batched_inputs["height"] / image_size[0])
        output_boxes = Boxes(proposal_boxes)

        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(image_size)

        return output_boxes

    def select_boxes(self, output_boxes, scores):
        test_score_thresh = self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        test_nms_thresh = self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        cls_prob = scores.detach()
        cls_boxes = output_boxes.tensor.detach().reshape(1000,80,4)
        max_conf = torch.zeros((cls_boxes.shape[0]))
        for cls_ind in range(0, cls_prob.shape[1]-1):
            cls_scores = cls_prob[:, cls_ind+1].to(self.model.device)
            det_boxes = cls_boxes[:,cls_ind,:].to(self.model.device)
            keep = np.array(nms(det_boxes, cls_scores, test_nms_thresh))
            max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
        keep_boxes = torch.where(max_conf >= test_score_thresh)[0]

        return keep_boxes, max_conf
    
    def filter_boxes(self, keep_boxes, max_conf, min_boxes, max_boxes):
        if len(keep_boxes) < min_boxes:
            keep_boxes = np.argsort(max_conf).numpy()[::-1][:min_boxes]
        elif len(keep_boxes) > max_boxes:
            keep_boxes = np.argsort(max_conf).numpy()[::-1][:max_boxes]
        return keep_boxes
    

    def get_visual_embeds(self, box_features, keep_boxes):
        return box_features[keep_boxes.copy()]

