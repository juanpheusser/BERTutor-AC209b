

cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

ve = GetVisualEmbeddings(cfg_path, cuda=False)

batch_size = 10

train_data['batch'] = np.arange(0, len(train_data)) // batch_size
output_visual_embeddings = []

i = 0

for _, batch in tqdm(train_data.groupby('batch')):

  batch_visual_embeddings = ve.get_visual_embeddings(batch)
  torch.save(batch_visual_embeddings, 'drive/MyDrive/visual_embeds/visual_embeds_'+ str(i) + '.pt')
  i += 1
  del batch_visual_embeddings
  gc.collect()