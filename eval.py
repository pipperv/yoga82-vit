import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import numpy as np
from tqdm import tqdm
import torch
from torch import nn

def model_analysis(model,dataloader,device='cuda',model_name='',dataset_name='Validation Dataset',conf_matrix=True,cmap="rainbow",figsize=25):
  # Acc. and Conf. Matriz function
  model.eval()
  y_batch_pred=[]
  y_batch_true=[]
  with torch.no_grad():
    for data in tqdm(dataloader):
      inputs = data[0].to(device)
      labels = data[1].to(device)
      outputs = model(inputs)
      #Compute accuracy
      smax = nn.LogSoftmax(dim=1)
      pred = torch.argmax(smax(outputs),dim=1)
      y_batch_pred.append(pred.tolist())
      y_batch_true.append(labels.tolist())
  y_pred = np.concatenate(y_batch_pred)
  y_true = np.concatenate(y_batch_true)
  acc = metrics.accuracy_score(y_true,y_pred,normalize=True)
  if dataset_name!=None:
    print('\n{} Accuracy for {}: {}'.format(model_name,dataset_name,acc))
  else:
    print('\n{} Accuracy: {}'.format(model_name,acc))

  #Confusion Matrix Plot
  if conf_matrix:
    conf_mat = metrics.confusion_matrix(y_true,y_pred,normalize='true')
    plt.figure(figsize=(figsize,figsize))
    cfm_plot = sns.heatmap(conf_mat,cmap=cmap,annot=True)
    if dataset_name!=None:
      plt.title('{} \n Confusion Matrix for {}'.format(model_name,dataset_name))
    else:
      plt.title('{} \n Confusion Matrix'.format(model_name))

  return acc