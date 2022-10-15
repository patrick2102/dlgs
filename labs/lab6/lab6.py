#!/usr/bin/env python
# coding: utf-8

# ## Example to load the network, sample 4 levels at random from the latent space and then plot them using matplotlib.
# 

# In[2]:


#from google.colab import drive
#drive.mount('/content/drive')


# In[ ]:


#!cp 'path/to/file' 'path/to/drive'


# In[3]:


#import os
#os.chdir("drive/My Drive/exercise_DL_pcg")


# In[4]:


#get_ipython().system('ls')


# In[5]:


import os
import torch
import matplotlib.pyplot as plt

from vae_mario import VAEMario
from plotting_utilities import plot_decoded_level


# In[6]:


# Loading the model
#model_name = "mario_vae_zdim_2_overfitted"
#model_name = "mario_vae_zdim_2_epoch_10"
#model_name = "mario_vae_zdim_2_epoch_20"
#model_name = "mario_vae_zdim_2_epoch_100"
#model_name = "mario_vae_zdim_2_epoch_200"
model_name = "mario_vae_zdim_2_final"
z_dim = 2
vae = VAEMario(z_dim=z_dim)
vae.load_state_dict(torch.load(f"./models/{model_name}.pt"))


# In[7]:


# Sampling random zs
zs = 2.5 * torch.randn((4, z_dim))


# In[8]:


# Getting levels from them using the decoder
levels = vae.decode(zs)
# print(levels.shape)
level_imgs = [plot_decoded_level(level) for level in levels]


# In[10]:


# Plotting
"""
_, axes = plt.subplots(1, 4, figsize=(7 * 4, 7))
for level_img, ax in zip(level_imgs, axes):
    ax.imshow(level_img)
    ax.axis("off")

plt.tight_layout()
plt.show()
"""


# # Latent Variable Evolution Lab
# 
# 
# ## Sample from the VAE and then implement some search algorithm of choise to search the latent space for a particular level (e.g. one with many ground tiles)


