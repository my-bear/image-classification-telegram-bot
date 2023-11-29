import os
import json
import argparse
import torch
import matplotlib.pyplot as plt

from glob import glob


# Parser for loading the user's path
parser = argparse.ArgumentParser(description = 'Setting download paths.')
parser.add_argument('--model_load', type = str, default = None, help = 'the path to loading the model')
args = parser.parse_args()

with open(os.path.join(os.path.dirname(__file__), \
                      "destination_path.json")) as f:
   destination = json.load(f)

if args.model_load is not None:
   destination['path_model_load'] = args.model_load

# loading the model checkpoint
checkpoint = torch.load(glob(os.path.abspath(os.path.join(\
                      destination['path_model_load'], "*.pth")))[-1])
train_loss = checkpoint['avg_train_losses']
valid_loss = checkpoint['avg_valid_losses']

fig = plt.figure(figsize = (10,8))
plt.plot(range(1,len(train_loss) + 1), train_loss, label = 'Training Loss')
plt.plot(range(1,len(valid_loss) + 1), valid_loss, label = 'Validation Loss')

# find position of lowest validation loss
minposs = valid_loss.index(min(valid_loss)) + 1 
plt.axvline(minposs, linestyle = '--', color = 'r',label = 'Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 0.5) # consistent scale
plt.xlim(0, len(train_loss) + 1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig(os.path.abspath(os.path.join( \
    destination['reports_fig'],'loss_plot.png')), bbox_inches = 'tight')
