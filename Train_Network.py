import torch
import AudioNetwork as net 
import torch.optim as optim
import matplotlib.pyplot as plt
import AudioNetworkTrainer as trn
import esc
import os

def show_results(orientation='horizontal', accuracy_bottom=None, loss_top=None, **histories):
    if orientation == 'horizontal':
        f, ax = plt.subplots(1, 2, figsize=(16, 5))
    else:
        f, ax = plt.subplots(2, 1, figsize=(16, 16))
    for i, (name, h) in enumerate(histories.items()):
        if len(histories) == 1:
            ax[0].set_title("Best test accuracy: {:.2f}% (train: {:.2f}%)".format(
                max(h['test_accuracy']) * 100,
                max(h['train_accuracy']) * 100
            ))
        else:
            ax[0].set_title("Accuracy")
        ax[0].plot(h['train_accuracy'], color='C%s' % i, linestyle='--', label='%s train' % name)
        ax[0].plot(h['test_accuracy'], color='C%s' % i, label='%s test' % name)
        ax[0].set_xlabel('epochs')
        ax[0].set_ylabel('accuracy')
        if accuracy_bottom:
            ax[0].set_ylim(bottom=accuracy_bottom)
        ax[0].legend()

        if len(histories) == 1:
            ax[1].set_title("Minimal train loss: {:.4f} (test: {:.4f})".format(
                min(h['train_loss']),
                min(h['test_loss'])
            ))
        else:
            ax[1].set_title("Loss")
        ax[1].plot(h['train_loss'], color='C%s' % i, linestyle='--', label='%s train' % name)
        ax[1].plot(h['test_loss'], color='C%s' % i, label='%s test' % name)
        ax[1].set_xlabel('epochs')
        ax[1].set_ylabel('loss')
        if loss_top:
            ax[1].set_ylim(top=loss_top)
        ax[1].legend()

    plt.show()

dataset = esc.ESC50(root= os.path.abspath(os.getcwd()),download =True)
total_count = len(dataset)
train_count = int(0.8 * total_count)
#valid_count = int(0.1 * total_count)
valid_count = 0
test_count = int(0.2 * total_count)
#test_count = total_count - train_count - valid_count
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(   # Change random splits to Fold splits
    dataset, (train_count, valid_count, test_count)
)
network = net.AudioNetwork()
optimizer = optim.Adam(network.parameters(), lr=0.001, amsgrad=True)
trainer = trn.AudioNetworkTrainer(train_dataset, test_dataset, batch_size=32)
history = trainer.train(network, optimizer, n_epochs=300)
show_results(model=history)
