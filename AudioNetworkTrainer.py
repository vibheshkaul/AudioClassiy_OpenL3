import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

class AudioNetworkTrainer:
    def __init__(self, train_dataset, test_dataset, batch_size=128):
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        wandb.login()
        wandb.init(project='AudioClassify', entity='vibheshk', group="OpenL3",
               name="Audio benchmark ESC Random-split")

    def train(self, model, optimizer, loss_fn=torch.nn.functional.cross_entropy, n_epochs=300):
        self.logs = {'train_loss': [], 'test_loss': [], 'train_accuracy': [], 'test_accuracy': []}
        model = model.to(self.device)
        wandb.watch(model,self.criterion, log="all", log_freq = 10)

        correct, numel = 0, 0
        for e in range(1, n_epochs + 1):
            model.train()
            for x,sr,y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                output = model(x)
                y_pred = torch.argmax(output, dim=1)
                correct += torch.sum(y_pred == y).item()
                numel += self.batch_size
                loss = self.criterion(output, y)
                wandb.log({"epoch": e, "loss": loss})
                loss.backward()
                optimizer.step()

            self.logs['train_loss'].append(loss.item())
            self.logs['train_accuracy'].append(correct / numel)
            wandb.log({"train_accuracy": correct / numel})
            correct, numel = 0, 0

            model.eval()
            with torch.no_grad():
                for x_test, sr,y_test in self.test_loader:
                    x_test = x_test.to(self.device)
                    y_test = y_test.to(self.device)
                    output = model(x_test).to(self.device)
                    y_pred = torch.argmax(output, dim=1)
                    y_pred = y_pred.to(self.device)
                    correct += torch.sum(y_pred == y_test).item()
                    numel += self.batch_size
                loss = self.criterion(output, y_test)

            self.logs['test_loss'].append(loss.item())
            self.logs['test_accuracy'].append(correct / numel)
            wandb.log({"test_accuracy": correct / numel})
            correct, numel = 0, 0
            print("Epoch --- ",e)
        return self.logs

