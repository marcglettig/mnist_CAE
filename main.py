import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, conditional=0):
        self.num_labels = 0
        super(VAE, self).__init__()
        x_dim_ = x_dim
        z_dim_ = z_dim

        self.conditional = conditional
        if self.conditional > 0:
            self.num_labels = 10
            z_dim_ = z_dim + self.num_labels
        if self.conditional > 1:
            x_dim_ = x_dim + self.num_labels
        # encoder part
        self.fc1 = nn.Linear(x_dim_, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)

        # decoder part
        self.fc4 = nn.Linear(z_dim_, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h))

    def forward(self, x, c=None):
        x = x.view(-1, 784)
        c = idx2onehot(c, n=10)
        if self.conditional > 1:
            x = torch.cat((x, c), dim=-1)
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        if self.conditional > 0:
            z = torch.cat((z, c), dim=-1)
        return self.decoder(z), mu, log_var

    def enc(self, x, c=None):
        x = x.view(-1, 784)
        c = idx2onehot(c, n=10)
        if self.conditional > 1:
            x = torch.cat((x, c), dim=-1)
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        if abs(z[:, 0]).max() > 1000:
            print(abs(z[:, 0]).max())
            print('z: ', z[abs(z[:, 0]).argmax(), 0])
            print('log_var: ', log_var[abs(z[:, 0]).argmax()])
            print('mu: ', mu[abs(z[:, 0]).argmax()])
            print('x: ', x[abs(z[:, 0]).argmax()])
        if abs(z[:, 1]).max() > 1000:
            print(abs(z[:, 1]).max())
            print('z: ', z[abs(z[:, 1]).argmax(), 1])
            print('log_var: ', log_var[abs(z[:, 1]).argmax()])
            print('mu: ', mu[abs(z[:, 1]).argmax()])
            print('x: ', x[abs(z[:, 1]).argmax()])
        return z


def idx2onehot(idx, n):
    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    return onehot


def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, y) in enumerate(train_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            y = y.cuda()
        optimizer.zero_grad()
        recon_batch, mu, log_var = vae(data, y)
        loss = loss_function(recon_batch, data, mu, log_var)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    return train_loss / len(train_loader.dataset)


def test():
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for data, y in test_loader:
            if torch.cuda.is_available():
                data = data.cuda()
            recon, mu, log_var = vae(data, y)

            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


def enc_plot(y, data):
    vae.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            data = data.cuda()
            y = y.cuda()
        print(data.type())
        z = vae.enc(data, y)
    return z


def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


if __name__ == '__main__':
    bs = 100
    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

    # build model
    conditional_inp = input('Conditional? ')
    conditional = 2 if conditional_inp.lower() in ['y', 't', 'yes', 'true'] else 0
    if conditional_inp.lower() in ['semi', 's']:
        conditional = 1
    vae = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=2, conditional=conditional)
    if torch.cuda.is_available():
        vae.cuda()

    optimizer = optim.Adam(vae.parameters())

    train_loss = []
    test_loss = []

    epochs = int(input('Num epochs? '))

    for epoch in range(1, epochs+1):
        train_loss.append(train(epoch))
        test_loss.append(test())

    plt.plot(np.arange(1, epochs+1), np.array(train_loss), np.array(test_loss))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    if conditional > 1:
        plt.title('conditional')
        plt.savefig('samples/loss_cond_' + '.png')
    elif conditional > 0:
        plt.title('semi-conditional')
        plt.savefig('samples/loss_semi_cond_' + '.png')
    else:
        plt.title('un-conditional')
        plt.savefig('samples/loss_un_cond_' + '.png')

    df = pd.DataFrame(test_dataset.data.view(-1, 28*28)).astype('int')
    df['label'] = pd.DataFrame(test_dataset.targets).astype('int')
    df = df.groupby('label').head(100)
    y = torch.tensor(df['label'].values)
    X = torch.tensor(df[df.columns[:-1]].values).float()
    z = enc_plot(y, X)
    df['x'] = pd.DataFrame(z[:, 0]).astype('float')
    df['y'] = pd.DataFrame(z[:, 1]).astype('float')
    g = sns.lmplot(x='x', y='y', hue='label', data=df, fit_reg=False, legend=True)
    if conditional > 1:
        g.set(title='Conditional')
        g.savefig('samples/latent_cond_' + '.png')
    elif conditional > 0:
        g.set(title='Semi-conditional')
        g.savefig('samples/latent_semi_cond_' + '.png')
    else:
        g.set(title='Un-conditional')
        g.savefig('samples/latent_' + '.png')

    with torch.no_grad():
        z = torch.load('samples/z.pt')
        c = idx2onehot(torch.load('samples/c.pt'), 10)
        if conditional > 0:
            z = torch.cat((z, c), dim=-1)
        sample = vae.decoder(z)
        if torch.cuda.is_available():
            z = z.cuda()
            sample = sample.cuda()
        if conditional > 1:
            save_image(sample.view(64, 1, 28, 28), 'samples/sample_cond_' + '.png')
        elif conditional > 0:
            save_image(sample.view(64, 1, 28, 28), 'samples/sample_semi_cond_' + '.png')
        else:
            save_image(sample.view(64, 1, 28, 28), 'samples/sample_' + '.png')



