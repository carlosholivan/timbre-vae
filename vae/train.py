from tqdm import tqdm
import torch

# Our modules
from vae import loss


def fit(model, dataloader, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset)/dataloader.batch_size)):
        data = data['input']
        data = data.to(device)

        # backprop
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        total_loss, kld = loss.loss_function(bce_loss, mu, logvar)

        kld += kld.item()

        running_loss += total_loss.item()
        total_loss.backward()

        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    kld = kld/len(dataloader.dataset)
    return train_loss, kld


def validate(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset)/dataloader.batch_size)):
            data = data['input']
            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            total_loss, kld = loss.loss_function(bce_loss, mu, logvar)

            running_loss += total_loss.item()

            kld += kld.item()

    val_loss = running_loss/len(dataloader.dataset)
    kld = kld/len(dataloader.dataset)
    return val_loss, kld
