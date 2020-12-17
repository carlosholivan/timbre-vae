import torch
from torch import optim, nn
from torch.utils.data import DataLoader

# Our modules
import configs
import data_utils
import model
import train


# Compute inputs from audio
#dataset_path = '/media/carlos/FILES/INVESTIGACION/Datasets/London Philarmonic Orchestra/'
#data_utils.store_inputs(dataset_path)


# Dataloader
train_dataset = data_utils.AudioDataset(data_path=configs.ParamsConfig.DATA_PATH)
train_dataloader = DataLoader(train_dataset, batch_size=configs.ParamsConfig.BATCH_SIZE, num_workers=0)

val_dataset = data_utils.AudioDataset(data_path=configs.ParamsConfig.DATA_PATH)
val_dataloader = DataLoader(val_dataset, batch_size=configs.ParamsConfig.BATCH_SIZE, num_workers=0)

print('Number of files in the training dataset:', len(train_dataset))
print('Number of files in the training dataset:', len(val_dataset))

# Model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.VAE().to(device)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)

optimizer = optim.Adam(model.parameters(), lr=configs.ParamsConfig.LEARNING_RATE)
criterion = nn.BCELoss(reduction='sum')  # reconstruction loss


train_loss = []
val_loss = []
for epoch in range(configs.ParamsConfig.NUM_EPOCHS):
    print(f"Epoch {epoch+1} of {configs.ParamsConfig.NUM_EPOCHS}")
    train_epoch_loss = train.fit(model, train_dataloader)
    val_epoch_loss = train.validate(model, val_dataloader)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")

    # Save trained model every 5 epochs
    if epoch % 5 == 0:
        torch.save(model.state_dict(), "./trained_models/saved_model_" + str(epoch) + "epochs.bin")
