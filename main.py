import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
import os

from Audio_Dataset import AudioDataset
from ssast.src.models import ASTModel

SSAST_PATH = os.path.join(os.getcwd(), 'ssast', 'pretrained_model', 'SSAST-Base-Patch-400.pth')

# ---------------------------------------------------------------------------------
# hyperparameters
# ---------------------------------------------------------------------------------
LEARNING_RATE = 0.001
BATCH_SIZE = 10
EPOCHS = 5

# helper vars
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

root_path = os.path.join(os.getcwd(),'LibriSpeech')

new_data_path = os.path.join(root_path, 'preprocessed')

AUGMENTATION_FILE = os.path.join(new_data_path, 'augmentation.csv')


#########################
# Helper functions      #
#########################

def train_one_epoch(model, dataloader, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.squeeze().to(DEVICE)
        
        # calculate matrix transpose to match correct input dimension for ssast
        X_permuted = X.permute(0, 2, 1).to(DEVICE)

        # forward
        pred = model(X_permuted, task='ft_avgtok')
        
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # compute metrics
    train_loss /= num_batches
    correct /= size
    print(
        f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")


def evaluate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.squeeze().to(DEVICE)
            pred = model(X, task='ft_cls')
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # compute metrics
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")




def main():
    
    ####
    # Data
    ####
        
    dataset = AudioDataset(AUGMENTATION_FILE)
    
    train_data, test_data = random_split(dataset, (0.8, 0.2))
    
    # create samplers
    train_sampler = RandomSampler(train_data)
    test_sampler = SequentialSampler(test_data)

    # Create data loaders.
    data_loader_train = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
    )
    data_loader_test = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        sampler=test_sampler,
    )
    
    
    ####
    # Model
    ####
    
    input_tdim = dataset.input_tdim  # fine-tuning data length can be different with pretraining data length
    label_dim = dataset.label_dim

    model = ASTModel(label_dim=label_dim,
                fshape=16, tshape=16, fstride=10, tstride=10,
                input_fdim=128, input_tdim=input_tdim, model_size='base',
                pretrain_stage=False, load_pretrained_mdl_path=SSAST_PATH).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_fn = torch.nn.CrossEntropyLoss()
    
    for e in range(0, EPOCHS):
        print(f"************\nTrain Epoch {e}\n************\n")
        train_one_epoch(model=model, dataloader=data_loader_train, loss_fn=loss_fn, optimizer=optimizer)
    
    evaluate(model=model, dataloader=data_loader_test, loss_fn=loss_fn)
    



if __name__ == '__main__':
    main()
    
    
    
"""
    test_input = torch.zeros([10, input_tdim, 128])
    prediction = model(test_input, task='ft_avgtok')
    # output should in shape [batch_size, label_dim]
    print(prediction.shape)

    print(prediction)
    # calculate the loss, do back propagate, etc
"""
