# Multi Head CNN
This is a module to explore the multi head CNNs.

## Setup
This package can be installed using:
```py
!pip install git+https://github.com/timowendner/MultiHeadCNN
```
Then we can import it in and run it as following:
```py
import mhcnn

datapath = '/path/to/folder'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = 10
batch_size = 16
lr = 0.0001
num_epochs = 100
layers = [32, 64, 64, 64]
trainloader, testloader = mhcnn.get_dataloaders(
    datapath, classes=classes,
    device=device, batch_size=batch_size
)

model = mhcnn.CNN(layers)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
result = mhcnn.Result(classes)
model, optimizer = mhcnn.train_network(
    model, optimizer, criterion,
    trainloader, testloader, result,
    num_epoch=num_epochs
)
```
