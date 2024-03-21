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
mhcnn.run('path/to/config.toml')
```

## Advanced Functionality
```py
# define the variables
datapath = '/path/to/folder'
classes = 10
batch_size = 16
lr = 0.0001
in_channels = 3
num_epochs = 100
conv_layers = [128, 256, 256, 256, 256]
linear_layers = [256, 256, 128]

# load the dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainloader, testloader = mhcnn.get_dataloaders(
    datapath, classes=classes,
    device=device, batch_size=batch_size
)

# train the model
model = mhcnn.CNN(
    conv_layers, 
    linear_layers, 
    in_channels=in_channels, 
    out_channels=classes
)
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
