import torch
from torch import nn
import torchvision 
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim
import tqdm
def topk_error(preds, labels, k):
    topk_preds = preds.topk(k, dim=1)[1]
    correct = topk_preds.eq(labels.view(-1, 1).expand_as(topk_preds))
    correct_k = correct.view(-1).float().sum(0, keepdim=True)
    error = 1 - correct_k.div_(labels.size(0))
    return error.item()
class ClassifierModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ClassifierModel, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

barlow_weight=torch.load("weight_dir/model_0.pt")
barlow_backbone = torchvision.models.resnet18(pretrained=False)
barlow_backbone=nn.Sequential(*list(barlow_backbone.children())[:-1])
barlow_backbone.load_state_dict(barlow_weight)
barlow_backbone.eval()
classifier_model = ClassifierModel(num_features=512, num_classes=10)

classification_criterion = nn.CrossEntropyLoss()
cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor()) 
dataloader = DataLoader(cifar10_train, batch_size=4, shuffle=True, num_workers=0)
# Define your optimizer for the classifier head (e.g., Adam, SGD, etc.)
classification_optimizer = optim.Adam(classifier_model.parameters(), lr=0.001)
num_epochs = 2
# Assuming you have a dataloader for your dataset
for epoch in range(num_epochs):  # Loop over the dataset multiple times
    running_loss = 0.0
    for i, data in tqdm.tqdm(enumerate(dataloader, 0)):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Zero the parameter gradients
        classification_optimizer.zero_grad()

        # Extract features from the pre-trained Barlow Twins model
        with torch.no_grad():
            barlow_features = barlow_backbone(inputs)
            barlow_features = torch.flatten(barlow_features, 1)
        # Forward pass through the classifier head
        classification_outputs = classifier_model(barlow_features)

        # Compute classification loss
        classification_loss = classification_criterion(classification_outputs, labels)

        # Backward pass and optimization
        classification_loss.backward()
        classification_optimizer.step()

        # Print statistics
        running_loss += classification_loss.item()
        if i % 2000 == 1999:  # Print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Fine-tuning')