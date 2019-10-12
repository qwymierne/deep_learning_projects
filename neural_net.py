import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

TRAIN_SET = 'fruits-360/Training'
TEST_SET = 'fruits-360/Test'
NUMBER_OF_CLASSES = 95
LEARNING_RATE = 0.0005

INPUT_CHANNELS = 3
CONV1_FILTERS = 20
CONV2_FILTERS = 30
CONV3_FILTERS = 40

LINEAR1_INPUT = 10 * 10 * CONV3_FILTERS
LINEAR1_OUTPUT = 250
LINEAR2_OUTPUT = NUMBER_OF_CLASSES


class BatchNormalization(nn.Module):

    def __init__(self, num_of_params, training, momentum=0.1):
        super(BatchNormalization, self).__init__()
        self.beta = nn.Parameter(torch.randn(num_of_params))
        self.gamma = nn.Parameter(torch.randn(num_of_params))
        self.training = training
        self.momentum = momentum

        self.register_buffer('running_mean', torch.zeros(num_of_params))
        self.register_buffer('running_var', torch.ones(num_of_params))

    def forward(self, input):

        if input.dim() == 2:

            means = torch.mean(input, 0)
            variances = torch.mean((input - means.view([1, -1])) ** 2, 0)

            if self.training:
                input_shifted = (input - means.view([1, -1])) / torch.sqrt(variances.view([1, -1]) + 1e-5)
                self.running_mean = self.running_mean * self.momentum + means * (1.0 - self.momentum)
                self.running_var = self.running_var * self.momentum + variances * (1.0 - self.momentum)
            else:
                input_shifted = (input - self.running_mean.view([1, -1])) /\
                                torch.sqrt(self.running_var.view([1, -1]) + 1e-5)

            out = self.gamma.view([1, -1]) * input_shifted + self.beta.view([1, -1])

        if input.dim() == 4:

            means = torch.mean(input, [0, 2, 3])
            variances = torch.mean((input - means.view([1, -1, 1, 1])) ** 2, [0, 2, 3])

            if self.training:
                input_shifted = (input - means.view([1, -1, 1, 1])) / torch.sqrt(variances.view([1, -1, 1, 1]) + 1e-5)
                self.running_mean = self.running_mean * self.momentum + means * (1.0 - self.momentum)
                self.running_var = self.running_var * self.momentum + variances * (1.0 - self.momentum)
            else:
                input_shifted = (input - self.running_mean.view([1, -1, 1, 1])) \
                                / torch.sqrt(self.running_var.view([1, -1, 1, 1]) + 1e-5)

            out = self.gamma.view([1, -1, 1, 1]) * input_shifted + self.beta.view([1, -1, 1, 1])

        return out


def load_dataset(data_path, transformer, shuffle=True):
    image_folder = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transformer
    )
    loader = torch.utils.data.DataLoader(
        image_folder,
        batch_size=64,
        num_workers=0,
        shuffle=shuffle
    )
    return loader


def evaluate_per_class(model, test_loader, classes, device):

    class_correct = [0. for _ in range(len(classes))]
    class_total = [0. for _ in range(len(classes))]
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(c)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    return {classes[i]: class_correct[i] / class_total[i] for i in range(len(classes))}


def train_model(model, criterion, optimizer, num_of_epochs, train_loader, device, test_loader=None):

    for epoch in range(num_of_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print(f'[{epoch}, {i + 1}] loss: {running_loss / 100}')
                running_loss = 0.0

        if test_loader:
            print(f'Evaluation on test set after epoch {epoch}')
            eval_score = evaluate_model(model, test_loader, device)
            print(f'Accuracy of the network on test images: {eval_score}')

    print('Finished Training')


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, CONV1_FILTERS, 3)
        self.conv1_bn = BatchNormalization(CONV1_FILTERS, training=self.training)
        self.conv2 = nn.Conv2d(CONV1_FILTERS, CONV2_FILTERS, 3)
        self.conv2_bn = BatchNormalization(CONV2_FILTERS, training=self.training)
        self.conv3 = nn.Conv2d(CONV2_FILTERS, CONV3_FILTERS, 3)
        self.conv3_bn = BatchNormalization(CONV3_FILTERS, training=self.training)
        self.fc1 = nn.Linear(LINEAR1_INPUT, LINEAR1_OUTPUT)
        self.fc1_bn = BatchNormalization(LINEAR1_OUTPUT, training=self.training)
        self.fc2 = nn.Linear(LINEAR1_OUTPUT, LINEAR2_OUTPUT)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = nn.MaxPool2d(2, 2)(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = nn.MaxPool2d(2, 2)(x)
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)
        x = nn.MaxPool2d(2, 2)(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def main():
    device = torch.device("cuda")

    transformer = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1))])

    train_loader = load_dataset(TRAIN_SET, transformer)
    test_loader = load_dataset(TEST_SET, transformer, shuffle=False)

    classes = tuple(sorted(os.listdir(TRAIN_SET)))
    net = Net()
    net.to(device)
    print(net)

    train_model(
        net,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(net.parameters(), lr=LEARNING_RATE),
        num_of_epochs=3,
        train_loader=train_loader,
        device=device,
        test_loader=test_loader
    )

    torch.save(net, 'trained_model.h5')

    eval_score = evaluate_model(net, test_loader, device)
    print(f'Accuracy of the network on test images: {eval_score}')

    eval_per_class = evaluate_per_class(net, test_loader, classes, device)
    for class_ in eval_per_class:
        print(f'Accuracy of the network on {class_}: {eval_per_class[class_]}')


if __name__ == '__main__':
    main()

