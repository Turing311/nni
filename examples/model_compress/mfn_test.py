import torch
from models.mfn import MfnModel
import torch.nn.functional as F
from torchvision import datasets, transforms

def train(model, device, train_loader, optimizer, sparse_bn=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        # L1 regularization on BN layer
        if sparse_bn:
            updateBN(model)
        optimizer.step()
        if batch_idx % 300 == 299:
            print('{:2.0f}%  Loss {}'.format(100 * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100 * correct / len(test_loader.dataset)

    print('Loss: {}  Accuracy: {}%)\n'.format(
        test_loss, acc))
    return acc

if __name__ == '__main__':
    
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                         transform=transforms.Compose([
#                             transforms.Pad(4),
                             transforms.Resize(128),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor()
                         ])),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor()
        ])),
        batch_size=128, shuffle=False)

    model = MfnModel("./models/mfn.npy")
    model.freeze()
    print(model.classifier)
    model.to(device)
    # Train the base VGG-19 model
    if True:
        print('=' * 10 + 'Train the unpruned base model' + '=' * 10)
        epochs = 10
        optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        for epoch in range(epochs):
            if epoch in [epochs * 0.5, epochs * 0.75]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
            print("epoch {}".format(epoch))
            train(model, device, train_loader, optimizer, False)
            test(model, device, test_loader)
        torch.save(model.state_dict(), 'mfn_retrain.pth')