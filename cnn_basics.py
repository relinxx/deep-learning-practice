import torch
import torch.nn as nn
import torch.optim as optim


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_one_epoch(model, optimizer, criterion, x_batch, y_batch):
    model.train()
    optimizer.zero_grad()

    logits = model(x_batch)
    loss = criterion(logits, y_batch)

    loss.backward()
    optimizer.step()

    predictions = logits.argmax(dim=1)
    accuracy = (predictions == y_batch).float().mean().item()
    return loss.item(), accuracy


def main():
    torch.manual_seed(42)

    batch_size = 32
    num_classes = 10

    # Fake grayscale images: [batch, channels, height, width]
    x_batch = torch.randn(batch_size, 1, 28, 28)
    y_batch = torch.randint(0, num_classes, (batch_size,))

    model = SimpleCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 6):
        loss, acc = train_one_epoch(model, optimizer, criterion, x_batch, y_batch)
        print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
