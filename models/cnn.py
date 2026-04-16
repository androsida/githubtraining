class CNN(nn.Module):
    def __init__(self, input_size: int, input_channels: int, n_feature: int, output_size: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, n_feature, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_feature, n_feature, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(n_feature, n_feature, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(n_feature * 8 * 8, output_size)
        self.fc2 = nn.Linear(output_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        return_conv1: bool = False,
        return_conv2: bool = False,
        return_conv3: bool = False
    ) -> torch.Tensor:

        x = self.conv1(x)
        x = F.relu(x)
        if return_conv1:
            return x
        x = self.pool(x)          

        x = self.conv2(x)
        x = F.relu(x)
        if return_conv2:
            return x

        x = self.conv3(x)
        x = F.relu(x)
        if return_conv3:
            return x
        x = self.pool(x)          # 16 -> 8

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
