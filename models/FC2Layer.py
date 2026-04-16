class FC2Layer(nn.Module):
    def __init__(
        self, input_size: int, input_channels: int, n_hidden: int, output_size: int
    ) -> None:
        """
        Simple MLP model

        :param input_size: number of pixels in the image
        :param input_channels: number of color channels in the image
        :param n_hidden: size of the hidden dimension to use
        :param output_size: expected size of the output (e.g. number of classes if you are in a classification task)
        """

        input_size=32*32
        input_channels=1
        n_hidden=20
        output_size=10
        
        super().__init__()
        self.network = nn.Sequential(
            # TODO define a linear NN made of 3 linear layers
            # N.B. do not forget ReLU!!
            nn.Linear(input_size * input_channels, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: batch of images with size [batch, 1, w, h]
        

        :returns: predictions with size [batch, output_size]
        """
        
        #TODO define the forward
        x=torch.flatten(x,start_dim=1)
        x=self.network(x)
        return x
