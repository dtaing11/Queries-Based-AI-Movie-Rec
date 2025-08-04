from torch import nn

class MovieEncoder(nn.Module):
    def __init__(self, movie_text_dim=768):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(movie_text_dim, 128),
            nn.ReLU(),
        )
    def forward(self, movieEmbedding ):
        return self.fc(movieEmbedding)