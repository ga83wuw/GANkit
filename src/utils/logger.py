# src/utils/logger.py

class Logger:
    """
    Logger class for tracking losses and scores.
    """
    def __init__(self, num_epochs, num_batches):
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.epoch = 0
        self.batch = 0

    def log(self, d_loss, g_loss, real_score, fake_score, epoch, batch):
        """
        Logs the training progress.

        Args:
            d_loss (float): Discriminator loss.
            g_loss (float): Generator loss.
            real_score (float): Real images score.
            fake_score (float): Fake images score.
            epoch (int): Current epoch.
            batch (int): Current batch.
        """
        self.epoch = epoch
        self.batch = batch
        if (batch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Step [{batch+1}/{self.num_batches}], '
                  f'D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}, '
                  f'Real Score: {real_score:.4f}, Fake Score: {fake_score:.4f}')

