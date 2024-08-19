from matplotlib import pyplot as plt


class TrainTracker:
    def __init__(self, config):
        self.cfg = config
        self.curr_epoch = 1
        self.curr_iter = 0
        self.last_iter_cp = 0
        self.running_loss = 0.0
        self.running_iter_loss = 0.0
        self.running_epoch_loss = 0.0
        self.training_loss = []
        self.lrs = []

    def reset_epoch(self, epoch_loss, epoch):
        self.training_loss.append(epoch_loss)
        self.curr_epoch = epoch
        self.running_epoch_loss = 0.0
        self.running_loss = 0.0

    def reset_iter(self):
        self.training_loss.append(self.running_iter_loss / self.cfg.TRAIN.save_iter)
        self.running_iter_loss = 0.0

    def plot_lrs(self, save_path):
        plt.plot(self.lrs)
        plt.xlabel("Iterations")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.savefig(save_path)
        plt.close()

    def plot_loss(self, save_path):
        plt.plot(self.training_loss)
        if self.cfg.TRAIN.epoch_based:
            plt.xlabel("Epochs")
        else:
            plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    pass
