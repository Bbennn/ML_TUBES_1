import matplotlib.pyplot as plt

def plot_training_results(loss_results):
    """
    Plots training and validation loss from multiple experiments.

    Args:
        loss_results: List of tuples (train_loss, val_loss), where val_loss can be empty.
    """
    plt.figure(figsize=(7, 5))

    for i, (train_loss, val_loss) in enumerate(loss_results):
        plt.plot(train_loss, label=f'Train {i+1}')
        if val_loss:  # Only plot val_loss if it's not empty
            plt.plot(val_loss, label=f'Val {i+1}', linestyle='dashed')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Training - Variasi Width')
    plt.legend(loc='best', fontsize=8)
    plt.show()