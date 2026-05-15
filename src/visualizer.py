import matplotlib.pyplot as plt

def plot_training_results(metrics_list):
    # Slice data to show only up to epoch 15
    data = metrics_list[:20]
    epochs = range(1, len(data) + 1)
    
    # Extract loss
    train_loss = [m['train_loss'] for m in data]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Loss (Full scale for visibility)
    ax1.plot(epochs, train_loss, 'r-', label='Train Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_xlim(1, 20)
    ax1.legend()

    # Plot 2: Accuracies (Modified axes)
    acc_keys = ['train_acc', 'val_acc', 'syn_acc', 'val_base']
    for key in acc_keys:
        ax2.plot(epochs, [m[key] for m in data], label=key)
        
    ax2.set_title('Accuracy Metrics')
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(1, 20)
    ax2.set_ylim(0.4, 0.7)  # Your requested range
    ax2.legend()

    plt.tight_layout()
    plt.show()
