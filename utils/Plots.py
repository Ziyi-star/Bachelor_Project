import pandas as pd  
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import tensorflow as tf 
from sklearn.metrics import confusion_matrix, classification_report


def plot_accelerometer_data(df, name):
    """
    Plot Acc-X, Acc-Y, and Acc-Z for handlebar accelerometer data over time.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'Acc-X', 'Acc-Y', 'Acc-Z' columns with a time-based index.
    """
    df['NTP'] = pd.to_datetime(df['NTP'])
    df.set_index('NTP', inplace=True)
    plt.figure(figsize=(14, 7), dpi=300)
    plt.title(name)
    plt.plot(df.index, df['Acc-X'], label='Acc-X', zorder=3)
    plt.plot(df.index, df['Acc-Y'], label='Acc-Y', zorder=2)
    plt.plot(df.index, df['Acc-Z'], label='Acc-Z', zorder=1)
    plt.legend()
    plt.grid()
    # Rotate date labels
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation=45)
    # Get the current axes and set major ticks every 120 seconds
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.SecondLocator(interval=120))
    plt.xlabel('Time')
    plt.ylabel('Acceleration (m/s^2)')
    plt.show()


def plot_anomaly_reconstruction_loss(normal_data, abnormal_data, model, threshold, y_limit=1.5):
    """
    Plot anomaly detection results showing reconstruction loss and detected anomalies.
    
    Parameters:
    -----------
    normal_data : numpy.ndarray
        Scaled normal data samples
    abnormal_data : numpy.ndarray
        Scaled abnormal data samples
    model : tensorflow.keras.Model
        Trained model, like an autoencoder or LSTM autoencoder
    threshold : float
        Threshold value for anomaly detection
    y_limit : float, optional
        Y-axis limit for the plot (default=1.5)
    
    Returns:
    --------
    tuple
        (reconstruction_loss_np, anomaly_indices) containing the reconstruction
        loss values and indices where anomalies were detected
    """
    # Combine normal and abnormal data
    all_test_data = np.vstack([normal_data, abnormal_data])
    
    # Get reconstructions and calculate loss
    all_reconstructions = model.predict(all_test_data)
    reconstruction_loss = tf.keras.losses.mae(all_reconstructions, all_test_data)
    reconstruction_loss_np = reconstruction_loss.numpy().flatten()
    
    # Find anomaly indices
    anomaly_indices = np.where(reconstruction_loss_np > threshold)[0]
    
    plt.figure(figsize=(15, 6))
    
    # Plot reconstruction loss
    plt.plot(range(len(reconstruction_loss_np)), reconstruction_loss_np, 
             'b-', linewidth=0.5, label='Reconstruction Loss')
    
    # Plot anomalies
    plt.vlines(x=anomaly_indices, 
              ymin=threshold, 
              ymax=reconstruction_loss_np[anomaly_indices],
              color='red',
              linewidth=1,
              alpha=0.5)
    
    # Add threshold line
    plt.axhline(y=threshold, color='green', linestyle='--', label='Threshold')
    
    # Customize the plot
    plt.xlabel('Sequence Index', fontsize=14)
    plt.ylabel('Reconstruction Loss (MAE)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, y_limit)
    plt.grid(True, alpha=0.3)
    plt.legend(['Reconstruction Loss', 'Anomalies', 'Threshold'], fontsize=14)
    
    plt.show()
    
    return reconstruction_loss_np, anomaly_indices

def plot_confusion_matrix(y_true, y_pred):
    """
    Creates and displays a confusion matrix visualization for binary classification results.
    
    Parameters:
        y_true (array-like): Ground truth labels (0 for normal, 1 for abnormal)
        y_pred (array-like): Predicted labels from the model
    
    Displays:
        - Confusion matrix heatmap with color-coded cells
        - Cell values showing counts of predictions
        - Classification report with precision, recall, and F1-score
    """
    confusion_mat = confusion_matrix(y_true, y_pred)
    # Convert to percentages
    confusion_mat_percent = (confusion_mat.astype('float') / 
                           confusion_mat.sum(axis=1)[:, np.newaxis] * 100)
    
    plt.imshow(confusion_mat, cmap=plt.cm.Blues)
    #plt.colorbar()
    
    # Add labels with binary classes
    labels = ['Normal', 'Abnormal']
    plt.xticks([0, 1], labels,fontsize=14)
    plt.yticks([0, 1], labels,fontsize=14)
    plt.ylabel('True Label',fontsize=14)
    plt.xlabel('Predicted Label',fontsize=14)
    
    # Add numbers to cells
    threshold = confusion_mat.max() / 2.
    # Add percentage values to cells
    threshold = confusion_mat_percent.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f'{confusion_mat_percent[i, j]:.1f}%',
                    ha="center", va="center",
                    fontsize=14,
                    color="white" if i==0 and j==0 else "black")
    
    plt.show()
    
    # Print classification report with binary labels
    print(classification_report(y_true, y_pred, 
                              target_names=['Normal (0)', 'Abnormal (1)'],
                              digits=3))
    

def plt_optimal_threshold(normal_losses, abnormal_losses):
    """
    Calculate and visualize optimal threshold for autoencoder/lstm using precision, recall, and F1 scores.
    
    Parameters:
    -----------
    normal_losses : tf.Tensor
        Reconstruction losses for normal data
    abnormal_losses : tf.Tensor
        Reconstruction losses for abnormal data
        
    Returns:
    --------
    tuple
        (best_threshold, best_f1_score)
    """
    def calculate_metrics(threshold, normal_losses, abnormal_losses):
        normal_losses = normal_losses.numpy()
        abnormal_losses = abnormal_losses.numpy()
        
        predictions_normal = (normal_losses > threshold).astype(int)
        predictions_abnormal = (abnormal_losses > threshold).astype(int)
        
        tp = np.sum(predictions_abnormal == 1)
        fp = np.sum(predictions_normal == 1)
        fn = np.sum(predictions_abnormal == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1

    # Generate threshold values
    min_loss = min(np.min(normal_losses.numpy()), np.min(abnormal_losses.numpy()))
    max_loss = max(np.max(normal_losses.numpy()), np.max(abnormal_losses.numpy()))
    thresholds = np.linspace(min_loss, max_loss, 100)
    
    # Calculate metrics for each threshold
    metrics = [calculate_metrics(t, normal_losses, abnormal_losses) for t in thresholds]
    precisions, recalls, f1_scores = zip(*metrics)
    
    # Find optimal threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(thresholds, precisions, 'b-', label='Precision', alpha=0.7)
    plt.plot(thresholds, recalls, 'g-', label='Recall', alpha=0.7)
    plt.plot(thresholds, f1_scores, 'r-', label='F1 Score', linewidth=2)
    plt.plot(best_threshold, best_f1, 'r*', markersize=15, 
             label=f'Best F1: {best_f1:.3f} at {best_threshold:.3f}')
    
    plt.xlabel('Threshold Value (Reconstruction Error MAE)', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    # Optional: Also increase tick label size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)  # Also increase legend font size
    plt.tight_layout()
    plt.show()
    
    print(f"Optimal threshold (best F1 score): {best_threshold:.4f}")
    print(f"Best F1 score: {best_f1:.4f}")
    
    return best_threshold, best_f1
