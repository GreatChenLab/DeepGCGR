
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, accuracy_score, auc,  \
    precision_recall_curve, roc_auc_score



def train_and_validate(model, train_loader, val_loader, optimizer, criterion, epochs, model_path, early_stop_patience=20):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_precisions = []
    val_precisions = []
    train_recalls = []
    val_recalls = []
    train_f1s = []
    val_f1s = []
    train_accuracies = []
    val_accuracies = []
    true_train_labels = []
    train_scores = []
    true_val_labels = []
    val_scores = []
    train_aucs = []
    val_aucs = []

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        epoch_train_probs = []
        epoch_train_targets = []

        for data in train_loader:
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output.squeeze(), data.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            output = output.squeeze()
            probs = torch.sigmoid(output)  # logits → prob
            epoch_train_probs.extend(probs.tolist())
            epoch_train_targets.extend(data.y.tolist())

        train_losses.append(total_train_loss / len(train_loader))
        train_scores.extend(epoch_train_probs)
        true_train_labels.extend(epoch_train_targets)

        train_predictions = [1 if val > 0.5 else 0 for val in train_scores]
        train_precision = precision_score(true_train_labels, train_predictions, zero_division=0)
        train_recalls.append(recall_score(true_train_labels, train_predictions, zero_division=0))
        train_f1s.append(f1_score(true_train_labels, train_predictions, zero_division=0))
        train_accuracy = accuracy_score(true_train_labels, train_predictions)
        train_accuracies.append(train_accuracy)
        train_auc = roc_auc_score(true_train_labels, train_scores)
        train_precisions.append(train_precision)
        train_aucs.append(train_auc)

        model.eval()
        total_val_loss = 0
        epoch_val_probs = []
        epoch_val_targets = []

        with torch.no_grad():
            for data in val_loader:
                output, _ = model(data)
                loss = criterion(output.squeeze(), data.y)
                total_val_loss += loss.item()

                output = output.squeeze()
                probs = torch.sigmoid(output)
                epoch_val_probs.extend(probs.tolist())
                epoch_val_targets.extend(data.y.tolist())

        val_losses.append(total_val_loss / len(val_loader))
        val_scores.extend(epoch_val_probs)
        true_val_labels.extend(epoch_val_targets)

        val_predictions = [1 if val > 0.5 else 0 for val in val_scores]
        val_precision = precision_score(true_val_labels, val_predictions, zero_division=0)
        val_recall = recall_score(true_val_labels, val_predictions, zero_division=0)
        val_f1 = f1_score(true_val_labels, val_predictions, zero_division=0)
        val_accuracy = accuracy_score(true_val_labels, val_predictions)
        val_auc = roc_auc_score(true_val_labels, val_scores)

        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        val_accuracies.append(val_accuracy)
        val_aucs.append(val_auc)

        print(f'Epoch {epoch + 1}/{epochs} - Training loss: {train_losses[-1]:.4f} - Validation loss: {val_losses[-1]:.4f}')
        print(f'Training Precision: {train_precision:.4f} - Validation Precision: {val_precision:.4f}')
        print(f'Training Recall: {train_recalls[-1]:.4f} - Validation Recall: {val_recall:.4f}')
        print(f'Training F1 Score: {train_f1s[-1]:.4f} - Validation F1 Score: {val_f1:.4f}')
        print(f'Training Accuracy: {train_accuracy:.4f} - Validation Accuracy: {val_accuracy:.4f}')
        print(f'Training AUC: {train_auc:.4f} - Validation AUC: {val_auc:.4f}')

        scheduler.step(val_losses[-1])

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            torch.save(model.state_dict(), model_path)
            print("Saved best pre-trained model with validation loss:", best_val_loss)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"Validation loss did not improve. Early stop counter: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break



def compute_roc(true_labels, scores):
    fpr, tpr, _ = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(true_labels, scores)
    aupr = auc(recall, precision)
    return fpr, tpr, roc_auc, recall, precision, aupr

def pretrain(model, train_loader, val_loader, optimizer, criterion, epochs, save_path):
    model._init_weights(freeze_layers=False)
    train_and_validate(model, train_loader, val_loader, optimizer, criterion, epochs, save_path)
