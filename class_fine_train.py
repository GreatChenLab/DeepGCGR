import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, accuracy_score, auc, \
    precision_recall_curve,  roc_auc_score
from sklearn.metrics import confusion_matrix


def train(model, train_loader, optimizer, criterion, epochs, model_path, early_stop_patience=20):
    best_train_loss = float('inf')
    train_losses = []
    train_precisions = []
    train_recalls = []
    train_f1s = []
    train_accuracies = []
    true_train_labels = []
    train_scores = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        epoch_probs = []
        epoch_targets = []

        for data in train_loader:
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output.squeeze(), data.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            output = output.squeeze()
            probs = torch.sigmoid(output) 
            epoch_probs.extend(probs.tolist())
            epoch_targets.extend(data.y.tolist())

        train_losses.append(total_train_loss / len(train_loader))
        train_scores.extend(epoch_probs)
        true_train_labels.extend(epoch_targets)

        train_predictions = [1 if val > 0.5 else 0 for val in train_scores]
        train_precision = precision_score(true_train_labels, train_predictions, zero_division=0)
        train_precisions.append(train_precision)
        train_recall = recall_score(true_train_labels, train_predictions, zero_division=0)
        train_recalls.append(train_recall)
        train_f1 = f1_score(true_train_labels, train_predictions, zero_division=0)
        train_f1s.append(train_f1)
        train_accuracy = accuracy_score(true_train_labels, train_predictions)
        train_accuracies.append(train_accuracy)

        train_auc = roc_auc_score(true_train_labels, train_scores)

        print(f'Epoch {epoch + 1}/{epochs} - Training loss: {train_losses[-1]:.4f}')
        print(f'Training Precision: {train_precision:.4f}')
        print(f'Training Recall: {train_recall:.4f}')
        print(f'Training F1 Score: {train_f1:.4f}')
        print(f'Training Accuracy: {train_accuracy:.4f}')
        print(f'Training AUC: {train_auc:.4f}')

        scheduler.step(train_losses[-1])

        if train_losses[-1] < best_train_loss:
            best_train_loss = train_losses[-1]
            torch.save(model.state_dict(), model_path)
            print("Saved best fine-tuning model with training loss:", best_train_loss)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"Training loss did not improve. Early stop counter: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break


def test(model, test_loader, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    predictions = []
    true_labels = []
    scores = []

    with torch.no_grad():
        for data in test_loader:
            output, _ = model(data)
            output = output.squeeze()
            probs = torch.sigmoid(output)
            pred = torch.round(probs)

            predictions.extend(pred.tolist())
            true_labels.extend(data.y.tolist())
            scores.extend(probs.tolist())

    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    accuracy = accuracy_score(true_labels, predictions)
    fpr, tpr, roc_auc, precisions, recalls, aupr = compute_roc(true_labels, scores)
    precisions, recalls, thresholds = precision_recall_curve(true_labels, scores)
    aupr = auc(recalls, precisions)


    print("True Labels:", true_labels)
    print("Predictions:", predictions)
    print("Prediction Probabilities:", scores)
    print("Confusion Matrix:\n", confusion_matrix(true_labels, predictions))

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'AUC: {roc_auc:.4f}')
    print(f'AUPR: {aupr:.4f}')

    return fpr, tpr, roc_auc, precisions, recalls, aupr




def compute_roc(true_labels, scores):
    fpr, tpr, _ = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(true_labels, scores)
    aupr = auc(recall, precision)
    return fpr, tpr, roc_auc, recall, precision, aupr

def compute_pos_weight(graph_data):

    labels = [data.y.item() for data in graph_data]
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos
    if num_pos == 0:
        return torch.tensor([1.0])
    return torch.tensor([num_neg / num_pos])
