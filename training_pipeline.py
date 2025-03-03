import torch
import torch.nn as nn
import torch.optim as optim
import time

import os
from serialization import save
from training_functions import train, evaluate

def add_prefix_to_path(path, prefix):
    dirpath, filename = os.path.split(path)
    name, ext = os.path.splitext(filename)
    file = f"{name}_{prefix}{ext}"
    new_path = os.path.join(dirpath, file)
    return new_path

def repeat_training(n, init_model, lr, model_path, history_path, epochs, train_dataloader, val_dataloader, test_dataloader, device):
    for i in range(n):
        model = init_model()
        print(f"training iteration: {i+1} of {n}")
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=lr)

        # training only the last layer
        last_layer = None
        if hasattr(model, 'fc'):
            last_layer = model.fc
        elif hasattr(model, 'classifier'):
            last_layer = model.classifier

        optimizer = optim.Adam(last_layer.parameters(), lr=lr)

        model_path_idx = add_prefix_to_path(model_path, i+1)
        history_path_idx = add_prefix_to_path(history_path, i+1)

        start_time = time.time()
        print("starting training...")
        training_history = train(epochs, model, train_dataloader, val_dataloader, optimizer, criterion, device,
                                 model_path_idx)
        print("training finished\n")
        print(training_history)
        end_time = time.time()
        print(f"training time: {end_time - start_time}\n")

        print("evaluating model...")
        best_model = init_model()
        best_model.load_state_dict(torch.load(model_path_idx, weights_only=True))
        test_accuracy, test_avg_loss = evaluate(best_model, test_dataloader, criterion, device)
        print(f"test loss: {test_avg_loss}, test accuracy: {test_accuracy}")

        training_history["accuracy_test"] = test_accuracy
        training_history["loss_test"] = test_avg_loss

        save(training_history, history_path_idx)
        print("training history saved\n")
