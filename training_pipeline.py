import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
from serialization import save, load
from training_functions import train, evaluate
import pandas as pd
import math


def add_prefix_to_path(path, prefix):
    dirpath, filename = os.path.split(path)
    name, ext = os.path.splitext(filename)
    file = f"{name}_{prefix}{ext}"
    new_path = os.path.join(dirpath, file)
    return new_path

def repeat_training(n, init_model, lr, model_path, history_path, epochs, train_dataloader, val_dataloader, test_dataloader, device, dropout=False, betas=(0.9, 0.999), tolerance=math.inf):
    for i in range(n):
        if not dropout:
            model = init_model()
        else:
            model = init_model(dropout)

        model.to(device)

        print(f"training iteration: {i+1} of {n}")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)

        # training only the last layer
        # last_layer = None
        # if hasattr(model, 'fc'):
        #     last_layer = model.fc
        # elif hasattr(model, 'classifier'):
        #     last_layer = model.classifier
        #
        # optimizer = optim.Adam(last_layer.parameters(), lr=lr)

        model_path_idx = add_prefix_to_path(model_path, i+1)
        history_path_idx = add_prefix_to_path(history_path, i+1)

        start_time = time.time()
        print("starting training...")
        training_history = train(epochs, model, train_dataloader, val_dataloader, optimizer, criterion, device,
                                 model_path_idx, tolerance)
        print("training finished\n")
        print(training_history)
        end_time = time.time()
        print(f"training time: {end_time - start_time}\n")

        print("evaluating model...")
        if not dropout:
            best_model = init_model()
        else:
            best_model = init_model(dropout)

        best_model.to(device)

        best_model.load_state_dict(torch.load(model_path_idx, weights_only=True))
        test_accuracy, test_avg_loss = evaluate(best_model, test_dataloader, criterion, device)
        print(f"test loss: {test_avg_loss}, test accuracy: {test_accuracy}")

        training_history["accuracy_test"] = test_accuracy
        training_history["loss_test"] = test_avg_loss

        save(training_history, history_path_idx)
        print("training history saved\n")

def train_with_different_parameters(n, init_model, epochs, train_dataloader, val_dataloader, test_dataloader, device, batchsize, lrs=[0.001], dropouts=[0.5], betas=[(0.9,0.999)]):
    for lr in lrs:
        for drop in dropouts:
            for beta in betas:
                newpath_history = f'output/history/cnn_lr={lr}_drop={drop}_beta={beta}_batch={batchsize}/'
                newpath_model = f'output/models/cnn_lr={lr}_drop={drop}_beta={beta}_batch={batchsize}/'
                if not os.path.exists(newpath_history):
                    os.makedirs(newpath_history)
                if not os.path.exists(newpath_model):
                    os.makedirs(newpath_model)
                history_path = os.path.join(newpath_history, 'history')
                model_path = os.path.join(newpath_model, 'model')
                repeat_training(n,init_model, lr, model_path, history_path, epochs, train_dataloader, val_dataloader, test_dataloader, device, dropout=drop, betas=beta)

def plot_results(n, batchsize, lrs, dropouts, betas, x_values, x_label):
    data = []
    for lr in lrs:
        for drop in dropouts:
            for beta in betas:
                accuracy_results = []
                for i in range(1, n+1):
                    newpath_history = f'output/history/cnn_lr={lr}_drop={drop}_beta={beta}_batch={batchsize}/'
                    history_path = os.path.join(newpath_history, f'history_{i}')
                    history = load(history_path)
                    accuracy_test = history["accuracy_test"]
                    accuracy_results.append(accuracy_test)
                data.append(accuracy_results)

    plot_data = []
    for i in range(len(x_values)):
        param = x_values[i]
        results = data[i]
        for res in results:
            plot_data.append({x_label: param, "test accuracy": res})

    df = pd.DataFrame(plot_data)

    plt.figure(figsize=(8, 5))
    sns.boxplot(x=x_label, y="test accuracy", data=df)
    plt.title('Boxplot of Results for Each Parameter Value')
    plt.show()
