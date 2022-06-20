from contextlib import nullcontext
import pathlib
import time
from tokenize import String
from typing import Union
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
import pandas as pd
import seaborn as sns 
from utils_dataset import HemorrhageDataset


def import_nn(num_classes, device):
    """
    Instanciando o modelo usando as classes fornecidas pelo pytorch
    O modelo já é iniciado com pesos pré-definidos por meio de transfer learning
    Vamos treinar a rede para melhorar os pesos para nosso problema
    """


    # instantiate transfer learning model
    model = models.resnext50_32x4d(pretrained=True)

    # set all paramters as trainable
    for param in model.parameters():
        # param.requires_grad = False
        param.requires_grad = True

    # get input of fc layer
    n_inputs = model.fc.in_features

    # redefine fc layer / top layer/ head for our classification problem
    model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                    nn.SELU(),
                                    nn.Dropout(p=0.4),
                                    nn.Linear(2048, num_classes))

    # set all paramters of the model as trainable
    for name, child in model.named_children():
        for name2, params in child.named_parameters():
            params.requires_grad = True

    # set model to run on GPU or CPU absed on availibility
    model.to(device);

    return model

def define_config(model, device):
    '''
    Configuração de treino
    Loss usada como CrossEntropyLoss
    SGD optimizer com 0.9 de momentum e learning rate 3e-4.
    According to many Deep learning experts and researchers such as Andrej karpathy 3e-4
    is a good learning rate choice.
    '''

    # loss function
    # if GPU is available set loss function to use GPU
    criterion = nn.BCEWithLogitsLoss(weight=torch.FloatTensor([2,1,1,1,1,1]).cuda()).to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1.4e-4)

    return criterion, optimizer

class nn_modes:
    train = 'train'
    valid = 'valid'
    test = 'test'

def run_nn(model: models.ResNet,
        gen: DataLoader[HemorrhageDataset],
        criterion: nn.modules.loss._Loss, optimizer: Union[torch.optim.Optimizer , None],
        device: torch.device, mode: str):

    if mode in ['train']:
        model.train()
    elif mode in ['valid', 'test']:
        model.eval()
    else:
        raise RuntimeError(f'Unexpected mode {mode}')

    # empty correct and test correct counter as 0 during every iteration
    total_correct = 0
    # contador de imagens totais
    total_images = 0

    loss:float = 0

    original_labels:list[torch.Tensor] = []
    predicted_labels:list[torch.Tensor] = []

    if mode in ['valid', 'test']:
        cm = torch.no_grad()
    else:
        cm = nullcontext()

    with cm:
        Data: torch.Tensor
        labels: torch.Tensor
        for Data, labels in gen:
            # set variables to device
            Data, labels = Data.to(device), labels.to(device)
            # forward pass image
            y_pred = model(Data)
            # calculate loss
            temp_loss = criterion(y_pred.float(), labels)
            # get argmax of predicted tensor, which is our label
            predicted = torch.sigmoid(y_pred).data
            predicted[predicted >= 0.5] = 1.0
            predicted[predicted < 0.5] = 0
            # if predicted label is correct as true label, calculate the sum for samples
            total_correct += (predicted == labels).sum().item()
            # Contando imagens
            total_images += labels.shape[0]

            if mode in ['train']:
                if optimizer is None:
                    raise RuntimeError('Optimizer is None')
                # set optimizer gradients to zero
                optimizer.zero_grad()
                # back propagate with loss
                temp_loss.backward()
                # perform optimizer step
                optimizer.step()

            if mode in ['test']:
                original_labels.append(labels.data)
                predicted_labels.append(predicted)
            
            loss += temp_loss.item() * Data.size(0)
    loss /= len(gen.dataset)
    accuracy = total_correct * 100 / total_images / 6 # 6 é o numero de classes

    return accuracy, loss, original_labels, predicted_labels


def train_valid(model: models.ResNet, epochs: int,
            train_gen: DataLoader[HemorrhageDataset], 
            valid_gen: DataLoader[HemorrhageDataset],
            criterion: nn.modules.loss._Loss, optimizer: torch.optim.Optimizer,
            device: torch.device,
            path_salvar_modelo: str):

    # Local onde o modelo treinado será salvo
    pathlib_salvar_modelo = pathlib.Path(path_salvar_modelo)
    modelo_salvo = pathlib_salvar_modelo / "modelo.pt"
    # Criando pasta caso não exista
    pathlib_salvar_modelo.mkdir(parents=True, exist_ok=True)

    # empty lists to store losses and accuracies
    train_losses:list[float] = []
    valid_losses:list[float] = []
    train_accs:list[float] = []
    valid_accs:list[float] = []

    # Salvar melhor modelo
    max_acc = 0
    saved_loss = 9999

    # logger.info("\n\nIniciando treinamento/validação\n")
    print("Iniciando treinamento/validação\n")

    # set training start time
    start_time = time.time()

    # start training
    for i in range(epochs):
        e_start = time.time()

        accuracy, loss, _, _ = run_nn(model, train_gen, criterion, optimizer, device, nn_modes.train)

        e_end = time.time()
        hours, minutes = divmod((e_end - e_start) / 60, 60)
        
        train_losses.append(loss)
        train_accs.append(accuracy)

        print(f'Epoch {i+1}/{epochs}')
        print(f'Train Accuracy: {accuracy:2.2f}% Train Loss: {loss:2.4f}')
        print(f'Duration: {hours:.0f}h:{minutes:.0f} minutes\n')

        e_start = time.time()

        accuracy, loss, _, _ = run_nn(model, valid_gen, criterion, optimizer, device, nn_modes.valid)

        e_end = time.time()
        hours, minutes = divmod((e_end - e_start) / 60, 60)

        print(f'Validation Accuracy {accuracy:2.2f}% Validation Loss: {loss:2.4f}')
        print(f'Duration: {hours:.0f}h:{minutes:.0f} minutes\n')

        # Salvando o modelo com a melhor acurácia
        if accuracy >= max_acc and loss < saved_loss:
            torch.save(model.state_dict(), modelo_salvo) # TODO

            print(f'\nSalvando modelo com acurácia {accuracy:2.2f}% e loss {loss:2.4f}\n em {modelo_salvo}\n')
            max_acc = accuracy
            saved_loss = loss

        # some metrics storage for visualization
        valid_losses.append(loss)
        valid_accs.append(accuracy)

    # set total training's end time
    hours, minutes = divmod((time.time() - start_time) / 60, 60)

    # print training summary
    print(f'\nTotal Duration: {hours:.0f}h:{minutes:.0f} minutes\n')

    # Plot de loss
    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    ax = plt.gca()
    ax.set(yticks=np.arange(0, 3, 0.3))
    plt.title('Loss Metrics')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    # plt.show()
    plt.savefig(f'{path_salvar_modelo}losses.png', bbox_inches='tight')


    # Plot de acurácia
    plt.figure()
    plt.plot(train_accs, label='Training accuracy')
    plt.plot(valid_accs, label='Validation accuracy')
    ax = plt.gca()
    ax.set(yticks=range(0, 100+1, 10))
    plt.title('Accuracy Metrics')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    # plt.show()
    plt.savefig(f'{path_salvar_modelo}accuracy.png', bbox_inches='tight')

def test(model: models.ResNet,
         test_gen: DataLoader[HemorrhageDataset],
         criterion: nn.modules.loss._Loss,
         device: torch.device,
         path_salvar_modelo: str,
         show_info: bool = True):

    # Local onde o modelo treinado foi salvo (assume-se que o nome é modelo.pt)
    pathlib_salvar_modelo = pathlib.Path(path_salvar_modelo)
    modelo_salvo = pathlib_salvar_modelo / "modelo.pt"
    # Criando pasta caso não exista
    pathlib_salvar_modelo.mkdir(parents=True, exist_ok=True)

    # Carregando o modelo salvo
    model.load_state_dict(torch.load(modelo_salvo))

    if show_info:
        print("\nIniciando teste\n")

    accuracy, loss, original_labels, predicted_labels = run_nn(model, test_gen, criterion, None, device, nn_modes.test)

    if show_info:
        print(f"Test Loss: {loss:.4f}")
        print(f'Test accuracy: {accuracy:.2f}%')

        # Convert list of tensors to tensors -> Para usar nas estatísticas
        labels = torch.cat(original_labels)
        pred = torch.cat(predicted_labels)

        # Define ground-truth labels as a list
        LABELS = ['any', 'epidural', 'subdural', 'subarachnoid', 'intraventricular', 'intraparenchymal',]

        arr = multilabel_confusion_matrix(labels.cpu(), pred.cpu()) # corrigir no colab, essa linha estava errada, ytrue vem antes de ypred
        print(arr)

        for n, lab in enumerate(LABELS):
            df_cm = pd.DataFrame(arr[n], ['False','True'], ['False','True'])
            # Plot the confusion matrix
            plt.figure(figsize = (9,6))
            sns.heatmap(df_cm, annot=True, fmt="d", cmap='viridis')
            plt.xlabel("Prediction")
            plt.ylabel("Target")
            plt.savefig(f'{path_salvar_modelo}confusion_matrix_{lab}.png', bbox_inches='tight')

        print(f"Clasification Report\n\n{classification_report(pred.cpu(), labels.cpu())}")
    return accuracy, loss
