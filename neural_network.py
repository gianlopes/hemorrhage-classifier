import pathlib
import time
import torch
from torch import nn
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
import pandas as pd
import seaborn as sns


def import_nn(num_classes, device):
    """
    Instanciando o modelo usando as classes fornecidas pelo pytorch
    O modelo já é iniciado com pesos pré-definidos por meio de transfer learning
    Vamos treinar a rede para melhorar os pesos para nosso problema
    """


    # instantiate transfer learning model
    model = models.resnet50(pretrained=True)

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
                                    nn.Linear(2048, 2048),
                                    nn.SELU(),
                                    nn.Dropout(p=0.4),
                                    nn.Linear(2048, num_classes),
                                    nn.LogSigmoid())

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
    criterion = nn.BCEWithLogitsLoss().to(device)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=3e-4)

    return criterion, optimizer

def train_valid(model, epochs,
                train_gen, valid_gen,
                criterion, optimizer,
                device,
                path_salvar_modelo):

    # Local onde o modelo treinado será salvo
    pathlib_salvar_modelo = pathlib.Path(path_salvar_modelo)
    modelo_salvo = pathlib_salvar_modelo / "modelo.pt"
    # Criando pasta caso não exista
    pathlib_salvar_modelo.mkdir(parents=True, exist_ok=True)

    # empty lists to store losses and accuracies
    train_losses = []
    valid_losses = []
    train_correct = []
    valid_correct = []

    # Salvar melhor modelo
    max_acc = 0
    saved_loss = 9999

    # logger.info("\n\nIniciando treinamento/validação\n")
    print("Iniciando treinamento/validação\n")

    # set training start time
    start_time = time.time()

    # start training
    for i in range(epochs):
        model.train()
        # empty training correct and test correct counter as 0 during every iteration
        trn_corr = 0
        val_corr = 0

        # contador de imageis totais
        total_images = 0

        # set epoch's starting time
        e_start = time.time()

        temp_loss_train = 0

        for X, y in train_gen:
            # set label as cuda if device is cuda
            X, y = X.to(device), y.to(device)

            # forward pass image sample
            y_pred = model(X)
            # calculate loss
            loss = criterion(y_pred.float(), y)

            # get argmax of predicted tensor, which is our label
            predicted = torch.sigmoid(y_pred).data
            predicted[predicted >= 0.5] = 1.0
            predicted[predicted < 0.5] = 0
            # predicted = torch.argmax(y_pred, dim=1).data
            # if predicted label is correct as true label, calculate the sum for samples
            batch_corr = (predicted == y).sum()
            # increment train correct with correcly predicted labels per batch
            trn_corr += batch_corr

            # Contando imagens
            total_images += y.shape[0]

            # set optimizer gradients to zero
            optimizer.zero_grad()
            # back propagate with loss
            loss.backward()
            # perform optimizer step
            optimizer.step()

            temp_loss_train += loss.item() * X.size(0)
        temp_loss_train /= len(train_gen.dataset)
        # set epoch's end time
        e_end = time.time()

        # print training metrics
        # logger.info(f'\n\nEpoch {(i+1)}\nAccuracy: {trn_corr.item()*100/(total_images):2.2f} %  Loss: {temp_loss_train.item():2.4f}  Duration: {((e_end-e_start)/60):.2f} minutes\n') # total_images = 4 images per batch * 8 augmentations per image * batch length
        print(f'Epoch {(i+1)}/{epochs}\nAccuracy: {trn_corr.item()*100/(total_images/6):2.2f} %  Loss: {temp_loss_train:2.4f}  Duration: {((e_end-e_start)/60):.2f} minutes') # total_images = 4 images per batch * 8 augmentations per image * batch length

        train_losses.append(temp_loss_train)
        train_correct.append(trn_corr.item())

        X, y = None, None

        # contador de imageis totais
        total_images = 0

        # validate using validation generator
        # do not perform any gradient updates while validation
        model.eval()
        loss = 0

        with torch.no_grad():
            for X, y in valid_gen:
                # set label as cuda if device is cuda
                X, y = X.to(device), y.to(device)

                # forward pass image
                y_val = model(X)

                # get argmax of predicted tensor, which is our label
                predicted = torch.sigmoid(y_val).data
                predicted[predicted >= 0.5] = 1.0
                predicted[predicted < 0.5] = 0

                # increment test correct with correcly predicted labels per batch
                val_corr += (predicted == y).sum()

                # Contando imagens
                total_images += y.shape[0]

                # get loss of validation set
                loss += criterion(y_val.float(), y).item() * X.size(0)

        # média da loss, diferente do que está no notebook no drive
        loss /= len(valid_gen.dataset)
        # print validation metrics
        print(f'Validation Accuracy {val_corr.item()*100/(total_images)/6:2.2f} % Validation Loss: {loss:2.4f}')
        # logger.info(f'\n\nValidation Accuracy {tst_corr.item()*100/(total_images):2.2f} % Validation Loss: {loss.item():2.4f}\n')

        # Salvando o modelo com a melhor acurácia
        acc = val_corr.item()*100/(total_images)/6
        if acc >= max_acc and loss < saved_loss:
            torch.save(model.state_dict(), modelo_salvo) # TODO
            # logger.info(f'\n\nSalvando modelo com acurácia {val_corr.item()*100/(total_images):2.2f} % e loss {loss:2.4f}\n em {modelo_salvo}\n\n')
            print(f'\nSalvando modelo com acurácia {acc:2.2f} % e loss {loss:2.4f}\n em {modelo_salvo}\n')
            max_acc = acc
            saved_loss = loss

        # some metrics storage for visualization
        valid_losses.append(loss)
        valid_correct.append(val_corr.item())

    # set total training's end time
    end_time = time.time() - start_time

    # print training summary
    print(f"\nTraining Duration {(end_time/60):.2f} minutes")
    # logger.info("\n\nTraining Duration {:.2f} minutes".format(end_time/60))

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
    plt.plot([t*100/int(len(train_gen.dataset)/6) for t in train_correct], label='Training accuracy')
    plt.plot([t*100/int(len(valid_gen.dataset)/6) for t in valid_correct], label='Validation accuracy')
    ax = plt.gca()
    ax.set(yticks=range(0, 100+1, 10))
    plt.title('Accuracy Metrics')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    # plt.show()
    plt.savefig(f'{path_salvar_modelo}accuracy.png', bbox_inches='tight')

def test(model,
         test_gen,
         criterion,
         device,
         path_salvar_modelo,
         show_info=True):

    # Local onde o modelo treinado foi salvo (assume-se que o nome é modelo.pt)
    pathlib_salvar_modelo = pathlib.Path(path_salvar_modelo)
    modelo_salvo = pathlib_salvar_modelo / "modelo.pt"
    # Criando pasta caso não exista
    pathlib_salvar_modelo.mkdir(parents=True, exist_ok=True)

    # Carregando o modelo salvo
    model.load_state_dict(torch.load(modelo_salvo))

    if show_info:
        # logger.info("\n\nIniciando teste\n\n")
        print("\nIniciando teste\n")

    # Modo de teste
    model.eval()

    loss = 0

    # contador de imagens totais
    total_images = 0

    # Impede o cálculo de gradientes, poupa memória e tempo
    with torch.no_grad():
        correct = 0
        labels = []
        pred = []

        # Teste
        for X, y in test_gen:
            # mandando imagens e labels para a gpu
            X, y = X.to(device), y.to(device)

            # Guardando os labels originais para visualizar a matriz de confusão
            labels.append(y.data)

            # Predict
            y_val = model(X)

            # get argmax of predicted values, which is our label
            predicted = torch.sigmoid(y_val).data
            predicted[predicted >= 0.5] = 1.0
            predicted[predicted < 0.5] = 0
            pred.append(predicted)

            loss += criterion(y_val.float(), y).item() * X.size(0)

            # número de acertos
            correct += (predicted == y).sum()

            # Contando imagens
            total_images += y.shape[0]
    loss /= len(test_gen.dataset)

    if show_info:

        # logger.info(f"Test Loss: {loss:.4f}")
        # logger.info(f'Test accuracy: {correct.item()*100/(total_images):.2f}%')
        print(f"Test Loss: {loss:.4f}")
        print(f'Test accuracy: {correct.item()*100/(total_images)/6:.2f}%')

        # Convert list of tensors to tensors -> Para usar nas estatísticas
        labels = torch.stack(labels)
        pred = torch.stack(pred)

        # Define ground-truth labels as a list
        LABELS = ['any', 'epidural', 'subdural', 'subarachnoid', 'intraventricular', 'intraparenchymal',]

        arr = multilabel_confusion_matrix(labels.view(-1).cpu(), pred.view(-1).cpu()) # corrigir no colab, essa linha estava errada, ytrue vem antes de ypred
        for lab in LABELS:
            df_cm = pd.DataFrame(arr, ['False','True'], ['False','True'])
            # Plot the confusion matrix
            plt.figure(figsize = (9,6))
            sns.heatmap(df_cm, annot=True, fmt="d", cmap='viridis')
            plt.xlabel("Prediction")
            plt.ylabel("Target")
            plt.savefig(f'{path_salvar_modelo}confusion_matrix_{lab}.png', bbox_inches='tight')



        # plt.show()

        # Print the classification report
        # logger.info(f"Clasification Report\n\n{classification_report(pred.view(-1).cpu(), labels.view(-1).cpu())}") # TODO
        print(f"Clasification Report\n\n{classification_report(pred.view(-1).cpu(), labels.view(-1).cpu())}") # TODO
    return correct.item()*100/(total_images)/6, loss
