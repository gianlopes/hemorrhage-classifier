import pathlib
from typing import Any, Callable, Union
from xmlrpc.client import boolean
import torch
import torch.utils
import torch.utils.data
from torchvision.datasets import ImageFolder
from utils_dataset import HemorrhageBaseDataset, HemorrhageDataset, ResizeDataset, aplicar_artefatos_dataset, BrainTumorDataset, aplicar_rotacao_dataset, get_targets
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# Função pode adicionar diferentes tipos de artefatos, dar resize nas imagens

def process_dataset(data_path: pathlib.Path,
                    fold_list: list[int],
                    img_size:int = 256,
                    funcao_geradora_artefato: Union[None, Callable[[Any], Any]] = None,
                    nivel_degradacao:int = 5,
                    nivel_aleatorio_teto:int = 10,
                    nivel_aleatorio:bool = False,
                    augmentation:bool = False,
                    balance:bool = False,
                    train: bool = False) -> HemorrhageDataset:

    # Carrega o dataset manualmente separado na estrutura ImageFolder do pytorch
    # data_original = ImageFolder(dataset_path)
    data_original = HemorrhageBaseDataset(data_path, fold_list, balance)

    # Resize nas imagens
    data_resize = ResizeDataset(data_original, img_size=img_size)

    # Aplicando o artefato nas imagens
    data_artefatos = aplicar_artefatos_dataset(data_resize, funcao_geradora_artefato,
                                               nivel_degradacao, nivel_aleatorio_teto,
                                               nivel_aleatorio)

    # Augmentation
    data_aug_list = []

    # adiciona o dataset sem rotação
    data_aug_list.append(data_artefatos)

    if augmentation:
        # adiciona o dataset com outros 7 tipos de rotação
        data_aug_list.append(aplicar_rotacao_dataset(data_artefatos))

        # pode adicionar outros tipos de augmentation aqui

    # Juntando todos
    data_aug = torch.utils.data.ConcatDataset(data_aug_list)

    # Classe wrapper final
    # data_final = BrainTumorDataset(data_aug, num_classes=num_classes)
    data_final = HemorrhageDataset(data_aug, train=train)

    return data_final


# Função wrapper para gerar os 3 datasets com as mesmas configurações
# Espera que os dados estejam separados como:
# dataset_path/train
# dataset_path/valid
# dataset_path/test
def process_dataset_train_valid_test(
        dataset_path: str,
        img_size=256,
        funcao_geradora_artefato=None,
        nivel_degradacao=5,
        nivel_aleatorio_teto=10,
        nivel_aleatorio=False):

    data_path = pathlib.Path(dataset_path)

    train_folds = list(range(0, 6))
    valid_folds = list(range(6, 8))
    test_folds = list(range(8, 10))

    train_set = process_dataset(data_path, train_folds, img_size, funcao_geradora_artefato,
                                nivel_degradacao, nivel_aleatorio_teto, nivel_aleatorio, augmentation=False, balance=True, train=True)
    valid_set = process_dataset(data_path, valid_folds, img_size, funcao_geradora_artefato,
                                nivel_degradacao, nivel_aleatorio_teto, nivel_aleatorio, augmentation=False, balance=True)
    test_set = process_dataset(data_path, test_folds, img_size, funcao_geradora_artefato,
                               nivel_degradacao, nivel_aleatorio_teto, nivel_aleatorio, augmentation=False, balance=True)

    return train_set, valid_set, test_set

# Gera o dataloader
# Com balancear_dataset=True, uma tentativa de balancear as classes é feita para
# tentar dar a mesma probabilidade de seleção para todas as classes.
# Esta opção pode tanto gerar undersampling quanto supersampling ao mesmo tempo


def generate_dataloader(dataset: HemorrhageDataset,
                        batch_size: int,
                        balancear_dataset: bool = False):
    if not balancear_dataset:
        # num_workers=0 para evitar problemas do docker # , drop_last=True
        gen: DataLoader[HemorrhageDataset] = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         pin_memory=True, num_workers=0, drop_last=True)
        return gen

    # Caso balancear_dataset=True
    labels_unique, counts = np.unique(np.concatenate(get_targets(dataset)), return_counts=True)
    class_weights = [sum(counts) / c for c in counts]  # Calcula o pesos
    # Precisamos criar um array com um peso para cada imagem do dataset final(pode ter mais imagens que o final)
    labels_unique_dict = {k: v for v, k in enumerate(labels_unique)}
    example_weights = []
    for e in get_targets(dataset):
        tot = 0
        for es in e:
            tot += class_weights[labels_unique_dict[es]]
        tot /= len(e)
        example_weights.append(tot)


    # example_weights = [(class_weights[es] for es in e) for e in get_targets(dataset)]
    sampler = WeightedRandomSampler(
        example_weights, len(dataset))  # E cria um sampler

    gen = DataLoader(dataset, batch_size=batch_size, pin_memory=True,
                     num_workers=0, sampler=sampler, drop_last=True)  # drop_last=True
    return gen

# É importante manter o batchsize pequeno, ou o colab vai crashar


def generate_dataloader_train_valid_test(train_set, valid_set, test_set,
                                         balancear_dataset=True):
    train_batch_size = 10
    valid_batch_size = 10
    test_batch_size = 20
    train_gen = generate_dataloader(
        train_set, train_batch_size, balancear_dataset)
    valid_gen = generate_dataloader(
        valid_set, valid_batch_size, balancear_dataset)
    test_gen = generate_dataloader(
        test_set, test_batch_size, balancear_dataset)

    return train_gen, valid_gen, test_gen
