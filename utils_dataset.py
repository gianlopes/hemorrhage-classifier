import pickle
import pandas as pd
import torch
import torch.utils
import torch.utils.data
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
import pathlib
import pydicom as dcm
import dicom_misc as misc
import h_labels

class ResizeDataset(torch.utils.data.Dataset):
    '''
    # Classe para reduzir o tamanho das imagens de 512x512 para 256x256.
    O colab passa do limite de VRAM treinando com imagens 512x512, portanto precisamos do resize. A performance da rede praticamente não foi afetada por essa medida.

    O resize deve ser feito antes de aplicar os artefatos para manter a consistência nos testes.
    '''
    def __init__(self, dataset, img_size=256):
        self.dataset = dataset
        self.transform = transforms.Resize((img_size, img_size))

    def __len__(self):
    # return length of image samples
        return len(self.dataset)

    def __getitem__(self, idx):
        data, y = self.dataset[idx]
        data = self.transform(data)
        # não precisa do one-hot encode nos labels, a outra classe já faz isso
        return (data, y)

class AdicionarArtefatosDataset(torch.utils.data.Dataset):
    '''
    Classe para colocar artefatos em um dataset.
    Da para adicionar os artefatos em um nível específico ou um range de 0 a teto aleatóriamente

    artefato deve ser uma função que recebe uma imagem e um deg_level de 1 a 10.
    se nivel_aleatorio=True então nivel_degradação é ignorado e um valor de 0 até nivel_aleatorio_teto
    é escolhido aleatoriamente
    '''
    def __init__(self, dataset, artefato=None, nivel_degradacao=5, nivel_aleatorio_teto=10, nivel_aleatorio=False):
        self.dataset = dataset
        self.artefato = artefato
        self.deg_level = nivel_degradacao
        self.nivel_aleatorio_teto = nivel_aleatorio_teto
        self.nivel_aleatorio = nivel_aleatorio
        self.rng = np.random.default_rng(seed=42)

    def definir_artefato(self, funcao):
        self.artefato = funcao

    def remover_artefato(self):
        self.artefato = None

    def set_deg_level(self, deg_level):
        # de 1 (pouco) a 10 (muito)
        self.deg_level = deg_level

    def __len__(self):
    # return length of image samples
        return len(self.dataset)

    def __getitem__(self, idx):
        # perform transformations on one instance of X
        # Original image as a tensor

        data, y = self.dataset[idx]

        if self.artefato:
            data = np.array(data)
            data = np.dot(data, [0.299, 0.587, 0.114]).astype(np.uint8) # converte para um canal


            if self.nivel_aleatorio:
                self.random_number = self.rng.integers(0, self.nivel_aleatorio_teto + 1)
                if self.random_number != 0:  # Se for entre 1-10, aplica o artefato, caso contrário, não aplica
                    data = self.artefato(data, self.random_number)
            else:
                data = self.artefato(data, self.deg_level) # aplica o artefato


            data = np.repeat(data[..., np.newaxis], 3, -1) # converte para 3 canais
            data = Image.fromarray(data)

        # não precisa do one-hot encode nos labels, a outra classe já faz isso

        return (data, y)


class RotationDataset(torch.utils.data.Dataset):
    '''
    Adiciona rotações aleatórias no dataset de 0 até rotation graus
    '''
    def __init__(self, dataset, rotation):
        self.dataset = dataset
        # Transformation for converting original image array to an image, rotate it randomly between -`rotation` degrees and `rotation` degrees, and then convert it to a tensor
        self.transform = transforms.Compose([
            transforms.RandomRotation(rotation),
        ])
    def __len__(self):
        # return length of image samples
        return len(self.dataset)

    def __getitem__(self, idx):
        X, y = self.dataset[idx]
        # perform transformations on one instance of X
        # Original image as a tensor
        data = self.transform(X)

        return data, y

def to_0255(img):
    maxi = np.max(img)
    if maxi == 0:
        maxi = 1
    img = img - np.min(img)
    img = img / maxi
    img = img * 255
    return img.astype(np.uint8)

def filter_df_by_label(df, label: str):
    return df[df.labels.apply(lambda st: st.split()).apply(lambda arr: label in arr)]

# adapted from https://github.com/appian42/kaggle-rsna-intracranial-hemorrhage/
class HemorrhageBaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: pathlib.Path, fold_list: list, balance:bool = False):
        self.data_path = data_path
        input_df_path = self.data_path / 'folds_df.pkl'
        with open(input_df_path, 'rb') as f:
            df = pickle.load(f)

        df = df.loc[df['fold'].isin(fold_list)] # Filtrando folds

        if balance:
            df_positive = df[df.labels != '']
            df_negative = df[df.labels == '']
            df_sampled = df_negative.sample(len(df_positive))
            df = pd.concat([df_positive, df_sampled], sort=False)

        df = df.sample(2000)

        self.dataset = df

    def __len__(self) -> int:
        return len(self.dataset)

    def getId(self, idx: int):
        img_id: str = self.dataset.iloc[idx]["ID"]
        return img_id

    def __getitem__(self, idx):
        df_row = self.dataset.iloc[idx]

        dicom_path = self.data_path / 'stage_2_train' / f'{df_row["ID"]}.dcm'

        # Ajustando window
        dicom = dcm.dcmread(dicom_path)
        img = dicom.pixel_array
        img = misc.rescale_image(
            img, df_row.RescaleSlope, df_row.RescaleIntercept, df_row.BitsStored, df_row.PixelRepresentation)
        # img = misc.apply_window(img, 40, 80)
        image1 = misc.apply_window(img, 40, 80)
        image2 = misc.apply_window(img, 80, 200)
        image3 = misc.apply_window(img, 40, 380)
        img = np.array([image1, image2, image3,]).transpose(1,2,0)
        img = to_0255(img)
        # img = np.repeat(img[..., np.newaxis], 3, -1) # converte para 3 canais
        img = Image.fromarray(img)

        # aqui os labels serão entregues com string em uma lista
        labels = df_row.labels.split()

        return img, labels

class HemorrhageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, base_dataset: HemorrhageBaseDataset, num_classes=6, train=False):
        self.dataset = dataset
        self.num_classes = num_classes
        self.base_dataset = base_dataset
        # Transformation for converting to a tensor
        if train:
            self.transform = transforms.Compose([
                transforms.RandomRotation(330),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        X, y = self.dataset[idx]
        # perform transformations on one instance of X
        # Original image as a tensor
        data: torch.Tensor = self.transform(X)
        # one-hot encode the labels
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        for v in y:
            label[h_labels.label_to_num[v]] = 1.0

        img_id = self.base_dataset.getId(idx)

        return data, label, img_id


class BrainTumorDataset(torch.utils.data.Dataset):
    '''
    Classe final do dataset, onde as imagens são transformadas em tensores
    e os labels recebem o onehot encode
    '''
    def __init__(self, dataset, num_classes=3):
        self.dataset = dataset
        self.num_classes = num_classes
        # Transformation for converting to a tensor
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, y = self.dataset[idx]
        # perform transformations on one instance of X
        # Original image as a tensor
        data = self.transform(X)
        # one-hot encode the labels
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        label[int(y)] = 1.0

        return data, label

def aplicar_artefatos_dataset(dataset, funcao_geradora_artefato=None, nivel_degradacao=5, nivel_aleatorio_teto=10, nivel_aleatorio=False):
    '''
    funcao_geradora_artefato pode ser uma função ou uma lista de funções
    se nivel_aleatorio=True então nivel_degradação é ignorado e um valor de 0 até nivel_aleatorio_teto
    é escolhido aleatoriamente
    '''
    # Aplicando o artefato nas imagens:

    # Se só um artefato foi fornecido ou nenhum
    if type(funcao_geradora_artefato) != list:
        data_artefatos = AdicionarArtefatosDataset(dataset, funcao_geradora_artefato, nivel_degradacao, nivel_aleatorio_teto, nivel_aleatorio)

    # Se uma lista de artefatos foi fornecida temos que criar um dataset para cada artefato.
    # esses datasets não são todos carregados na memória de uma vez, apenas uma lista com o nome dos arquivos é carregada.
    else:
        temp_data_artefatos = []
        # Cria um dataset para cada artefato
        for funcao in funcao_geradora_artefato:
            temp_data_artefatos.append(AdicionarArtefatosDataset(dataset, funcao, nivel_degradacao, nivel_aleatorio_teto, nivel_aleatorio))
        # Concatena
        data_artefatos = torch.utils.data.ConcatDataset(temp_data_artefatos)

    return data_artefatos

def aplicar_rotacao_dataset(dataset):
    # angles = [45, 90, 120, 180, 270, 300, 330]
    angles = [180, 330]
    data_aug_list = []
    for angle in angles:
        data_aug_list.append(RotationDataset(dataset, angle)) # adicionando versões rotacionadas
    data_aug = torch.utils.data.ConcatDataset(data_aug_list)
    return data_aug

# Função para pegar todos os labels de um dataset
# suporta ImageFoldes e variantes derivadas dele (Subset, ConcatDataset e customs)
def get_targets(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        targets = get_targets(dataset.dataset)
        return targets[dataset.indices]
    if isinstance(dataset, ImageFolder): # Image folder já vem com os targets definidos
        return dataset.targets
    if isinstance(dataset, torch.utils.data.ConcatDataset): # Para o concatDataset temos que concatenar o targets
        return np.concatenate([get_targets(sub_dataset) for sub_dataset in dataset.datasets])
    if isinstance(dataset, HemorrhageBaseDataset):
        return dataset.dataset.labels.apply(lambda st: st.split()).apply(lambda st: [''] if (st == []) else st).apply(lambda arr: [x for x in arr if x != 'any']).tolist()
        # raise NotImplementedError('Hemorrhage dataset get_targets não implementado, faça o balanceamento usando o df da classe')
    else: # Para os outros datasets que criamos, temos que ir procurando os targets no dataset de origem
        return get_targets(dataset.dataset)

# Função para trocar o nivel da degradação (deg_level) de um dataset
# suporta ImageFolders e variantes derivadas dele (Subset, ConcatDataset e customs)
def mudar_deg_level(dataset, deg_level):
    if deg_level < 1 or deg_level > 10:
        raise UserWarning("Cuidado, nível de deg_level com comportamento indefinido (use valores entre 1 e 10)")

    if isinstance(dataset, torch.utils.data.Subset):
        mudar_deg_level(dataset.dataset, deg_level)
    elif isinstance(dataset, AdicionarArtefatosDataset): # Image folder já vem com os targets definidos
        dataset.deg_level = deg_level
    elif isinstance(dataset, torch.utils.data.ConcatDataset): # Para o concatDataset temos que concatenar o targets
        for sub_dataset in dataset.datasets:
            mudar_deg_level(sub_dataset, deg_level)
    else: # Para os outros datasets que criamos, temos que ir procurando os targets no dataset de origem
        mudar_deg_level(dataset.dataset, deg_level)

# Função para trocar o artefato de um dataset

# Atenção: Se o dataset foi criado originalmente com múltiplos artefatos, essa função
# mudará todos eles para o fornecido na função.

# suporta ImageFolders e variantes derivadas dele (Subset, ConcatDataset e customs)
def mudar_artefato(dataset, func_artefato):
    if isinstance(dataset, torch.utils.data.Subset):
        mudar_artefato(dataset.dataset, func_artefato)
    elif isinstance(dataset, AdicionarArtefatosDataset): # Image folder já vem com os targets definidos
        dataset.artefato = func_artefato
    elif isinstance(dataset, torch.utils.data.ConcatDataset): # Para o concatDataset temos que concatenar o targets
        if len(dataset.datasets) > 1:
            raise UserWarning("Mudando artefato de múltiplos datasets. Por favor crie um dataset usando apenas um artefato antes de usar essa função")
        for sub_dataset in dataset.datasets:
            mudar_artefato(sub_dataset, func_artefato)
    else: # Para os outros datasets que criamos, temos que ir procurando os targets no dataset de origem
        mudar_artefato(dataset.dataset, func_artefato)

# Função que corta todas as classes do dataset no mesmo tamanho. O corte não é aleatório,
# Basicamente vai pegar as primeiras n imagens de todas as classes,
# onde n é o número de samples da menor classe
def undersample_dataset(datasetToUndersample):
    tgs, cnts = np.unique(get_targets(datasetToUndersample), return_counts=True)
    min_cnt = np.min(cnts)
    underIndices = []
    counters = [0] * cnts.shape[0] # TODO check if works
    for i, label in enumerate(get_targets(datasetToUndersample)):
        if counters[label] < min_cnt:
            underIndices.append(i)
            counters[label] += 1
    return torch.utils.data.Subset(datasetToUndersample, indices=underIndices)

def shuffle_pacientes(train_set, valid_set, test_set):
    fulldata = torch.utils.data.ConcatDataset([train_set, valid_set, test_set])

    # Divisão entre os datasets de treino, teste e validação
    porcentagem_treino = 70

    # Calculando o número de exemplos para cada dataset
    total = len(fulldata)
    n_treino = round(len(fulldata)*(porcentagem_treino/100))
    n_teste = round(len(fulldata)* (100 - porcentagem_treino)/(2*100))
    n_valid = len(fulldata) - n_treino - n_teste

    # Cria uma lista de indices aleatória que será usada para dividir os datasets
    indices = list(range(total))
    split = n_treino
    split2 = n_treino+n_valid
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    train_idx, valid_idx, test_idx = indices[:split], indices[split:split2], indices[split2:]

    data_train = torch.utils.data.Subset(fulldata, indices=train_idx)
    data_valid = torch.utils.data.Subset(fulldata, indices=valid_idx)
    data_test = torch.utils.data.Subset(fulldata, indices=test_idx)

    return data_train, data_valid, data_test
