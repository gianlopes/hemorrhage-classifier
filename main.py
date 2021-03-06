import pathlib
from artefatos import ringing, contrast, blurring, ruido_gaussiano, ghosting
from neural_network import import_nn, define_config, train_valid, test
from artefatos_testes import teste_artefatos
from process_dataset import generate_dataloader, process_dataset, process_dataset_train_valid_test, generate_dataloader_train_valid_test
import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Escolhendo o device para realizar treino/teste
    if torch.cuda.is_available():
        # device_name = "cuda" # colab
        device_name = "cuda:1" # servidor
    else:
        device_name = "cpu"
    device = torch.device(device_name)
    print(f'Treinando em: {device_name}\n')

    # Artefatos utilizados
    artefatos = [ringing.generate_ringing,
                contrast.generate_contrast,
                blurring.generate_blurring,
                ruido_gaussiano.generate_ruido_gaussiano,
                ghosting.generate_ghosting]
    artefatos_nomes = ["ringing", "contrast", "blurring", "ruido_gaussiano", "ghosting"]

    img_size = 512
    dataset_path = "/mnt/nas/GianlucasLopes/hemorragia/rsna-intracranial-hemorrhage-detection/"
    
    #Treino sem degradação
    path_salvar_modelo = "./resultados/treino_7_3/"
    train_test_full(device = device,
                    epochs = 8,
                    dataset_path = dataset_path,
                    path_salvar_modelo = path_salvar_modelo,
                    img_size = img_size,
                    path_salvar_resultado="./resultados/test_results/")


def train_test_full(device,
                    img_size = 256,
                    path_salvar_modelo = './',
                    dataset_path = "./rsna-intracranial-hemorrhage-detection/",
                    funcao_geradora_artefato = None,
                    nivel_degradacao = 5,
                    nivel_aleatorio = False,
                    nivel_aleatorio_teto = 10,
                    epochs = 15,
                    balancear_dataset = False,
                    shuffle_pacientes_flag = False,
                    path_salvar_resultado = "./resultados/test_results/",):
    """
    Código usado para usar um artefato e um nível de degradação específico na fase
    de treino (validação também).
    Todas as imagens usadas no treino terão o mesmo artefato no mesmo nível.

    Posteriormente, outros experimentos podem ser realizados misturando níveis
    diferentes de degradação ou até mesmo diferentes artefatos em conjunto com
    imagens não degradadas.

    Para mudar o artefato e o nível, utilize as variáveis nivel_degradacao e
    funcao_geradora_artefato. Não são necessárias mudanças na parte de teste.
    """

    print(f"Informações:\nimg_size={img_size}\npath_salvar={path_salvar_modelo}\ndataset={dataset_path}\nfunc={funcao_geradora_artefato}\nnivel={nivel_degradacao}, aleatorio={nivel_aleatorio}, teto={nivel_aleatorio_teto}\nepochs={epochs}\nbalancear_dataset={balancear_dataset}, shuffle_pacientes_flag={shuffle_pacientes_flag}\n")

    num_classes = 6 # talvez deixar como parâmetro? mas já tem um monte

    # Criando datasets
    # dataset_path = "/content/drive/MyDrive/2020-12-BRICS/Neural Black/patientImages/splits"
    # train_set, valid_set, test_set = process_dataset_train_valid_test(
    #                 dataset_path,
    #                 img_size,
    #                 funcao_geradora_artefato,
    #                 nivel_degradacao,
    #                 nivel_aleatorio_teto,
    #                 nivel_aleatorio)

    # # Criando dataloaders
    # train_gen, valid_gen, test_gen = generate_dataloader_train_valid_test(
    #     train_set, valid_set, test_set, balancear_dataset)

    # # Importando modelo. Para mudar o tipo da rede, modifique a função import_nn
    # model = import_nn(num_classes, device)

    # # loss function e optimizer
    # criterion, optimizer = define_config(model, device)

    # # Treino
    # train_valid(model, epochs,
    #             train_gen, valid_gen,
    #             criterion, optimizer,
    #             device,
    #             path_salvar_modelo)

    # # Testa no dataset de teste criado acima, com tudo misturado
    # test(model, test_gen,
    #      criterion, device,
    #      path_salvar_modelo)

    folds = list(range(0, 10))
    data_path = pathlib.Path(dataset_path)
    dataset = process_dataset(data_path, folds, img_size, None,
                               nivel_degradacao, nivel_aleatorio_teto, nivel_aleatorio, augmentation=False, balance=False)
    gen = generate_dataloader(dataset, 40, balancear_dataset)
    model = import_nn(num_classes, device)
    criterion, optimizer = define_config(model, device)
    test(model, gen,
         criterion, device,
         path_salvar_modelo,
         path_salvar_resultado,
         True)

    plt.close('all')

if __name__ == "__main__":
    # Para conseguir reproduzir resultados
    torch.manual_seed(42)
    np.random.seed(42)
    plt.ioff() # Desabilita o modo interativo do matplotlib
    main()
