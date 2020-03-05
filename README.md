# Image Captioning

Este repositorio em como objetivo criar uma algortitmo que possa gerar legandas
automaticamente para imagens.

Este algotritmo é baseado no paper [*Show and Tell*](https://arxiv.org/abs/1411.4555).

### O problema
Algoritmos para geração de legendas para imagens compõem um problema completo de AI, ou seja, eles precisam entender mais de um área de AI para conseguir resolver o problema.
Primeiro o algoritmo deve entender dados de imagens e em segundo lugar devem ser hábil com texto em linguagem natural.

![a imagem mostra o fluxo de processamento de um algoritmo de geração de legendas para imagens. O fluxo se inicia com uma imagem de uma gato siamês dormindo, em segunda o fluxo avança para uma caixa que simboliza uma rede neural convolucional, e está escrito “Características visuais”. Na sequência o fluxo avança para uma segunda caixa chamada rede neural recorrente onde está escrito “características textuais”. Por fim o fluxo avança para uma última caixa escrito “Predição da resposta” que aponta para a resposta do algoritmo : Resposta Rq “Gato dormindo”.](docs/images/image-captioning.png)

*Figura 1: A imagem mostra o fluxo de processamento de um algoritmo de geração de legendas para imagens. O fluxo se inicia com uma imagem de uma gato siamês dormindo, em segunda o fluxo avança para uma caixa que simboliza uma rede neural convolucional, e está escrito “Características visuais”. Na sequência o fluxo avança para uma segunda caixa chamada rede neural recorrente onde está escrito “características textuais”. Por fim o fluxo avança para uma última caixa escrito “Predição da resposta” que aponta para a resposta do algoritmo : Resposta Rq “Gato dormindo”.*

### Imagens e as CNNs

Para imagens existem as redes neurais Convolucionais, de diversos tipos e tamanhos disponíveis em trabalhos acadêmicos ou mesmo no github.

![Redes Neurais Convolucionais](docs/images/cnn.png)

*Figura 2: A imagem mostra o fluxo de uma rede neural convolucional, que é especializada em processamento de imagem. No fluxo, temos como entrada uma imagem de um leão, a imagem é dividida em quadros e enviada para uma camda da rede neural chamada Convolution, a saída dessa camada é enviada para uma outra camada chamada Pooling, depois da Camada de Pooling, a imagem passa por outra camada COnvolution e por mais uma de Pooling, por fim o processamento chega na ultima camada (camada de saída) que prediz o resultado. Como exemplo de saída temos outra imagem de um Leão e uma imagem de um Tigre e a Rede seleciona a imagem do Leão como sendo a correta.*

### Texto e as RNNs

Para lidar com texto, usualmente são utilizadas as redes neurais recorrentes, que são ótimas para processamento de dados sequenciais (texto é uma cadeia de dados sequenciais).

![Redes Neurais Recorrentes](docs/images/rnn.png)

Para gerar legenda em imagens, teremos que combinar essas duas redes neurais.

Esse problema é relativamente antigo, desde de 2015 temos soluções bem robustas para ele, e por isso existe um grande número de trabalhos e exemplos para consulta.

O grande problema que enfrentaremos é que esses algoritmos exigem uma quantidade muito grande (milhões) de exemplos para que possam ser devidamente treinados. Existem algumas dezenas de modelos treinados disponíveis na web, mas quase todos geram legenda em inglês. Para português, temos que treinar nós mesmos os algoritmo, o que demandará tempo e esforço.


Uma solução de contorno será a utilização de um modelo treinado em inglês e submeter a sair de texto para um tradutor automático.

### Show And Tell

Algoritmo [Show and Tell(Bengio, 2015)](https://arxiv.org/abs/1411.4555) é bem famoso para esse tipo de tarefa, porém ele é antigo, sofrendo algumas atualizações como [Show, Tell and Atend (Bengio, 2015)](https://arxiv.org/abs/1502.03044). O algoritmo *Show and Tell* foi liberado pelo google em 2016, tendo várias implementações no github. O problema com esse algoritmo é que ele tem que ser treinado pelo prṕrio usuário.

### Datasets

Para este algoritimo estamos usando o conjunto de dados [COCO Dataset](http://cocodataset.org/#home). composto por mais 82000 imagens, com descrições (*em inglês*).

Ainda estamos buscando um c onjunto de dados com descrições em português, ou mesmo traduzir as descrições que atualmentes encontram-se em inglês.

### Treinamento do Modelo

O passo a passo utilizando para carregar, limpar e processar o *dataset*, como como treinar o modelo podem ser vistos [aqui](docs/train.md).

## Links e Fontes Bibliográficas
https://olhardigital.com.br/noticia/google-libera-algoritmo-que-descreve-imagens-com-94-de-precisao/62443

http://openaccess.thecvf.com/content_CVPR_2019/html/Zheng_Intention_Oriented_Image_Captions_With_Guiding_Objects_CVPR_2019_paper.html

https://www.captionbot.ai/

https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8

https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/

http://karpathy.github.io/2015/05/21/rnn-effectiveness/

http://colah.github.io/posts/2015-08-Understanding-LSTMs/

https://docs.google.com/document/d/1nT_Z79RISuVfeHeDGXrt-nWYlRjmgIYsa5-IDd5v_ks/edit?ts=5dded12b#


https://arxiv.org/abs/1703.09137

https://arxiv.org/abs/1708.02043

https://arxiv.org/abs/1505.06798

https://www.tensorflow.org/tutorials/text/image_captioning

## Configuração do ambiente

para facilitar o desenvolvimento, utilizaremos o anaconda. o Anaconda é um gerenciador de ambientes python focado em data science e machine leraning, com ele podemos facilmente instalar e configurar bibliotecas de desenvolvimento com algumas linhas de comando.

### Instalando o anaconda

#### Linux (ubuntu 16.04 / 18.04)

`curl -fsSl https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh | bash`

depois digite `yes` para aceitar o termos aperte enter para confirmar as opções padrões.

#### Mac

`curl -fsSl https://repo.anaconda.com/archive/Anaconda3-2019.10-MacOSX-x86_64.sh | bash`


### Criando o ambiente com o conda

Após a instalação, é necessário reinicar o shell para que o comando `conda` esteja disponivel.

#### Criando o ambiente
`conda create --name image-captioning tensorflow`

Caso você tenha placa de vídeo Nvidia com a biblioteca de aceleração `cuda` instalada (Apenas Linux)

`conda create --name image-captioning tensorflow-gpu`

Agora ative o ambinete

`conda activate image-captioning`

#### Instalando biblotecas necessárias

Dotenv

`conda install -c conda-forge python-dotenv`


Pillow

`conda install -c conda-forge pillow`


Tqdm

`conda install -c conda-forge tqdm`

## Para usar GPU

Para rodar os experimentos com aceleração de hardware (10x-15x mais rápido) é necessário ter uma (ou mais :flushed:) placas de vídeo Nvidia (outra marca não serve) das séries GT/X 8XX, 9XX, 10XX, RTX 20XX, ou da serie Pascal :moneybag: :moneybag: :moneybag:.

#### Instalando o CUDA

Se você possui placa de vídeo para rodar os experimentos, instale o [CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) antes de começar.

## Google Colab

Uma opção de ambiente de desevolvimento é o [google colab](https://colab.research.google.com/) que permite criar notebooks online vinculados a conta do google drive onde podemos rodar códigos python em gpus fornecidas pelo google gratuitamente, com GPU :smiley:.
