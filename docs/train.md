# Fluxo de Trabalho

![](images/workflow.png)

## O Algoritimo

Como primeira parte do algoritimo, iremos extrair vetores de características das imagens do conjunto de dados usando uma rede neural convolucional (mais precisamente a rede neural [VGG16](https://arxiv.org/abs/1505.06798)).

Para processar as descrições, utilizaremos uma rede neural Recorrente padrão (*Vanilla*).

Por fim os vetores extraídos serão utilizados como entrada para uma rede neural Feed-Forward simples,que tem a função de o preditor.

Esse fluxo foi baseado nos estudos abaixo:

 * [Where to put the Image in an Image Caption Generator
](https://arxiv.org/abs/1703.09137)


* [What is the Role of Recurrent Neural Networks (RNNs) in an Image Caption Generator?](https://arxiv.org/abs/1708.02043)



### Carregando o Dataset

Para carregar o nosso conjunto de dados de treinamento, primeiro faço o download das [imagens](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip) e das [descrições das imagens](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip).

### Processando as Imagens

As redes neurais não entendem "imagens" ou "texto", elas entendem arrays de características, e para dar entrada em uma rede neural, precisamos vetorizar as imagens.

para isso rode o script `image_preprocessing.py`, ele irá vetorizar as imagens e gravar em um arquivo chamado `outputs/vectors/features.pkl`

### Processando texto

Para preprocesar o texto temos que realizar uma "limpeza", que irá :
 * Converter palavras para minúsculo
 * Remover pontuação
 * Remover palavras com apenas uma letra
 * Remover algarismos (e.g. 1,2,3,4,5,6,7,8,9)

 Para isso rode o script `text_preprocessing.py`, ele irá gerar o arquivo `outputs/descriptions/descriptions.txt` como as descrições de cada imagem.

### Trainamento da Rede Neural

Agora estamos prontos para treinar nosso modelo de rede neural.

Neste passo iremos definir o modelo de rede neural e ajustá-lo ao nosso conjunto de treinamento.

para executaremos os passos:

 * Carregar conjunto de dados
 * Definir o modelo
 * Ajustar o modelo aos dados

Rode o script `train.py`

### Predizendo as legendas
