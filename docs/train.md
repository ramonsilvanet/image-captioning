# Fluxo de Trabalho

![](images/workflow.png)

## O Algoritmo

 * **Photo Feature Extractor** Como primeira parte do algoritmo, iremos extrair vetores de características das imagens do conjunto de dados usando uma rede neural convolucional (mais precisamente a rede neural [VGG16](https://arxiv.org/abs/1505.06798)).

* **Sequence Processor** Para processar as descrições, utilizaremos uma Rede Neural Recorrente de Memória de Curto e Longo Prazo ([LSTM-RNN](https://en.wikipedia.org/wiki/Long_short-term_memory)).

* **Decoder** tanto o **feature extractor** quanto **sequence processor** produzem um vetor de comprimento fixo. Eles são mesclados e processados ​​por uma camada *Densa* (camadas são etapas de processamento de uma rede neural) para fazer uma previsão final.

Esse fluxo foi baseado nos estudos abaixo:

 * [Where to put the Image in an Image Caption Generator
](https://arxiv.org/abs/1703.09137)


* [What is the Role of Recurrent Neural Networks (RNNs) in an Image Caption Generator?](https://arxiv.org/abs/1708.02043)



### Carregando o Dataset

Para carregar o nosso conjunto de dados de treinamento, primeiro faço o download das [imagens](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip) e das [descrições das imagens](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip).

### Processando as Imagens

As redes neurais não entendem "imagens" ou "texto", elas entendem arrays de características, e para dar entrada em uma rede neural, precisamos vetorizar as imagens.

para isso rode o script `01_image_preprocessing.py`, ele irá vetorizar as imagens e gravar em um arquivo chamado `outputs/vectors/features.pkl`

### Processando texto

Para pre-processar o texto temos que realizar uma "limpeza", que irá :
 * Converter palavras para minúsculo
 * Remover pontuação
 * Remover palavras com apenas uma letra
 * Remover algarismos (e.g. 1,2,3,4,5,6,7,8,9)

 Para isso rode o script `02_text_preprocessing.py`, ele irá gerar o arquivo `outputs/descriptions/descriptions.txt` como as descrições de cada imagem.

### Treinamento da Rede Neural

Agora estamos prontos para treinar nosso modelo de rede neural.

Neste passo iremos definir o modelo de rede neural e ajustá-lo ao nosso conjunto de treinamento.

para isso, executaremos os seguintes passos:

 * Carregar conjunto de dados
 * Definir o modelo
 * Ajustar o modelo aos dados de treinamento
 * Validar o ajuste do modelo, utilizando dados de teste e validação

Rode o script `03_train.py`

### Predizendo as legendas

Uma vez treinado, podemos utilizar o modelo para avaliar a sua capacidade de previsão em um conjunto de dados de teste de validação.

Avaliaremos o modelo gerando descrições para todas as fotos no conjunto de dados de teste e avaliando essas previsões com uma *função de custo* padrão.

> Na otimização matemática, estatística, teoria da decisão, aprendizado de máquina e neurociência computacional, uma **função de perda** ou **função de custo** é uma função que mapeia um evento ou valores de uma ou mais variáveis num número real intuitivamente representando algum "custo" associado ao evento. Um problema de otimização procura minimizar uma função de perda. Uma função objetivo é uma função de perda ou sua função negativa (às vezes chamada função de recompensa, função de lucro, função de utilidade, função de aptidão, etc.), neste caso, ela deve ser maximizada.

[Fonte: Wikipedia](https://pt.wikipedia.org/wiki/Fun%C3%A7%C3%A3o_de_perda)


Para isso rode o script

```
python 04_single_evaluate.py
```

 Para avaliar a qualidade das respostas, utilizaremos uma métrica chamada [*BLEU Score*](https://machinelearningmastery.com/calculate-bleu-score-for-text-python/).
