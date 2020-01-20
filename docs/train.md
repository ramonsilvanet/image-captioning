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

### Processando as Imagens

### Processando texto

### Trainamento da Rede Neural

### Predizendo as legendas
