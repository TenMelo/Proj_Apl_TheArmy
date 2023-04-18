# Proj_Apl_TheArmy
Projeto Aplicado teste

-------------
1. INTRODUÇÃO 

Prosseguindo com o conceito adotado na primeira fase, em que o escopo do projeto foi definido e a partir deste momento, passar-se-á à etapa de desenvolvimento
do projeto,conforme as orientações do Professor, serão apresentados os tópicos concernentes a escolha da base de dados e o seu tratamento para fins de subsidiar a análise 
exploratória e os testes futuros para cálculo da acurácia.

-----------------------------
2. DEFINIÇÃO  DAS BIBLIOTECAS 

Considerando que a escolha do tema convergiu para o processamento de imagens por meio de algoritmo, o qual visa identificar pessoas com deficiência física,
mais especificamente de locomoção, visando promover uma melhor acessibilidade de cadeirantes em um determinado estabelecimento, dentre as bibliotecas disponíveis
para implementação do projeto, foram selecionadas as bibliotecas: OpenCV, Matplotlib, Numpy, Os e TensorFlow. 

A seguir serão explicados os motivos pelos quais foram escolhidas essas bibliotecas.

->OpenCV é uma  biblioteca de processamento de imagens de código aberto que permite criar aplicativos de visão computacional com suporte para diversos 
sistemas operacionais e linguagens de programação.

->Matplotlib é uma biblioteca de visualização de dados em Python que permite criar gráficos e plots de alta qualidade. É comumente usada para visualizar
dados científicos e de engenharia, bem como dados financeiros, de negócios e outros tipos de dados.

->NumPy é uma biblioteca numérica em Python que fornece suporte para arrays e matrizes multidimensionais, bem como funções matemáticas e estatísticas para 
manipular esses arrays. É frequentemente usada em análise de dados, computação científica e aprendizado de máquina.

->TensorFlow é uma biblioteca de aprendizado de máquina de código aberto criada pelo Google. É usada para criar, treinar e implantar modelos de aprendizado 
de máquina, particularmente modelos de aprendizado profundo. TensorFlow oferece suporte para vários tipos de modelos, incluindo redes neurais convolucionais,
redes neurais recorrentes e modelos de sequência para processamento de linguagem natural.

->Os é uma biblioteca do Python que fornece uma interface para interagir com o sistema operacional subjacente. Ela oferece uma ampla gama de funcionalidades,
incluindo operações de arquivos e diretórios, gerenciamento de processos, manipulação de caminhos de arquivos, entre outras. A biblioteca os é frequentemente 
usada para criar programas que precisam interagir com o sistema operacional em que estão sendo executados, como programas de gerenciamento de arquivos e sistemas 
de automação de tarefas.

-----------------
3. DEFINIÇÃO DA BASE DE DADOS

Para a base dados foi escolhido um database contendo imagens de pessoas cadeirantes e não cadeirantes, para que seja feito um algoritmo capaz de analisar essas imagens e gerar um resultado indicando em qual categoria a imagem se enquadra.
A base de dados foi obtida através do Open Images Dataset V7 (https://storage.googleapis.com/openimages/web/index.html), um repositório com mais de 9 milhões de imagens catalogadas e rotuladas, tendo sido utilizado como referência para pesquis as classes “man” e “wheelchair”.
3.1 Análise Exploratória  dos dados

Para a análise exploratória dos dados, foi verificada a qualidade dos dados, tendo sido apurado e classificado sumariamente entre as classes “cadeirante” e “não-cadeirante”, de forma que um tratamento mais profundo foi feito posteriormente.

----------------------------
4. TRATAMENTO DA BASE DE DADOS

4.1 Preparação da base de dados

Durante a preparação de dados, foram tomadas medidas necessárias para fazer com que os dados se tornassem adequado ao modelo que estávamos construindo, que após a ação de análise exploratória dos dados, se mostrou necessário uma normalização das dimensões, de forma que foi utilizado as seguintes linhas de código para superar esta barreira:

train = ImageDataGenerator(rescale = 1/255)

validation = ImageDataGenerator(rescale = 1/255)


4.2 Treinamento da base de dados

Eleita a base de dados a ser utilizada no presente Projeto Aplicado e após passar pelo processo de preparação supramencionado, o primeiro passo para o treinamento dessa base é compreender os dados e quais são os aspectos de relevância para a visualização, como por exemplo as dimensões das imagens , escala de cores, níveis de sombras e eventuais outros detalhes que possam induzir a um outlier.
Sequencialmente, os parâmetros serão normalizados para um determinado default, a fim do algoritmo possa se adequar ao processo de classificação. 
A medida em que o modelo for finalizado e submetido ao treinamento, será possível realizar os devidos ajustes,a  fim de evitar o underfitting/overfitting.
Aplicando ao nosso modelo, foi utilizado o seguinte comando para o treinamento com o dataset:


train_dataset=train.flow_from_directory("C:/Users/Joao_Pedro/Desktop/Data_Set/Treino/",
                                         			target_size = (600,600),
                                         			batch_size = 10,
                                         			class_mode = 'binary')


Foi utilizado o comando “train.flow_from_directory” para treinar e parametrizar os dados, de forma que em target size foi escolhido o tamanho de 600x600 pixels, devido ao fato de grande parte das imagens do banco de dados ser de alta resolução, um batch size de 10, devido ao valor da database ser de aproximadamente 100 fotos, e o class_mode = ‘binary’ pelo fato de a classificação ser binária - ou é ou não é cadeirante.

------------------------
5. BASE TEÓRICA DOS MÉTODOS ANALÍTICOS

A natureza dos dados que compõem tanto a base de dados a ser comparada, quanto as informações que serão submetidas ao crivo do algoritmo, capturadas por um sensor óptico, tal como uma câmera de circuito fechado de vídeo, são classificadas como dados quantitativos, vez que em seu output constarão valores que podem ser mensurados.
O sistema de Aprendizado de Máquina a ser utilizado é o de Aprendizado Supervisionado, pois se trata de uma abordagem definida pelo uso de conjuntos de dados rotulados, ou seja, basicamente haverão dois tipos de imagens a serem distinguidas uma da outra: a de uma pessoa cadeirante e outra que não é.
Para fins didáticos e para que se evite interpretações dúbias ou extensivas, será adotada a palavra “cadeirante” para denominar o tipo de imagem a ser incluída no banco de dados.
É de bom alvitre explicitar que muito embora o termo correto a ser empregado seria o de “pessoa com deficiência”, em razão de ser um termo genérico, poderia se correr o risco de haver inclusão ou mesmo inferência de outras tipos de deficiência que não aquela estudada neste caso, induzindo, dessa forma, a erro o escopo do trabalho.
Os dados rotulados como “true” ou “false” servirão para treinar os algoritmos, possibilitando construir um cabedal de conhecimento diuturnamente aperfeiçoado no sentido de mitigar erros e melhorar continuamente a acurácia do modelo.
Tendo em vista que o modelo em testilha se presta a categorizar os resultados em somente duas situações possíveis para poder posteriormente, adotar as medidas respectivas, tais hipóteses constituem valores discretos a constar no output, sendo denominados de classes em um modelo de classificação.
Para auxiliar na estimativa da probabilidade associada à ocorrência de determinado evento, em face de  um conjunto de variáveis binárias, existe uma técnica que considera a natureza dicotômica da variável dependente, chamada de Regressão Logística.
A regressão logística modela as probabilidades para problemas de classificação binários, abarcando dois possíveis resultados, no caso “true”/“false”, “yes”/”no” ou qualquer que sejam os rótulos a serem denominados, desde que sigam a lógica de resultados antagônicos entre si.  
Há que se observar que a finalidade principal do sistema proposto neste trabalho é o de reconhecer a condição individual das pessoas que adentrarão ao estabelecimento, propiciando no início de sua instalação, tanto a identificação acima quanto ao longo do tempo, uma estimativa de previsão desses resultados.
Aliado à função primária de reconhecimento e adoção das medidas cabíveis e respectivas a cada classe, também fornecerá paralelamente as estimativas para uma manutenção contínua e equilibrada dos serviços prestados naquele estabelecimento, desde a contratação de efetivo ou mudança de layout do local ou aquisição de novos recursos, mediante a demanda a ser estimada pelo modelo.   

---------------------------------
6. CÁLCULO DA ACURÁCIA

O modelo ainda não foi totalmente submetido a teste, não sendo possível por enquanto, estimar sua acurácia, contudo  será utilizado a estimativa de acurácia via score, de forma que ao final do treinamento e posterior correção, será feita uma avaliação para medir a porcentagem de acertos de classificação do modelo treinado, de forma a direcionar possíveis correções ou indicar a possibilidade de erros, como o enviesamento do sistema.
