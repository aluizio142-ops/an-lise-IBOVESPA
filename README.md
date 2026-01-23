# ANÁLISE PREDITIVA DO FECHAMENTO DO IBOVESPA 
Objetivo do projeto era verificar, por machine learning, o fechamento do indice bovespa para o dia seguinte, onde o mesmo fecharia em alta ou baixa.
# Fonte de dados
Foi utilizado o dataset do https://br.investing.com/indices/bovespa-historical-data em busca dos ultimos 30 anos de dados.
# ferramentas utilizadas
Foi usado o python, bem como as bibliotecas pandas, sklearng, matplotlib, seaborn, foco também em times series, redes neurais.
# Abordagem
Inicia-se o projeto com a limpeza de dados, focando em data, valor de fechamento, abertura, maxima e minima, pois o projeto seria embasado a esses dados.
Houve limpeza de dados, reestruturação do mesmo e inicio de busca do melhor modelo de machine learning para chegar ao objetivo (acurácia de 75%).
No notebook, é possível verificar comparações de modelos e seus erros para uma busca mais aprofundada e acertiva para o objetivo.
Posterior identificação do mesmo, é utilizado dados comparativos para chegar ao objetivo do projeto (predição de fechamento do ibovespa: alta ou baixa).

# resultado
Diante a isso, há 2 resultados satisfatórios, sendo eles: previsão de preço (valor), onde o modelo de ML Linear (L/lasso) e sarimax apresentam menos erro de magnitude (MAE/MAPE), porém, o objetivo era "direção do mercado do ibovespa" e a rede neural MLP apresentou uma melhor acurácia para o objetivo, consequentemente, esse foi o modelo escolhido para o projeto.

# notebook
É anexado o notebook do colab para verificação. Selecionar "executar tudo" para análise.
