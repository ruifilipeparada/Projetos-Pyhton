## Objetivo
Prever a tabela final da Premier League 2025/2026, com recurso a estatísticas históricas e à construção de um modelo preditivo.

---

## Descrição
Para este projeto, utilizei dados das últimas 5 temporadas (20/21 a 24/25) retirados do site football-data.co.uk, totalizando cerca de 1.900 jogos. Esta janela histórica pareceu-me suficiente para capturar padrões recentes sem sobrevalorizar temporadas antigas.
As etapas principais do projeto incluem: 
- Data Analysis
- Feature Engineering
- Data Processing
- Modeling

Ao longo destes blocos, tive em conta a otimização dos dados para o modelo, sem perder contextualidade, neste caso, rigor futebolístico. 
Outro ponto importante que tive em atenção foi ensinar o modelo a lidar com equipas promovidas que não tinham dados históricos na janela de 5 temporadas de PL utilizada.

Para a previsão, utilizei o XGBRegressor, com rank como target. Testei o modelo na temporada 24/25, realizei hyperparameter tuning e apliquei-o à temporada 25/26. 

Tendo em conta possíveis melhorias futuras a ser implementadas, pensei em: 
- incluir dados de divisões anteriores que mostrem o histórico real das equipas promovidas (ao invés de fazer o cálculos de valores proxy)
- incluir features relevantes para prever descidas e subidas abruptas de rendimentos das equipas, como:
  - se mudaram ou não de treinador 
  - transferências (jogadores, posições, investimento, receita de vendas)
  - dados financeiros (folha salarial, receitas de bilheteria e merchandising, situação do clube face às regras de fair play da Liga e da UEFA) 

> **Nota:** ambos os ficheiros .py contêm um índice no início, que ajuda a perceber a estrutura do documento 

---

## Estrutura
Projeto PL/ 
  - projeto_pl_code.py: ficheiro py de análise de dados, feature engineering e criação do modelo preditivo
  - main.py: ficheiro py de previsão da Premier League 2025/2026
  - notes.txt: ficheiro de texto onde são explicadas todas as features usadas ao longo do projeto 
  - data/:
    - (8 cvs: todos os ficheiros csv necessários)
  - modelo/:
    - (3 pkl: pasta onde são guardados os ficheiros do modelo, scaler e encoder)

> **Nota:** é vital que a estrutura da pasta "Projeto PL" seja mantida, de modo a não comprometer os upload dentro dos dois ficheiros .py 

---

## Linguagem e Ferramentas
- Linguagem: Python
- Editor: VS Code
- Dados: CSV 
