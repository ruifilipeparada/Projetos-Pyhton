print("\n-----------------------------PROJETO_PL_CODE.PY-----------------------------")
## ------------------------------------ÍNDICE---------------------------------------- ##

## 1. IMPORTS ##
## 2. DATA UPLOAD ## 
## 3. DATA ANALYSIS ## 
    # 3.1.) Análise inicial a df 
    # 3.2.) Análise gráfica 
    # 3.3.) Análise apenas de colunas não relacionadas a odds
## 4. FEATURE ENGINEERING ##
    # 4.1.) Escolha de features de df (para a cópia df_fe)
    # 4.2.) Criação de novas features 
        # 4.2.1.) Features derivadas do jogos (diferenças e odds-derived)
        # 4.2.2.) Transformação para team-perspective (uma linha por equipa por jogo)
        # 4.2.3.) Estatísticas cumulativas por equipa-season
    # 4.3.) Preparação para novas equipas (zero data)
## 5. DATA PROCESSING ##
## 6. MODELING ##
## 7. FEATURE IMPORTANCE | SHAP ##
## 8. MODEL and SCALERS SAVING ##

## ---------------------------------------------------------------------------------- ##





## 1. IMPORTS ##

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
import joblib
import shap





## 2. DATA UPLOAD ## 

"""
Transferi os dados das épocas 20/21 a 24/25 da Premier League no site Football-Data.co.uk
Concatenei os datasets em df e através da coluna Date criei as etiquetas para uma nova coluna Season
"""

df_season_1 = pd.read_csv("data/pl_2020-2021.csv")
df_season_2 = pd.read_csv("data/pl_2021-2022.csv")
df_season_3 = pd.read_csv("data/pl_2022-2023.csv")
df_season_4 = pd.read_csv("data/pl_2023-2024.csv")
df_season_5 = pd.read_csv("data/pl_2024-2025.csv")

df = pd.concat([df_season_1, df_season_2, df_season_3, df_season_4, df_season_5], ignore_index=True)
df.insert(0, "Season", "")

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

def get_season(date):
    if date >= pd.Timestamp("2020-08-01") and date < pd.Timestamp("2021-08-01"):
        return "Season 1"
    elif date >= pd.Timestamp("2021-08-01") and date < pd.Timestamp("2022-08-01"):
        return "Season 2"
    elif date >= pd.Timestamp("2022-08-01") and date < pd.Timestamp("2023-08-01"):
        return "Season 3"
    elif date >= pd.Timestamp("2023-08-01") and date < pd.Timestamp("2024-08-01"):
        return "Season 4"
    elif date >= pd.Timestamp("2024-08-01") and date < pd.Timestamp("2025-08-01"):
        return "Season 5"
    else:
        print(f"Erro a preencher Season na data: {date}")
        return None 
df["Season"] = df["Date"].apply(get_season)





## 3. DATA ANALYSIS ## 

"""
Para analisar os dados decidi fazer: 
    - uma análise inicial, shape, tipo de dados, número de nulos, etc.
    - análise gráfica (uni, bi e multivariada)
    - análise apenas às colunas não relacionadas a odds (para ter melhor noção de nulos)
"""

print("\n------------------------------3. DATA ANALYSIS------------------------------")

# 3.1.) Análise inicial a df 

print("\n-------------------------3.1.) Análise inicial a df-------------------------")

print("\nForma do dataset")
print(df.shape)

print("\nTipos de dados do dataset")
print(df.dtypes)

print("\nInformações base do dataset")
print(df.info())

print("\nDados descritivos do dataset")
print(df.describe())

print("\nNulos por coluna")
print(df.isnull().sum())

print("\nTotal de nulos")
print(df.isnull().sum().sum())



# 3.2.) Análise gráfica 

cols = {
    "categorical": ['FTR', 'HTR'],  
    "numerical": [                 
        'FTHG', 'FTAG', 'HTHG', 'HTAG',
        'HS', 'AS', 'HST', 'AST',
        'HC', 'AC', 'HF', 'AF',
        'HY', 'AY', 'HR', 'AR'
    ],
    "pairplot": ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST']  
}



# Análise univariada (Histograma e Boxplot -> numéricas | Countplot -> categóriocas)
for col in cols["numerical"] + cols["categorical"]:
    plt.figure(figsize=(10,4))
    
    if col in cols["numerical"]:
        plt.subplot(1,2,1)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Histograma de {col}')
        
        plt.subplot(1,2,2)
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot de {col}')
    else:  
        sns.countplot(x=df[col])
        plt.title(f'Contagem de {col}')
    
    plt.tight_layout()
    plt.show()



# Análise bivariada (heatmap de correlação | scatterplots: FT Goals x Shots)
plt.figure(figsize=(12,10))
sns.heatmap(df[cols["numerical"]].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Heatmap de correlação entre variáveis numéricas")
plt.show()

plt.figure(figsize=(10,5))
sns.scatterplot(x='HS', y='FTHG', data=df, label='Home Team')
sns.scatterplot(x='AS', y='FTAG', data=df, label='Away Team', color='orange')
plt.xlabel('Shots')
plt.ylabel('Full Time Goals')
plt.title('Relação entre Remates e Golos')
plt.legend()
plt.show()



# Análise multivariada (pairplot)
sns.pairplot(df[cols["pairplot"]])
plt.show()



# 3.3.) Análise apenas de colunas não relacionadas a odds

print("\n-----------3.3.) Análise apenas de colunas não relacionadas a odds-----------")

df_no_odds = df[cols["categorical"] + cols["numerical"]].copy()

print("\nForma do df_no_odds")
print(df_no_odds.shape)

print("\nTipos de dados do df_no_odds")
print(df_no_odds.dtypes)

print("\nInformações base do df_no_odds")
print(df_no_odds.info())

print("\nDados descritivos do df_no_odds")
print(df_no_odds.describe())

print("\nNulos por coluna em df_no_odds")
print(df_no_odds.isnull().sum())

print("\nTotal de nulos em df_no_odds")
print(df_no_odds.isnull().sum().sum())

# Deteção de outliers df_no_odds
print("\nOutliers por coluna numérica (baseado em IQR):")
for col in cols["numerical"]:
    Q1 = df_no_odds[col].quantile(0.25)
    Q3 = df_no_odds[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_no_odds[(df_no_odds[col] < lower_bound) | (df_no_odds[col] > upper_bound)][col]
    print(f"{col}: {len(outliers)} outliers")





## 4. FEATURE ENGINEERING ##

print("\n---------------------------4. FEATURE ENGINEERING---------------------------")

# 4.1.) Escolha de features de df (para a cópia df_fe)

"""
Tratamento de dados originais de df: 

Para iniciar, criei a cópia df_fe para não interferir com dados brutos originais.
Adicionei as 18 colunas estatísticas de cada jogo, analisadas em df_no_odds.
Não incorporei colunas irrelevantes como 'Div', 'Time' e 'Referee'.
Optei por não fazer tratamento de outliers destas colunas, por serem valores plausíveis dentro da imprevisibilidade de um jogo de futebol.

Deparei-me com um elevado número de NaN nas colunas relacionadas a odds, ao passo que poderiam ser colunas importantes para o modelo perceber padrões.
Decidi, assim, incorporar em df_fe 3 casas de apostas: B365, PS e WH.
Para o tratamento de nulos de WH pensei que não seria razoável aplicar .mean() ou .median().
Fazê-lo comprometeria o contexto da odd, já que cada row representa um jogo diferente.
Tomei a decisão de preencher os valores NaN de WH com a média da odd de B365 e PS para cada jogo, respetivamente (para isso usei axis=1). 
""" 

odds_to_keep = ['B365H', 'B365D', 'B365A', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA']

df_fe = df[["Season"] + ["Date"] + ["HomeTeam"] + ["AwayTeam"] + cols["categorical"] + cols["numerical"] + odds_to_keep].copy() #cópia df feature engineering, para não apagar dados do dataset original

print("\nShape de df_fe:", df_fe.shape)
print("Total de nulos em df_fe:", df_fe.isnull().sum().sum())
print("Total de nulos nas odds:", df_fe[odds_to_keep].isnull().sum())

# Tratamento de nulos das odds WH
df_fe.loc[df_fe['WHH'].isnull(), 'WHH'] = df_fe[['B365H', 'PSH']].mean(axis=1)
df_fe.loc[df_fe['WHD'].isnull(), 'WHD'] = df_fe[['B365D', 'PSD']].mean(axis=1)
df_fe.loc[df_fe['WHA'].isnull(), 'WHA'] = df_fe[['B365A', 'PSA']].mean(axis=1)

print("\nTotal de nulos em df_fe após tratar odds:", df_fe.isnull().sum().sum())
print("Total de nulos nas odds após tratar odds:", (df_fe[odds_to_keep].isnull().sum()))



# 4.2.) Criação de novas features 

# 4.2.1.) Features derivadas do jogos (diferenças e odds-derived)

"""
Criei variáveis que descrevem a diferença entre a equipa da linha e o adversário (golos, remates, faltas, cartões, cantos).
Calculei também médias e dispersões das odds e probabilidades implícitas, para capturar o consenso do mercado.
Estas features devem ajudar o modelo a entender o desempenho relativo de cada equipa em cada jogo.
"""

# Diferença de golos
df_fe['FTGD'] = df_fe['FTHG'] - df_fe['FTAG'] 
df_fe['HTGD'] = df_fe['HTHG'] - df_fe['HTAG']   

# Diferença de remates / remates à baliza / faltas / cantos / cartões (home - away)
df_fe['ShotsDiff'] = df_fe['HS'] - df_fe['AS']
df_fe['ShotsOTDiff'] = df_fe['HST'] - df_fe['AST']
df_fe['FoulsDiff'] = df_fe['HF'] - df_fe['AF']
df_fe['YCdiff'] = df_fe['HY'] - df_fe['AY']
df_fe['RCdiff'] = df_fe['HR'] - df_fe['AR']
df_fe['CornersDiff'] = df_fe['HC'] - df_fe['AC']

# Odds (por média e por diferença) 
df_fe['Avg_H'] = df_fe[['B365H', 'PSH', 'WHH']].mean(axis=1)
df_fe['Avg_D'] = df_fe[['B365D', 'PSD', 'WHD']].mean(axis=1)
df_fe['Avg_A'] = df_fe[['B365A', 'PSA', 'WHA']].mean(axis=1)

df_fe['ODDS_H_diff'] = df_fe[['B365H', 'PSH', 'WHH']].max(axis=1) - df_fe[['B365H', 'PSH', 'WHH']].min(axis=1)
df_fe['ODDS_D_diff'] = df_fe[['B365D', 'PSD', 'WHD']].max(axis=1) - df_fe[['B365D', 'PSD', 'WHD']].min(axis=1)
df_fe['ODDS_A_diff'] = df_fe[['B365A', 'PSA', 'WHA']].max(axis=1) - df_fe[['B365A', 'PSA', 'WHA']].min(axis=1)

# Probabilidades implícitas 
df_fe['Implied_B365H'] = 1.0 / df_fe['B365H']
df_fe['Implied_B365D'] = 1.0 / df_fe['B365D']
df_fe['Implied_B365A'] = 1.0 / df_fe['B365A']

df_fe['Implied_PSH'] = 1.0 / df_fe['PSH']
df_fe['Implied_PSD'] = 1.0 / df_fe['PSD']
df_fe['Implied_PSA'] = 1.0 / df_fe['PSA']

df_fe['Implied_WHH'] = 1.0 / df_fe['WHH']
df_fe['Implied_WHD'] = 1.0 / df_fe['WHD']
df_fe['Implied_WHA'] = 1.0 / df_fe['WHA']

df_fe['AvgProb_H'] = df_fe[['Implied_B365H', 'Implied_PSH', 'Implied_WHH']].mean(axis=1)
df_fe['AvgProb_D'] = df_fe[['Implied_B365D', 'Implied_PSD', 'Implied_WHD']].mean(axis=1)
df_fe['AvgProb_A'] = df_fe[['Implied_B365A', 'Implied_PSA', 'Implied_WHA']].mean(axis=1)

# Resultado esperado com base no mercado (min de avg odds)
df_fe['ExpectedResult'] = df_fe[['Avg_H', 'Avg_D', 'Avg_A']].idxmin(axis=1).map({
    'Avg_H': 'H', 'Avg_D': 'D', 'Avg_A': 'A'
})

# Indicador favorito da Bet365 (útil para interpretabilidade)
df_fe['Fav_B365'] = df_fe[['B365H','B365D','B365A']].idxmin(axis=1).map({
    'B365H':'H', 'B365D':'D', 'B365A':'A'
})



# 4.2.2.) Transformação para team-perspective (uma linha por equipa por jogo)

"""
Criei uma versão em perspetiva de equipa (Home e Away) para que cada linha represente
a performance individual de uma equipa num jogo -> df_fe_team_rows

Inclui:
- Golos, remates, cantos, faltas e cartões ajustados à perspetiva de cada equipa
- Pontos ganhos e indicador de casa/fora
- Diferenças estatísticas (ex: remates feitos - remates sofridos)

Isto permite que o modelo aprenda padrões de desempenho por equipa ao longo da época.
"""

# Perspetiva da equipa da casa 
home_cols = [
    'Season', 'Date', 'HomeTeam', 'FTR', 'FTHG', 'FTAG', 'HS', 'HST', 'HC', 'HF', 'HY', 'HR',
    'Avg_H', 'Avg_D', 'Avg_A', 'AvgProb_H', 'AvgProb_D', 'AvgProb_A'
]
home_df = df_fe[home_cols].copy()
home_df.rename(columns={
    'HomeTeam': 'Team',
    'FTR': 'Result',
    'FTHG': 'GF', 'FTAG': 'GA',
    'HS': 'Shots', 'HST': 'ShotsOT', 'HC': 'Corners', 'HF': 'Fouls', 'HY': 'YC', 'HR': 'RC',
    'Avg_H': 'AvgOdds_H', 'Avg_D': 'AvgOdds_D', 'Avg_A': 'AvgOdds_A'
}, inplace=True)
home_df['IsHome'] = 1
home_df['Points'] = home_df['Result'].map({'H': 3, 'D': 1, 'A': 0})

# Merge -> estatísticas do adversário (AwayTeam)
away_stats = df_fe[['Season','Date','AwayTeam','FTAG','AS','AST','AC','AF','AY','AR']].copy()
away_stats.rename(columns={
    'AwayTeam': 'Opponent',
    'FTAG': 'GA_opp', 'AS': 'Shots_opp', 'AST': 'ShotsOT_opp', 'AC': 'Corners_opp', 
    'AF': 'Fouls_opp', 'AY': 'YC_opp', 'AR': 'RC_opp'
}, inplace=True)
home_df = home_df.merge(
    away_stats,
    left_on=['Season','Date','Team'],
    right_on=['Season','Date','Opponent'],
    how='left'
)

# Diferenças estatísticas 
home_df['GD_this_game'] = home_df['GF'] - home_df['GA_opp']
home_df['ShotsDiff_this_game'] = home_df['Shots'] - home_df['Shots_opp']
home_df['ShotsOTDiff_this_game'] = home_df['ShotsOT'] - home_df['ShotsOT_opp']
home_df['CornersDiff_this_game'] = home_df['Corners'] - home_df['Corners_opp']
home_df['FoulsDiff_this_game'] = home_df['Fouls'] - home_df['Fouls_opp']
home_df['YCdiff_this_game'] = home_df['YC'] - home_df['YC_opp']
home_df['RCdiff_this_game'] = home_df['RC'] - home_df['RC_opp']

# Perspetiva da equipa visitante 
away_cols = [
    'Season', 'Date', 'AwayTeam', 'FTR', 'FTAG', 'FTHG', 'AS', 'AST', 'AC', 'AF', 'AY', 'AR',
    'Avg_H', 'Avg_D', 'Avg_A', 'AvgProb_H', 'AvgProb_D', 'AvgProb_A'
]
away_df = df_fe[away_cols].copy()
away_df.rename(columns={
    'AwayTeam': 'Team',
    'FTR': 'Result',
    'FTAG': 'GF', 'FTHG': 'GA',
    'AS': 'Shots', 'AST': 'ShotsOT', 'AC': 'Corners', 'AF': 'Fouls', 'AY': 'YC', 'AR': 'RC',
    'Avg_H': 'AvgOdds_H', 'Avg_D': 'AvgOdds_D', 'Avg_A': 'AvgOdds_A'
}, inplace=True)
away_df['IsHome'] = 0
away_df['Points'] = away_df['Result'].map({'H': 0, 'D': 1, 'A': 3})

# Merge -> estatísticas do adversário (HomeTeam)
home_stats = df_fe[['Season','Date','HomeTeam','FTHG','HS','HST','HC','HF','HY','HR']].copy()
home_stats.rename(columns={
    'HomeTeam': 'Opponent',
    'FTHG': 'GA_opp', 'HS': 'Shots_opp', 'HST': 'ShotsOT_opp', 'HC': 'Corners_opp',
    'HF': 'Fouls_opp', 'HY': 'YC_opp', 'HR': 'RC_opp'
}, inplace=True)
away_df = away_df.merge(
    home_stats,
    left_on=['Season','Date','Team'],
    right_on=['Season','Date','Opponent'],
    how='left'
)

# Diferenças estatísticas 
away_df['GD_this_game'] = away_df['GF'] - away_df['GA_opp']
away_df['ShotsDiff_this_game'] = away_df['Shots'] - away_df['Shots_opp']
away_df['ShotsOTDiff_this_game'] = away_df['ShotsOT'] - away_df['ShotsOT_opp']
away_df['CornersDiff_this_game'] = away_df['Corners'] - away_df['Corners_opp']
away_df['FoulsDiff_this_game'] = away_df['Fouls'] - away_df['Fouls_opp']
away_df['YCdiff_this_game'] = away_df['YC'] - away_df['YC_opp']
away_df['RCdiff_this_game'] = away_df['RC'] - away_df['RC_opp']

# Concatenar home + away 
df_fe_team_rows = pd.concat([home_df, away_df], ignore_index=True)

print("\nShape após fase 4.2.2. (perspetiva por equipa):", df_fe_team_rows.shape)
df_fe_team_rows.head()



# 4.2.3.) Estatísticas cumulativas por equipa-season 

"""
Para cada equipa em cada season, calculei médias e contagens cumulativas das principais métricas
(GF, GA, remates, etc.), sempre considerando apenas jogos anteriores (uso .shift(1) para evitar data leakage).

O objetivo é dar ao modelo um contexto temporal: cada linha passa a refletir o histórico de desempenho
da equipa até àquele ponto da temporada, sem incluir informação futura. 
O resultado é guardado em df_fe_team_stats.
"""

df_fe_team_stats = df_fe_team_rows.copy() # Cópia (não mexer no df_fe_team_rows)

numeric_stats_team = [
    'GF', 'GA', 'Shots', 'ShotsOT', 'Corners', 'Fouls', 'YC', 'RC', 'Points'
]

# Ordenar por Season, Team e Date
df_fe_team_stats = df_fe_team_stats.sort_values(['Season', 'Team', 'Date']).reset_index(drop=True)

for col in numeric_stats_team:
    df_fe_team_stats[f'cum_mean_{col}'] = 0.0
    df_fe_team_stats[f'cum_count_{col}'] = 0

def _compute_group_cumulatives(df_team):
    df_team = df_team.sort_values('Date').copy()
    
    for col in numeric_stats_team:
        # Soma cumulativa até ao jogo anterior
        df_team[f'cum_sum_{col}'] = df_team[col].shift(1).cumsum().fillna(0)
        # Média cumulativa até ao jogo anterior
        df_team[f'cum_mean_{col}'] = df_team[col].shift(1).expanding().mean().fillna(0)
        # Número de jogos até ao anterior
        df_team[f'cum_count_{col}'] = df_team[col].shift(1).expanding().count().fillna(0).astype(int)
    
    return df_team

df_fe_team_stats = df_fe_team_stats.groupby(['Season', 'Team'], sort=False, group_keys=False).apply(_compute_group_cumulatives).reset_index(drop=True)

print("\nShape após fase 4.2.2. de df_fe_team_stats (cumulativas):", df_fe_team_stats.shape)
print("Exemplo (primeiras linhas com algumas cumulativas):")
print(df_fe_team_stats[['Season','Team','Date','GF','cum_mean_GF','cum_count_GF']].head())



# 4.3.) Preparação para novas equipas (zero data)

"""
Criei variáveis de apoio para equipas sem histórico em seasons anteriores (equipas promovidas).
Marquei essas equipas com IsNewTeam = 1 e atribuí valores proxy baseados na média das equipas
promovidas na season anterior. Assim, evitei missing data e mantive consistência nas features
sem introduzir fuga de informação.

Este passo garante que o modelo consiga lidar com equipas novas sem enviesar previsões.
"""

df_fe_team_stats['IsNewTeam'] = 0

teams_by_season = df_fe_team_stats.groupby('Season')['Team'].unique().to_dict()
all_seasons = sorted(df_fe_team_stats['Season'].unique())

proxy_stats_cols = ['Points', 'GF', 'GA', 'Shots', 'ShotsOT', 'Corners', 'Fouls', 'YC', 'RC']

for col in proxy_stats_cols:
    df_fe_team_stats[f'NewTeam_{col}'] = 0.0

for i, season in enumerate(all_seasons):
    current_teams = set(teams_by_season[season])
    previous_teams = set().union(*[teams_by_season[s] for s in all_seasons[:i]]) if i > 0 else set()
    
    # Identifica equipas sem histórico
    new_teams = current_teams - previous_teams # identificar equipas sem histórico

    # Marcar IsNewTeam
    mask_new = (df_fe_team_stats['Season'] == season) & (df_fe_team_stats['Team'].isin(new_teams))
    df_fe_team_stats.loc[mask_new, 'IsNewTeam'] = 1

    # Aplicar proxies apenas se houver season anterior
    if i > 0 and new_teams:
    
        prev_teams = set(teams_by_season[all_seasons[i-1]])
        prev_prev_teams = set().union(*[teams_by_season[s] for s in all_seasons[:i-1]])
        prev_promoted_teams = prev_teams - prev_prev_teams

        prev_mask = (df_fe_team_stats['Season'] == all_seasons[i-1]) & (df_fe_team_stats['Team'].isin(prev_promoted_teams))
        
        if not df_fe_team_stats.loc[prev_mask].empty:
            for col in proxy_stats_cols:
                proxy_mean = df_fe_team_stats.loc[prev_mask, col].mean()
                df_fe_team_stats.loc[mask_new, f'NewTeam_{col}'] = proxy_mean





## 5. DATA PROCESSING ##

print("\n-----------------------------5. DATA PROCESSING-----------------------------")

"""
Carreguei o csv com as labels (Position) e criei o df_dp_team_season (para a nomenclatura ajudar a situar no projeto)
Guardei um csv com o df_dp_team_season antes de:
    - fazer o merge com as labels (para evitar data leakage)
    - aplicar os pesos por season (vai facilitar a agregação por equipa no main.py)
Apliquei os pesos referidos para dar ao modelo a ideia de que as seasons mais recentes são mais importantes futebolisticamente do que as mais antigas
Criei o df_merged, que faz o merge entre os dados das seasons e as labels 
Desenvolvi 3 def: 
    - process_data() 
    - split_data() 
    - apply_model() 
"""

# Carregar csv de labels 
df_labels = pd.read_csv("data/pl_results_s1-s5.csv", sep=';')

df_dp_team_season = df_fe_team_stats.copy()

agg_cols_mean = [col for col in df_dp_team_season.columns if col.startswith('cum_mean_')]
agg_cols_count = [col for col in df_dp_team_season.columns if col.startswith('cum_count_')]
agg_dict = {col: 'mean' for col in agg_cols_mean}
agg_dict.update({col: 'sum' for col in agg_cols_count})

df_dp_team_season = df_dp_team_season.groupby(['Season', 'Team'], as_index=False).agg(agg_dict)

# Guardar df_dp_team_season antes do merge com labels e de aplicar os pesos 
df_dp_team_season.to_csv("data/df_dp_team_season.csv", sep=';', index=False)
print("\ndf_dp_team_season guardado na sub pasta 'data' do Projeto PL") 

season_weights = {
    "Season 1": 0.2,
    "Season 2": 0.35,
    "Season 3": 0.75,
    "Season 4": 0.9,
    "Season 5": 1.0
}
df_dp_team_season['Season_num'] = df_dp_team_season['Season'].map({
    "Season 1": 1,
    "Season 2": 2,
    "Season 3": 3,
    "Season 4": 4,
    "Season 5": 5
})
df_dp_team_season['Season_weight'] = df_dp_team_season['Season'].map(season_weights)

for col in agg_cols_mean + agg_cols_count:
    df_dp_team_season[col] = df_dp_team_season[col] * df_dp_team_season['Season_weight']

# Merge com labels
df_merged = df_dp_team_season.merge(
    df_labels[['Season', 'Team', 'Position']],
    on=['Season', 'Team'],
    how='left'
)



def process_data(X, y=None, num_vars_dp=None, cat_vars_dp=None, scaler=None, encoder=None, fit=True):
    
    if fit:
        scaler = StandardScaler()
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_num = scaler.fit_transform(X[num_vars_dp])
        X_cat = encoder.fit_transform(X[cat_vars_dp])
    else:
        X_num = scaler.transform(X[num_vars_dp])
        X_cat = encoder.transform(X[cat_vars_dp])

    X_processed = np.concatenate([X_num, X_cat], axis=1)

    feature_names = num_vars_dp + list(encoder.get_feature_names_out(cat_vars_dp))

    if y is not None:
        y_array = y.to_numpy() if hasattr(y, "to_numpy") else y
        y_array = y_array.astype(int)
    else:
        y_array = None

    return X_processed, feature_names, y_array, scaler, encoder



def split_data(dataset, label_col='Position', season_train=4):
    
    train_df = dataset[dataset['Season_num'] <= season_train]
    test_df = dataset[dataset['Season_num'] > season_train]
    
    X_train = train_df.drop(columns=[label_col])
    y_train = train_df[label_col].to_numpy() - 1
    
    X_test = test_df.drop(columns=[label_col])
    y_test = test_df[label_col].to_numpy() - 1
    
    return X_train, X_test, y_train, y_test



def apply_model(X_train, X_test, y_train, y_test, X_test_orig=None):

    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        gamma=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )

    xgb.fit(X_train, y_train)

    y_pred_train = xgb.predict(X_train)
    y_pred_test = xgb.predict(X_test)

    diff_train = np.abs(y_pred_train - y_train)
    diff_test = np.abs(y_pred_test - y_test)

    print("Training MAE:", np.mean(diff_train))
    print("Test MAE:", np.mean(diff_test))

    # Tabela prevista
    df_pred_sorted = None
    if X_test_orig is not None:
        df_pred_sorted = X_test_orig.copy()
        df_pred_sorted["Pred_Score"] = y_pred_test
        df_pred_sorted["Predicted_Position"] = df_pred_sorted["Pred_Score"].rank(ascending=True, method="first").astype(int)
        df_pred_sorted["Actual_Position"] = y_test + 1
        df_pred_sorted["Abs_Error"] = abs(df_pred_sorted["Predicted_Position"] - df_pred_sorted["Actual_Position"])
        df_pred_sorted = df_pred_sorted.sort_values("Predicted_Position").reset_index(drop=True)

        print("\nTabela prevista vs real:")
        print(df_pred_sorted[["Team", "Predicted_Position", "Actual_Position", "Abs_Error"]])

    return xgb, df_pred_sorted

print("\nDef: process_data(), split_data() e apply_model() criados") 





## 6. MODELING ##

print("\n--------------------------------6. MODELING--------------------------------")

cat_vars = ['Season', 'Team']
num_vars = agg_cols_mean + agg_cols_count + ['Season_num']

X_train, X_test, y_train, y_test = split_data(df_merged, label_col='Position', season_train=4)

X_train_processed, feature_names, y_train_array, scaler, encoder = process_data(
    X_train, y_train, num_vars, cat_vars, fit=True
)
X_test_processed, _, y_test_array, _, _ = process_data(
    X_test, y_test, num_vars, cat_vars, scaler=scaler, encoder=encoder, fit=False
)

print(f"\nShape X_train_processed: {X_train_processed.shape}")
print(f"Shape X_test_processed: {X_test_processed.shape}")

best_model, df_table_pred = apply_model(
    X_train_processed,
    X_test_processed,
    y_train_array,
    y_test_array,
    X_test_orig=X_test
)

print(f"\nMédia de erro absoluto (total): {df_table_pred['Abs_Error'].mean():.2f}")
print(f"Máximo erro absoluto: {df_table_pred['Abs_Error'].max()}")





## 7. FEATURE IMPORTANCE | SHAP ##

"""
Usei F.I. e SHAP apenas para tentar perceber quais as variáveis com maior impacto no modelo. 

- Gráfico de barras (Feature Importance): mostra a importância média de cada feature na previsão do modelo.
- Gráfico de pontos (SHAP summary plot): mostra como cada feature influencia individualmente cada previsão
(cor indica valor da feature, posição horizontal indica efeito na previsão).
"""

# Feature Importance 
plt.figure(figsize=(12,6))
plt.barh(feature_names, best_model.feature_importances_)
plt.xlabel("Feature Importance")
plt.title("Importância das Features - XGBRegressor")
plt.gca().invert_yaxis()
plt.show()

# SHAP
explainer = shap.Explainer(best_model, X_train_processed)
shap_values = explainer(X_test_processed)

shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names, plot_type="bar")
shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names)





## 8. MODEL and SCALERS SAVING ##
print("\n------------------------8. MODEL and SCALERS SAVING------------------------")
joblib.dump(best_model, "modelo/xgb_model.pkl")
joblib.dump(scaler, "modelo/scaler.pkl")
joblib.dump(encoder, "modelo/encoder.pkl")
print("\nModelo, scaler e encoder guardados na pasta 'modelo' do Projeto PL")




