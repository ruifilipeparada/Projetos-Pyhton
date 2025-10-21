print("\n-----------------------------------MAIN.PY-----------------------------------")
## ------------------------------------ÍNDICE---------------------------------------- ##

## 1. IMPORTS ##
## 2. MODEL and SCALERS UPLOAD ## 
## 3. DATA UPLOAD ## 
## 4. TRATAMENTO DF_HISTORICO_AGREGADO ## 
    # 4.1.) Equipa sem dados: Sunderland 
    # 4.2.) Agregação das 5 seasons por equipa (para conseguir fazer merge)
    # 4.3.) Sincronização com as equipas da Season 6
## 5. DATA PROCESSING ##
## 6. PREDICT ##

## ---------------------------------------------------------------------------------- ##





## 1. IMPORTS ##

import pandas as pd 
import numpy as np 
import joblib
from xgboost import XGBRegressor

from projeto_pl_code import process_data

import warnings
warnings.filterwarnings("ignore")





## 2. MODEL and SCALERS UPLOAD ## 

model = joblib.load("modelo/xgb_model.pkl")
scaler = joblib.load("modelo/scaler.pkl")
encoder = joblib.load("modelo/encoder.pkl")





## 3. DATA UPLOAD ## 

df_season_6 = pd.read_csv("data/pl_predict_s6.csv", sep=';')
df_dp_team_season = pd.read_csv("data/df_dp_team_season.csv", sep=';')

df_historico_agregado = df_dp_team_season.copy()





## 4. TRATAMENTO DF_HISTORICO_AGREGADO ## 

"""
Para aplicar o modelo, precisei de fazer um merge entre df_season_6 e df_dp_team_season, que tinham um número de rows diferente.
Então planeei agrupar nas 5 seasons por equipa, na cópia df_historico_agregado.
Esta agregação tem em consideração os mesmos pesos por season aplicados no doc. projeto_pl_code.py
"""

print("\n--------------------4. TRATAMENTO DF_HISTORICO_AGREGADO--------------------")

# 4.1.) Equipa sem dados: Sunderland 

"""
O Sunderland é uma equipa que foi promovida para a Premier League na época 25/26 (season 6 - a prever)
Contudo, não participou na PL em nenhuma das 5 seasons usadas para treinar o modelo. Portanto, precisa de tratamento especial. 
Assim, decidi que usar valores proxy calculados a partir da média dos dados das 3 equipas promovidas na Season anterior, 
neste caso, o Leicester, o Ipswich e o Southampton. 
"""

df_historico_agregado.loc[len(df_historico_agregado)] = [ 'Seasons 1-5', 'Sunderland'] + [np.nan]*(len(df_historico_agregado.columns)-2)

teams_s5 = df_dp_team_season[df_dp_team_season['Season'] == 'Season 5']['Team'].tolist()
teams_s4 = df_dp_team_season[df_dp_team_season['Season'] == 'Season 4']['Team'].tolist()
promoted_s5 = [team for team in teams_s5 if team not in teams_s4 and team != "Sunderland"]
print("\nEquipas promovidas na Season 5:", promoted_s5)

for col in df_historico_agregado.columns:
    if col not in ['Season', 'Team']:
        mean_value = df_dp_team_season.loc[
            (df_dp_team_season['Season'] == 'Season 5') & 
            (df_dp_team_season['Team'].isin(promoted_s5)), col
        ].mean()
        df_historico_agregado.loc[
            df_historico_agregado['Team'] == 'Sunderland', col
        ] = df_historico_agregado.loc[
            df_historico_agregado['Team'] == 'Sunderland', col
        ].fillna(mean_value)



# 4.2.) Agregação das 5 seasons por equipa (para conseguir fazer merge)

"""
Os pesos correspondem aos usados em projeto_pl_code.py, normalizados para que a soma total seja 1
O resultado é uma row por equipa. Equipas com menos seasons não são penalizadas, 
pois normalizamos pelo total de pesos efetivos usados.
"""

season_weights_main = {
    "Season 1": 0.0625,
    "Season 2": 0.109375,
    "Season 3": 0.234375,
    "Season 4": 0.28125,
    "Season 5": 0.3125
}

num_cols = df_historico_agregado.columns.drop(['Season', 'Team'])

df_to_aggregate = df_historico_agregado[df_historico_agregado['Team'] != 'Sunderland'].copy() # ignorar o Sunderland

aggregated_rows = []

for team, team_df in df_to_aggregate.groupby('Team'):
    total_weight = 0
    weighted_sum = pd.Series(0, index=num_cols)
    
    for season, weight in season_weights_main.items():
        season_mask = team_df['Season'] == season
        if season_mask.any():
            weighted_sum += team_df.loc[season_mask, num_cols].iloc[0] * weight
            total_weight += weight
            
    if total_weight > 0:
        weighted_sum /= total_weight
    
    row = pd.Series({'Season': 'Seasons 1-5', 'Team': team, **weighted_sum.to_dict()})
    aggregated_rows.append(row)

df_historico_agregado_final = pd.DataFrame(aggregated_rows)

# Acrescentar Sunderland
sunderland_row = df_historico_agregado[df_historico_agregado['Team'] == 'Sunderland']
df_historico_agregado_final = pd.concat([df_historico_agregado_final, sunderland_row], ignore_index=True)

# Reordenar colunas
df_historico_agregado_final = df_historico_agregado_final[['Season', 'Team'] + num_cols.tolist()]



# 4.3.) Sincronização com as equipas da Season 6

"""
Nos dados históricos das seasons 1-5 tinha equipas que não participam na Season 6, às quais fiz drop
Isto assegura que apenas as equipas participantes na Season 6 entram na previsão final
"""

teams_s6 = df_season_6['Team'].tolist()
df_historico_agregado_final = df_historico_agregado_final[df_historico_agregado_final['Team'].isin(teams_s6)].reset_index(drop=True)

print("\nEquipas após a sincronização com a Season 6:")
print(df_historico_agregado_final) 





## 5. DATA PROCESSING ##

"""
Season_num é adicionada como variável numérica auxiliar, usada apenas pelo modelo 
"""

season_map = {
    "Season 1": 1,
    "Season 2": 2,
    "Season 3": 3,
    "Season 4": 4,
    "Season 5": 5,
    "Seasons 1-5": 5  # para a linha agregada
}

df_historico_agregado_final['Season_num'] = df_historico_agregado_final['Season'].map(season_map)

num_vars_dp = num_cols.tolist() + ['Season_num']
cat_vars_dp = ['Season', 'Team']

df_pred_merge = df_historico_agregado_final.set_index('Team').loc[df_season_6['Team']].reset_index() # ordem das equipas dá match

X_pred_processed, feature_names_pred, _, _, _ = process_data(
    X=df_pred_merge, 
    y=None,  # y a prever 
    num_vars_dp=num_vars_dp, 
    cat_vars_dp=cat_vars_dp, 
    scaler=scaler, 
    encoder=encoder, 
    fit=False
)





## 6. PREDICT ##

print("\n---------------------------------6. PREDICT---------------------------------")

predictions = model.predict(X_pred_processed)

df_season_6['Predicted_Position'] = predictions
df_season_6['Predicted_Position'] = df_season_6['Predicted_Position'].rank(method='min').astype(int)
df_season_6 = df_season_6.sort_values('Predicted_Position').reset_index(drop=True)

print("\nTabela prevista para a Season 6 | Época 25/26 PL:")
print(df_season_6[['Season', 'Team', 'Predicted_Position']])




