import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

cwd = os.getcwd()
df = pd.read_csv(f'{cwd}\\Anexocs.csv', sep=';', low_memory=False)

ln = len(df)

df.loc[13, 'Team'] = 'Terrorist'
df.loc[27, 'Team'] = 'CounterTerrorist'

for i in range(ln):
  r = df.loc[i, 'RoundWinner']
  if r == 'True':
    df.loc[i, 'RoundWinner'] = True
  elif r == 'False':
    df.loc[i, 'RoundWinner'] = False
  elif r == 'False4':
    df.loc[i, 'RoundWinner'] = False

df['Team'] = df.Team.replace(['Terrorist', 'CounterTerrorist'], [1,2])
df['RoundWinner'] = df.RoundWinner.replace([False, True], [0,1])
df['Survived'] = df.Survived.replace([False, True], [0,1])
model = df.drop(['Unnamed: 0', 'InternalTeamId', 'Map', 'MatchId', 'RoundId', 'TimeAlive', 'TravelledDistance', 'FirstKillTime', 'MatchWinner', 'MatchKills', 'MatchFlankKills', 'MatchAssists', 'MatchHeadshots', 'AbnormalMatch', 'Survived', 'RLethalGrenadesThrown', 'RNonLethalGrenadesThrown', 'PrimaryAssaultRifle', 'PrimarySniperRifle', 'PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol', 'RoundKills', 'RoundAssists', 'RoundHeadshots', 'RoundFlankKills'], axis=1)

x = model.drop(['RoundWinner'], axis=1)
y = model.drop(['Team', 'RoundStartingEquipmentValue', 'TeamStartingEquipmentValue'], axis=1)

sc = MinMaxScaler()
sc_x = sc.fit_transform(x)
sc_y = sc.fit_transform(y)

dfx = pd.DataFrame(sc_x)
dfy = pd.DataFrame(sc_y)

x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.33)
model = LogisticRegression()
model.fit(x_train, y_train.values.ravel())

filename = 'checkpoints/model.pkl'
pickle.dump(model, open(filename, 'wb'))