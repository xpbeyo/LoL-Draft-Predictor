from data_fetcher import *
import pandas as pd
from datetime import date, timedelta

def sqr_norm(lst):
    total = 0
    for item in lst:
        total += item ** 2
    return total

start_date = date(2021, 1, 1)
end_date = date(2021, 4, 19)
delta = timedelta(days=1)

# if winner = 1, team A won; else if winner = 0, team B won
comp_df = pd.DataFrame(columns=[
    "winner"
])

for i in range(1, 6):
    comp_df["team_A_champ_{}".format(i)] = []
for i in range(1, 6):
    comp_df["team_B_champ_{}".format(i)] = []

while start_date <= end_date:
    games = read_json("./data/games/game_{date}.json".format(date=start_date))
    for game in games:
        matches = game["Matches"]
        team_A_id = game["Game"]["TeamAId"]
        team_B_id = game["Game"]["TeamBId"]
        for match in matches:
            team_A_comp = []
            team_B_comp = []

            players = match["PlayerMatches"] 
            for player in players:
                if player["TeamId"] == team_A_id:
                    team_A_comp.append(player["Champion"]["ChampionId"])
                else:
                    assert(player["TeamId"] == team_B_id)
                    team_B_comp.append(player["Champion"]["ChampionId"])
            team_A_comp.sort()
            team_B_comp.sort()
            if not (len(team_A_comp) == len(team_B_comp) == 5):
                continue
            winner = match["WinningTeamId"] 
            winner_bit = 0 if winner==team_A_id else 1
            row_dict = dict()
            if sqr_norm(team_A_comp) <= sqr_norm(team_B_comp):
                row_dict["winner"] = winner_bit
                for i in range(1, len(team_A_comp) + 1):
                    row_dict["team_A_champ_{}".format(i)] = team_A_comp[i-1] 
                    row_dict["team_B_champ_{}".format(i)] = team_B_comp[i-1]
            else:
                row_dict["winner"] = -winner_bit + 1
                for i in range(1, len(team_A_comp) + 1):
                    row_dict["team_A_champ_{}".format(i)] = team_B_comp[i-1] 
                    row_dict["team_B_champ_{}".format(i)] = team_A_comp[i-1] 

            comp_df = comp_df.append(
                row_dict,
                ignore_index=True
            )
    start_date += delta

comp_df = comp_df.reset_index(drop=True)
comp_df = comp_df.astype(int)
comp_df.to_csv("./team_comp.csv", index=False)
print(comp_df)
