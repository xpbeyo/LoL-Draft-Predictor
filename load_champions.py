import pandas as pd
import json
champions_file = open("./data/champions.json")
champions_df = pd.json_normalize(json.load(champions_file))
champions_file.close()
champions_df = champions_df[["ChampionId", "Name"]]
champions_df.to_csv("champions.csv", index=False)
print(champions_df)
