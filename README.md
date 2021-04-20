# LoL-Ban-Phase-Picker
## Data Schema Description
`team_comp.csv`
- `winner`: 1 indicates team A won the game, 0 indicates team B won the game
- `team_A_champ_{index}`: Champion id of the champion team A picked at position `index`
- `team_B_champ_{index}`: Champion id of the champion team B picked at position `index` \
Each row represents a single game.

`champions.csv`
- `ChampionId`: The id of the champion
- `Name`: The name of the champion \
Each row represents a champion.
