import os.path
import requests
import simplejson as json

api_key = "0247b4b8b41d4a6eae76c97ba92621cb"
champions_link = "https://fly.sportsdata.io/v3/lol/stats/json/Champions"
boxscore_by_gameid_link = "https://fly.sportsdata.io/v3/lol/stats/json/BoxScore/{gameid}"
boxscore_by_date_link = "https://fly.sportsdata.io/v3/lol/stats/json/BoxScores/{date}"

def make_request(link):
    response = requests.get(link, headers={"Ocp-Apim-Subscription-Key": api_key})
    return response

def save_to_file(response, filename):
    file = open(filename, "w")
    file.write(json.dumps(json.loads(response.content), indent=2))
    file.close()

def fetch_save(link, filename):
    if not os.path.exists(filename):
        response = make_request(link)
        save_to_file(response, filename)

def read_json(filename):
    file = open(filename)
    games = json.load(file)
    file.close()
    return games

if __name__ == "__main__":
    my_date = "2021-4-02"
    filename = "boxscore_{}.json".format(my_date)
    link = boxscore_by_date_link.format(date=my_date)
    fetch_save(link, filename)
