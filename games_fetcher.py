from data_fetcher import *
from datetime import date, timedelta

start_date = date(2021, 1, 6)
end_date = date(2021, 4, 19)
delta = timedelta(days=1)
while start_date <= end_date:
    fetch_save(
        boxscore_by_date_link.format(date=str(start_date)),
        "./data/games/game_{date}.json".format(date=str(start_date))
    )
    start_date += delta
