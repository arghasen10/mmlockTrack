import csv
from datetime import datetime, timedelta

with open('aritra_activity1.csv', 'w') as file:
    fieldnames = ['Datetime', 'Activity', 'User', 'Position', 'Orientation']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

start_time = datetime.strptime("2022-11-26_20:29:04", "%Y-%m-%d_%H:%M:%S")
end_time = datetime.strptime("2022-11-26_20:40:28", "%Y-%m-%d_%H:%M:%S")

curr_time = start_time
while curr_time < end_time:
    with open('aritra_activity1.csv', 'a') as file:
        fieldnames = ['Datetime', 'Activity', 'User', 'Position', 'Orientation']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow({'Datetime': curr_time, 'Activity': 'Clapping', 'User': 'User-4', 'Position': 1, 'Orientation': 'Front'})

    curr_time += timedelta(seconds=1)

