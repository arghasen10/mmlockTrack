import csv
from datetime import datetime, timedelta

with open('salma_activity1.csv', 'w') as file:
    fieldnames = ['Datetime', 'Activity', 'User', 'Position', 'Orientation']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

start_time = datetime.strptime("2022-11-22_23:11:19", "%Y-%m-%d_%H:%M:%S")
end_time = datetime.strptime("2022-11-22_23:37:04", "%Y-%m-%d_%H:%M:%S")

curr_time = start_time
while curr_time < end_time:
    with open('salma_activity1.csv', 'a') as file:
        fieldnames = ['Datetime', 'Activity', 'User', 'Position', 'Orientation']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow({'Datetime': curr_time, 'Activity': 'Clapping', 'User': 'User-3', 'Position': 1, 'Orientation': 'Front'})

    curr_time += timedelta(seconds=1)

