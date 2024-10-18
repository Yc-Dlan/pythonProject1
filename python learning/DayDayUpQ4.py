#DayDayUpQ4.py
def dayup(df):
    dayup=1
    for i in range(365):
        if i%7 in [0,6]:
            dayup*=0.99
        else:
            dayup*=1.01
    return dayup
dayfactor=0.01
while dayup(dayfactor)<=37.78:
    dayfactor+=0.01
print(dayfactor)