#DayDayUpQ3.py
dayup=1.0
dayfactor=0.01
for i in range(1,366):
    if i%7 in [6,0]:
        dayup=dayup*(1-dayfactor)
    else:
        dayup=dayup*(1+dayfactor)
print(dayup)
