import csv
import statistics as stats

def parser1():
        
    filename = 'warehouse.csv'

    qrfreq = {}
      
    # reading csv file 
    with open('warehouse.csv', 'r') as csvfile: 
        for ls in csvfile.readlines():
            ls = ls.strip('\n')
            l = ls.split(',')
            
            if l[0] not in qrfreq:
                qrfreq[l[0]] = {}
            if l[1] not in qrfreq[l[0]]:
                qrfreq[l[0]][l[1]]=0
            qrfreq[l[0]][l[1]]+=1

    falloct = {}

    fallocq = {}
    print(qrfreq)
    for qr in qrfreq:
        ts = qrfreq[qr] 
        #peace
        d2 = []
        for a in ts:
            b=ts[a]
            d2.append((b,a))
        d2.sort()
        if d2[-1][1] not in falloct:
            falloct[d2[-1][1]] = set()
        falloct[d2[-1][1]].add(qr)
        fallocq[qr] = d2[-1][1]
    
    with open('out2.csv', 'r') as csvfile: 
        for ls in csvfile.readlines():
            ls = ls.strip('\n')

            l = ls.split(',')
            if l[0] in fallocq:
                fallocq[l[1]] = fallocq[l[0]]
                falloct[fallocq[l[0]]].add(l[1])
            elif l[1] in fallocq:
                fallocq[l[0]] = fallocq[l[1]]
                falloct[fallocq[l[1]]].add(l[0])
            else:
                print("Never seen before QR")
    
    with open('outputf.csv', 'a') as tz:
        for a in falloct:
            b=falloct[a]
            for c in b:
                tz.write(str(a) + "," + str(c)+"\n")

parser1()