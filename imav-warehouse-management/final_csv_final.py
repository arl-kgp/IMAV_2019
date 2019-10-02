import csv
import statistics as stats

def getout(id, fname):
    qrf = {}
    print("KYA")

    with open('outputf.csv', 'r') as tz:
        for ls in tz.readlines():
            ls = ls.strip('\n')
            l = ls.split(',')
            qrf[l[1]] = l[0]
    print(qrf, "f")
    needed = []
    with open(fname, 'r') as tz:
        for ls in tz.readlines():
            ls = ls.strip('\n')
            l = ls.split(',')

            if str(l[0]) == str(id):
                needed.extend(l[1:])
                break
    
    for f in needed:
        if f in qrf:
            print(f + ":" + qrf[f])
        else:
            print(f + ": not found")


if __name__ == '__main__':
    print("Enter input file name")
    fname = input()
    print("Enter ID")
    id = input()

    getout(id, fname)
