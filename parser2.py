with open('output.csv', 'r', encoding="utf8", errors='ignore') as t1, open('out2.csv', 'r', encoding="utf8", errors='ignore') as t2:
	fileone = t1.readlines()
	filetwo = t2.readlines()

with open('outputf.csv', 'w') as outFile:
	for line in filetwo:
		print(line)
		lins = line.split(",")
		print(lins)
		q1 = lins[0]
		q2 = lins[1]
		q2 = q2[:-1]

		for linea in fileone:
			print(linea)
			lis = linea.split(",")
			print(lis)
			q = lis[0]
			t = lis[1]
			t = t[:-1]
			if q == q1:
				outFile.write(q2+","+t+"\n")

with open('outputf.csv', 'rb') as tz:
	filez = tz.readlines()
	
with open('output.csv', 'a') as outFiledash:
	for line in filez:
		outFiledash.write(str(line))

			
