import csv
import statistics as stats

def parser1():
		
	filename = 'warehouse.csv'

	rows = []
	fields = []

	#open the csv file
	with open(filename,'r') as csvfile:
		csvreader = csv.reader(csvfile)

		fields = next(csvreader)

		for row in csvreader:
			rows.append(row)

		# qr_fields = set(row[0] for row in rows)

		qr_alph = {}
		# alph = []
		# row_prev = ''
		# for row in rows:
		# 	if row[0] == row_prev:
		# 		alph.append(row[1])
		# 	else:
		# 		alph = []
		# 		alph.append(row[1])
		# 	row_prev = row[0]
		# 	qr_alph.update({row[0]:alph})

		for row in rows:
			try:
				qr_alph[row[0]].append(row[1])
			except:
				qr_alph.update({row[0]:[row[1]]})

		qr_actual = qr_alph.copy()
		# print(qr_alph)
		for qr in qr_alph:
			print(qr_alph[qr])
			print(qr)
			try:
				max_val = stats.mode(qr_alph[qr])
				qr_actual[qr] = max_val
				# print(qr_alph[qr])
			except:
				pass
			

	out = 'output.csv'

	with open(out,'w') as outfile:
		write = csv.writer(outfile)

		for qr in qr_actual:
			al = []
			al.append(qr)
			al.append(qr_actual[qr])
			write.writerow(al)


def parser2():

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

			
