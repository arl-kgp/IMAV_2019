import csv
import statistics as stats

def parser1():
		
	filename = 'warehouse.csv'

	fields = [] 
	rows = [] 
	  
	# reading csv file 
	with open(filename, 'r') as csvfile: 
		# creating a csv reader object 
		csvreader = csv.reader(csvfile) 
		  
		# extracting field names through first row 
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

	with open(out,'a+') as outfile:
		write = csv.writer(outfile)

		for qr in qr_actual:
			al = []
			al.append(qr)
			al.append(qr_actual[qr])
			write.writerow(al)

def parserdash():
		
	filename = "out2.csv"

	rows = [] 
	  
	# reading csv file 
	with open(filename, 'r') as csvfile: 
		# creating a csv reader object 
		csvreader = csv.reader(csvfile)

		qr_alph = {}

		for row in csvreader: 
			rows.append(row) 

		# extracting each data row one by one 
		for row in rows:
			try:
				if(row[0] != row[1]):
					qr_alph[row[0]].append(row[1])
			except:
				qr_alph.update({row[0]:[row[1]]})

		for qr in qr_alph:
			qr_alph[qr] = set(qr_alph[qr])

		for qr in qr_alph:
			qr_alph[qr] = list(qr_alph[qr])

	out = 'out1.csv'

	with open(out,'w') as outfile:
		write = csv.writer(outfile)
		
		for qr in qr_alph:
			for i in range(len(qr_alph[qr])):
				al = []
				al.append(qr)
				al.append(qr_alph[qr][i])
				write.writerow(al)
	


def parser2():

	with open('output.csv', 'r') as t1, open('out1.csv', 'r') as t2:
		fileone = t1.readlines()
		filetwo = t2.readlines()

	with open('outputf.csv', 'w') as outFile:
		for line in filetwo:
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
	
	with open('outputf.csv', 'r') as tz:
		filez = tz.readlines()
		
	with open('output.csv', 'a') as outFileza:
		for line in filez:
			outFileza.write(str(line))

with open('outputf.csv', 'r') as tz:
		filez = tz.readlines()
		
with open('output.csv', 'a') as outFileza:
		for line in filez:
			outFileza.write(str(line))
	
	