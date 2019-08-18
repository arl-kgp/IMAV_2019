import csv
import statistics as stats

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



