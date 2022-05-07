# Utility stuff
lpl = 0
def fprint(fr, total_fr, description, print_every = 6):
		global lpl
		len_bar = 24
		ratio = round((fr + 1)/total_fr * len_bar)
		st = description+ ": [" + ratio * "=" + (len_bar - ratio) * " " + "]  " + str(fr + 1) + "/" + str(total_fr)
		if fr & (2 ** print_every - 1) == 0:
			print("\b" * lpl + st, end = "", flush = True)
		lpl = len(st)