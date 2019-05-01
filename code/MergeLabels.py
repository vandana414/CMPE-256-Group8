class MergeLabel:
	def set_label(cat):
		cause = 0
		print(cat)
		natural = ['Lightning']
		accidental = ['Structure','Fireworks','Powerline','Railroad','Smoking','Children','Campfire','Equipment Use','Debris Burning']
		malicious = ['Arson']
		other = ['Missing/Undefined','Miscellaneous']
		if cat in natural:
			cause = 1
		elif cat in accidental:
			cause = 2
		elif cat in malicious:
			cause = 3
		else:
			cause = 4
		return cause