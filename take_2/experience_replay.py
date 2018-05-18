import numpy as np
H=105
W=80
D=4

# ideja:
# kada dohvatam indeks i
# dohvatam par (s, a, r, s', t)
# takav da su s i s' depth-orke koje, redom
# pocinju na indeksu i i i+1
# a je action_buffer[i]
# r je reward_buffer[i]
# t je terminal_buffer[i]

# problem: dodavanje i odrzavanje duzine
# i mora biti iz [0, size-1]
# stoga je s (0, 1, 2, 3) do (size-1, size, size+1, size+2)
# a s' (1, 2, 3, 4) do (size, size+1, size+2, size+3)
# za ostale buffere nije problem

# treba osigurati da su svakih 5 uzastopnih smisleni tako da random samplovanje bude ok
# odnosno da kad biramo s i s' u indeksima ne postoji start/end
# ovo treba ispitati, pospan sam

class ExperienceReplay:
	def __init__(self, size, height, width, depth):
		self.frame_buffer=np.ndarray((size+depth-1, height, width), dtype=np.uint8)
		self.action_buffer=np.ndarray(size, dtype=uint8)
		self.reward_buffer=np.ndarray(size, dtype=float)
		self.terminal_buffer=np.ndarray(size, dtype=bool)
		self.start=0
		self.end=0

	def add(self, frame, reward, terminal):
		pass
	def sample(self):
		pass