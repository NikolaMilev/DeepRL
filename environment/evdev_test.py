from evdev import UInput, ecodes as e
import time
ui = UInput()

#https://media.readthedocs.org/pdf/python-evdev/latest/python-evdev.pdf#section.5.14


# accepts only KEY_* events by default
time.sleep(10)
ui.write(e.EV_KEY, e.KEY_LEFT, 1)  # KEY_A down
ui.write(e.EV_KEY, e.KEY_LEFT, 0)  # KEY_A up
ui.syn()
ui.write(e.EV_KEY, e.KEY_LEFT, 1)  # KEY_A down
ui.write(e.EV_KEY, e.KEY_LEFT, 0)  # KEY_A up
ui.syn()
ui.write(e.EV_KEY, e.KEY_LEFT, 1)  # KEY_A down
ui.write(e.EV_KEY, e.KEY_LEFT, 0)  # KEY_A up


ui.syn()

ui.close()
