#! /bin/bash





if [ "$(whoami)" != "root" ]
then
	echo "Usage: sudo ./setup.sh";
else
	# making a group for uinput
	echo -e "\e[32mGroup setup...\e[39m"
	groupadd -f uinput &
	chgrp uinput /dev/uinput &
	chmod g+rw /dev/uinput  &
	usermod $USER -g uinput &
	echo -e "\e[32mGroup setup successful!\e[39m" ;

	# install pip
	echo -e "\e[32mInstalling pip...\e[39m" &
	apt-get install python-pip &
	echo -e "\e[32mSuccessfully installed pip!\e[39m" ;

	# install dependencies; it's quite possible that these have dependencies of their own
	# evdev : events ; link: http://python-evdev.readthedocs.io/en/latest/tutorial.html
	pip install evdev
	# PIL : for screenshots
	apt-get install python-pil

fi