

# this is supposed to be a circular buffer of the length no greater than MAX_EXP_LEN
# see configuration.py for exact MAX_EXP_LEN
# API: must have these features:
#   -- add to buffer, erasing the oldest if needed
#   -- pick random k from the sequence
MAX_EXP_LEN=configuration