CHAR_SET_LENGTH=95
import random

def char2pos(chr):
    return ord(chr)-ord(' ')

def pos2char(pos):
    return chr(pos + ord(' '))

