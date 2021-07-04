#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""This file contains functions dedicated to helping find 9x9 Lights Out boards
that take the maximum number of moves to solve in an optimally.
"""


import numpy as np


# TODO: Is there a faster way to do this with numpy? Converting to strins a lot is prob bad
def add(a, b):
    """XORs two boards together, which is equivalent to adding the effect of two
    patterns in lights out. Returns the result as an 81-character string of 0's
    and 1's.
    """
    return ''.join(["0" if a[i] == b[i] else "1" for i in range(len(a))])

# After I generated the span of the null space, I don't need to again b/c it's expensive
# This gets every vector in the null space, but it might get duplicates
'''
def make_null_space(null_basis):
    null_space = null_basis[:]

    for i in range(len(null_basis)):
        for j in range(i+1, len(null_basis)):
            null_space.append(add(null_space[i], null_basis[j]))
     
    return null_space
'''

def min_clicks_9x9(b):
    """Given a board as an 81-character string of 0's and 1's representing a 9x9
    lights out pattern, this function finds the minimal number of clicks needed
    to solve it. It returns this minimal number of clicks along with the
    equivalent board using that number of clicks.
    """
    # 9x9 Null Space Basis
    # n1='100000001110000011101000101011101110000000000011101110101000101110000011100000001'
    # n2='010101000110101100010001010001110111000101000000011011000001000000000111000000010'
    # n3='000000101000001101000010100000111011001000100011010101101010001110001110100000100'
    # n4='010001010111011011000001010111000000010101010001101011000100010000011100000001000'
    # n5='101010100101010110000000101101011011101000101000110110100000100110111000100010000'
    # n6='010100010110110111010100000000000111010101010110101100010001000001110000000100000'
    # n7='001000001011100011100010101101010110001000100110111000001010000101100000101000000'
    # n8='000101010001101011010100010111011100000101000110110000000100000111000000010000000'

    # The entire null space: 255 Nulls: 2^8 - 1. The -1 is from the empty board
    # This list came from the commented out make_null_space function.
    null_space = [
    '100000001110000011101000101011101110000000000011101110101000101110000011100000001',
    '010101000110101100010001010001110111000101000000011011000001000000000111000000010',
    '000000101000001101000010100000111011001000100011010101101010001110001110100000100',
    '010001010111011011000001010111000000010101010001101011000100010000011100000001000',
    '101010100101010110000000101101011011101000101000110110100000100110111000100010000',
    '010100010110110111010100000000000111010101010110101100010001000001110000000100000',
    '001000001011100011100010101101010110001000100110111000001010000101100000101000000',
    '000101010001101011010100010111011100000101000110110000000100000111000000010000000',
    '110101001000101111111001111010011001000101000011110101101001101110000100100000011',
    '100000100110001110101010001011010101001000100000111011000010100000001101000000101',
    '110001011001011000101001111100101110010101010010000101101100111110011111100001001',
    '001010101011010101101000000110110101101000101011011000001000001000111011000010001',
    '110100011000110100111100101011101001010101010101000010111001101111110011100100001',
    '101000000101100000001010000110111000001000100101010110100010101011100011001000001',
    '100101011111101000111100111100110010000101000101011110101100101001000011110000001',
    '010101101110100001010011110001001100001101100011001110101011001110001001100000110',
    '000100010001110111010000000110110111010000010001110000000101010000011011000001010',
    '111111100011111010010001111100101100101101101000101101100001100110111111100010010',
    '000001010000011011000101010001110000010000010110110111010000000001110111000100010',
    '011101001101001111110011111100100001001101100110100011001011000101100111101000010',
    '010000010111000111000101000110101011000000000110101011000101000111000111010000010',
    '110101100000100010111011011010100010001101100000100000000011100000001010000000111',
    '100100011111110100111000101101011001010000010010011110101101111110011000100001011',
    '011111101101111001111001010111000010101101101011000011001001001000111100000010011',
    '100001011110011000101101111010011110010000010101011001111000101111110100100100011',
    '111101000011001100011011010111001111001101100101001101100011101011100100001000011',
    '110000011001000100101101101101000101000000000101000101101101101001000100110000011',
    '010001111111010110000011110111111011011101110010111110101110011110010010100001100',
    '101010001101011011000010001101100000100000001011100011001010101000110110000010100',
    '010100111110111010010110100000111100011101110101111001111011001111111110100100100',
    '001000100011101110100000001101101101000000000101101101100000001011101110001000100',
    '000101111001100110010110110111100111001101100101100101101110001001001110110000100',
    '110001110001010101101011011100010101011101110001010000000110110000010001000001101',
    '001010000011011000101010100110001110100000001000001101100010000110110101100010101',
    '110100110000111001111110001011010010011101110110010111010011100001111101000100101',
    '101000101101101101001000100110000011000000000110000011001000100101101101101000101',
    '100101110111100101111110011100001001001101100110001011000110100111001101010000101',
    '000100111001111010010010100110001100011000110010100101101111011110010101100001110',
    '111111001011110111010011011100010111100101001011111000001011101000110001000010110',
    '000001111000010110000111110001001011011000110101100010111010001111111001100100110',
    '011101100101000010110001011100011010000101000101110110100001001011101001001000110',
    '010000111111001010000111100110010000001000100101111110101111001001001001110000110',
    '100100110111111001111010001101100010011000110001001011000111110000010110000001111',
    '011111000101110100111011110111111001100101001000010110100011000110110010100010111',
    '100001110110010101101111011010100101011000110110001100010010100001111010000100111',
    '111101101011000001011001110111110100000101000110011000001001100101101010101000111',
    '110000110001001001101111001101111110001000100110010000000111100111001010010000111',
    '111011110010001101000001111010011011111101111001011101100100110110100100100011000',
    '000101000001101100010101010111000111000000000111000111010101010001101100000101000',
    '011001011100111000100011111010010110011101110111010011001110010101111100101001000',
    '010100000110110000010101000000011100010000010111011011000000010111011100010001000',
    '011011111100001110101001010001110101111101111010110011001100011000100111000011001',
    '100101001111101111111101111100101001000000000100101001111101111111101111100101001',
    '111001010010111011001011010001111000011101110100111101100110111011111111001001001',
    '110100001000110011111101101011110010010000010100110101101000111001011111110001001',
    '101110110100100001010000101011101100111000111001000110100101110110100011100011010',
    '010000000111000000000100000110110000000101000111011100010100010001101011000101010',
    '001100011010010100110010101011100001011000110111001000001111010101111011101001010',
    '000001000000011100000100010001101011010101010111000000000001010111011011010001010',
    '001110111010100010111000000000000010111000111010101000001101011000100000000011011',
    '110000001001000011101100101101011110000101000100110010111100111111101000100101011',
    '101100010100010111011010000000001111011000110100100110100111111011111000001001011',
    '100001001110011111101100111010000101010101010100101110101001111001011000110001011',
    '111011011010000000000011011010100000110101011010001000001110111000101010000011100',
    '000101101001100001010111110111111100001000100100010010111111011111100010100101100',
    '011001110100110101100001011010101101010101010100000110100100011011110010001001100',
    '010100101110111101010111100000100111011000110100001110101010011001010010110001100',
    '011011010100000011101011110001001110110101011001100110100110010110101001100011101',
    '100101100111100010111111011100010010001000100111111100010111110001100001000101101',
    '111001111010110110001001110001000011010101010111101000001100110101110001101001101',
    '110100100000111110111111001011001001011000110111100000000010110111010001010001101',
    '101110011100101100010010001011010111110000011010010011001111111000101101000011110',
    '010000101111001101000110100110001011001101100100001001111110011111100101100101110',
    '001100110010011001110000001011011010010000010100011101100101011011110101001001110',
    '000001101000010001000110110001010000011101110100010101101011011001010101110001110',
    '001110010010101111111010100000111001110000011001111101100111010110101110100011111',
    '110000100001001110101110001101100101001101100111100111010110110001100110000101111',
    '101100111100011010011000100000110100010000010111110011001101110101110110101001111',
    '100001100110010010101110011010111110011101110111111011000011110111010110010001111',
    '111110110011100001010100101101011100111101111110011010110001100111001000100110000',
    '100010101110110101100010000000001101100000001110001110101010100011011000001010000',
    '101111110100111101010100111010000111101101101110000110100100100001111000110010000',
    '011110111101100010111100000110110010111101111101110100011001001001001011000110001',
    '000010100000110110001010101011100011100000001101100000000010001101011011101010001',
    '001111111010111110111100010001101001101101101101101000001100001111111011010010001',
    '101011110101001101000101111100101011111000111110000001110000100111001111100110010',
    '110111101000011001110011010001111010100101001110010101101011100011011111001010010',
    '111010110010010001000101101011110000101000101110011101100101100001111111110010010',
    '001011111011001110101101010111000101111000111101101111011000001001001100000110011',
    '010111100110011010011011111010010100100101001101111011000011001101011100101010011',
    '011010111100010010101101000000011110101000101101110011001101001111111100010010011',
    '111110011011101100010110001101100111110101011101001111011011101001000110000110100',
    '100010000110111000100000100000110110101000101101011011000000101101010110101010100',
    '101111011100110000010110011010111100100101001101010011001110101111110110010010100',
    '011110010101101111111110100110001001110101011110100001110011000111000101100110101',
    '000010001000111011001000001011011000101000101110110101101000000011010101001010101',
    '001111010010110011111110110001010010100101001110111101100110000001110101110010101',
    '101011011101000000000111011100010000110000011101010100011010101001000001000110110',
    '110111000000010100110001110001000001101101101101000000000001101101010001101010110',
    '111010011010011100000111001011001011100000001101001000001111101111110001010010110',
    '001011010011000011101111110111111110110000011110111010110010000111000010100110111',
    '010111001110010111011001011010101111101101101110101110101001000011010010001010111',
    '011010010100011111101111100000100101100000001110100110100111000001110010110010111',
    '101111100100111010010101111010011100101000101111110001110101110111010100100111000',
    '110011111001101110100011010111001101110101011111100101101110110011000100001011000',
    '111110100011100110010101101101000111111000111111101101100000110001100100110011000',
    '001111101010111001111101010001110010101000101100011111011101011001010111000111001',
    '010011110111101101001011111100100011110101011100001011000110011101000111101011001',
    '011110101101100101111101000110101001111000111100000011001000011111100111010011001',
    '111010100010010110000100101011101011101101101111101010110100110111010011100111010',
    '100110111111000010110010000110111010110000011111111110101111110011000011001011010',
    '101011100101001010000100111100110000111101111111110110100001110001100011110011010',
    '011010101100010101101100000000000101101101101100000100011100011001010000000111011',
    '000110110001000001011010101101010100110000011100010000000111011101000000101011011',
    '001011101011001001101100010111011110111101111100011000001001011111100000010011011',
    '101111001100110111010111011010100111100000001100100100011111111001011010000111100',
    '110011010001100011100001110111110110111101111100110000000100111101001010101011100',
    '111110001011101011010111001101111100110000011100111000001010111111101010010011100',
    '001111000010110100111111110001001001100000001111001010110111010111011001100111101',
    '010011011111100000001001011100011000111101111111011110101100010011001001001011101',
    '011110000101101000111111100110010010110000011111010110100010010001101001110011101',
    '111010001010011011000110001011010000100101001100111111011110111001011101000111110',
    '100110010111001111110000100110000001111000111100101011000101111101001101101011110',
    '101011001101000111000110011100001011110101011100100011001011111111101101010011110',
    '011010000100011000101110100000111110100101001111010001110110010111011110100111111',
    '000110011001001100011000001101101111111000111111000101101101010011001110001011111',
    '001011000011000100101110110111100101110101011111001101100011010001101110110011111',
    '011100011101010100110110101101010001011101110000010100011011000100010000101100000',
    '010001000111011100000000010111011011010000010000011100010101000110110000010100000',
    '111100010011010111011110000110111111011101110011111010110011101010010011001100001',
    '110001001001011111101000111100110101010000010011110010111101101000110011110100001',
    '001001011011111000100111111100100110011000110000001111011010000100010111101100010',
    '000100000001110000010001000110101100010101010000000111010100000110110111010100010',
    '101001010101111011001111010111001000011000110011100001110010101010010100001100011',
    '100100001111110011111001101101000010010101010011101001111100101000110100110100011',
    '011100110101011001110100001101101010010101010011000001110001001010011110001100100',
    '010001101111010001000010110111100000011000110011001001111111001000111110110100100',
    '111100111011011010011100100110000100010101010000101111011001100100011101101100101',
    '110001100001010010101010011100001110011000110000100111010111100110111101010100101',
    '001001110011110101100101011100011101010000010011011010110000001010011001001100110',
    '000100101001111101010011100110010111011101110011010010111110001000111001110100110',
    '101001111101110110001101110111110011010000010000110100011000100100011010101100111',
    '100100100111111110111011001101111001011101110000111100010110100110111010010100111',
    '001101001010001111110111111010010001001000100001111111011111010100001100101101000',
    '000000010000000111000001000000011011000101000001110111010001010110101100010101000',
    '101101000100001100011111010001111111001000100010010001110111111010001111001101001',
    '100000011110000100101001101011110101000101000010011001111001111000101111110101001',
    '011000001100100011100110101011100110001101100001100100011110010100001011101101010',
    '010101010110101011010000010001101100000000000001101100010000010110101011010101010',
    '111000000010100000001110000000001000001101100010001010110110111010001000001101011',
    '110101011000101000111000111010000010000000000010000010111000111000101000110101011',
    '001101100010000010110101011010101010000000000010101010110101011010000010001101100',
    '000000111000001010000011100000100000001101100010100010111011011000100010110101100',
    '101101101100000001011101110001000100000000000001000100011101110100000001101101101',
    '100000110110001001101011001011001110001101100001001100010011110110100001010101101',
    '011000100100101110100100001011011101000101000010110001110100011010000101001101110',
    '010101111110100110010010110001010111001000100010111001111010011000100101110101110',
    '111000101010101101001100100000110011000101000001011111011100110100000110101101111',
    '110101110000100101111010011010111001001000100001010111010010110110100110010101111',
    '110110111000000010110110000000001010110101011000100010111011100010101000001110000',
    '111011100010001010000000111010000000111000111000101010110101100000001000110110000',
    '010110110110000001011110101011100100110101011011001100010011001100101011101110001',
    '011011101100001001101000010001101110111000111011000100011101001110001011010110001',
    '100011111110101110100111010001111101110000011000111001111010100010101111001110010',
    '101110100100100110010001101011110111111101111000110001110100100000001111110110010',
    '000011110000101101001111111010010011110000011011010111010010001100101100101110011',
    '001110101010100101111001000000011001111101111011011111011100001110001100010110011',
    '110110010000001111110100100000110001111101111011110111010001101100100110101110100',
    '111011001010000111000010011010111011110000011011111111011111101110000110010110100',
    '010110011110001100011100001011011111111101111000011001111001000010100101001110101',
    '011011000100000100101010110001010101110000011000010001110111000000000101110110101',
    '100011010110100011100101110001000110111000111011101100010000101100100001101110110',
    '101110001100101011010011001011001100110101011011100100011110101110000001010110110',
    '000011011000100000001101011010101000111000111000000010111000000010100010001110111',
    '001110000010101000111011100000100010110101011000001010110110000000000010110110111',
    '100111101111011001110111010111001010100000001001001001111111110010110100001111000',
    '101010110101010001000001101101000000101101101001000001110001110000010100110111000',
    '000111100001011010011111111100100100100000001010100111010111011100110111101111001',
    '001010111011010010101001000110101110101101101010101111011001011110010111010111001',
    '110010101001110101100110000110111101100101001001010010111110110010110011001111010',
    '111111110011111101010000111100110111101000101001011010110000110000010011110111010',
    '010010100111110110001110101101010011100101001010111100010110011100110000101111011',
    '011111111101111110111000010111011001101000101010110100011000011110010000010111011',
    '100111000111010100110101110111110001101000101010011100010101111100111010101111100',
    '101010011101011100000011001101111011100101001010010100011011111110011010010111100',
    '000111001001010111011101011100011111101000101001110010111101010010111001001111101',
    '001010010011011111101011100110010101100101001001111010110011010000011001110111101',
    '110010000001111000100100100110000110101101101010000111010100111100111101101111110',
    '111111011011110000010010011100001100100000001010001111011010111110011101010111110',
    '010010001111111011001100001101101000101101101001101001111100010010111110001111111',
    '011111010101110011111010110111100010100000001001100001110010010000011110110111111',
    '001101011010001000110110111010001010001101100000001000001110000010100000111000000',
    '101101010100001011011110010001100100001101100011100110100110101100100011011000001',
    '011000011100100100100111101011111101001000100000010011001111000010100111111000010',
    '111000010010100111001111000000010011001000100011111101100111101100100100011000011',
    '001101110010000101110100011010110001000101000011011101100100001100101110011000100',
    '101101111100000110011100110001011111000101000000110011001100100010101101111000101',
    '011000110100101001100101001011000110000000000011000110100101001100101001011000110',
    '111000111010101010001101100000101000000000000000101000001101100010101010111000111',
    '011100001101010011110111101101001010011000110001100011001010010010111100111001000',
    '111100000011010000011111000110100100011000110010001101100010111100111111011001001',
    '001001001011111111100110111100111101011101110001111000001011010010111011111001010',
    '101001000101111100001110010111010011011101110010010110100011111100111000011001011',
    '011100100101011110110101001101110001010000010010110110100000011100110010011001100',
    '111100101011011101011101100110011111010000010001011000001000110010110001111001101',
    '001001100011110010100100011100000110010101010010101101100001011100110101011001110',
    '101001101101110001001100110111101000010101010001000011001001110010110110111001111',
    '100111111111011110110110010111010001100101001000111110101110100100011000011010000',
    '000111110001011101011110111100111111100101001011010000000110001010011011111010001',
    '110010111001110010100111000110100110100000001000100101101111100100011111011010010',
    '010010110111110001001111101101001000100000001011001011000111001010011100111010011',
    '100111010111010011110100110111101010101101101011101011000100101010010110111010100',
    '000111011001010000011100011100000100101101101000000101101100000100010101011010101',
    '110010010001111111100101100110011101101000101011110000000101101010010001111010110',
    '010010011111111100001101001101110011101000101000011110101101000100010010011010111',
    '110110101000000101110111000000010001110000011001010101101010110100000100011011000',
    '010110100110000110011111101011111111110000011010111011000010011010000111111011001',
    '100011101110101001100110010001100110110101011001001110101011110100000011011011010',
    '000011100000101010001110111010001000110101011010100000000011011010000000111011011',
    '110110000000001000110101100000101010111000111010000000000000111010001010111011100',
    '010110001110001011011101001011000100111000111001101110101000010100001001011011101',
    '100011000110100100100100110001011101111101111010011011000001111010001101111011110',
    '000011001000100111001100011010110011111101111001110101101001010100001110011011111',
    '011001001100111111100010111010001101011000110110100100011111000011010000111100000',
    '111001000010111100001010010001100011011000110101001010110111101101010011011100001',
    '001100001010010011110011101011111010011101110110111111011110000011010111111100010',
    '101100000100010000011011000000010100011101110101010001110110101101010100011100011',
    '011001100100110010100000011010110110010000010101110001110101001101011110011100100',
    '111001101010110001001000110001011000010000010110011111011101100011011101111100101',
    '001100100010011110110001001011000001010101010101101010110100001101011001011100110',
    '101100101100011101011001100000101111010101010110000100011100100011011010111100111',
    '001000011011100100100011101101001101001101100111001111011011010011001100111101000',
    '101000010101100111001011000110100011001101100100100001110011111101001111011101001',
    '011101011101001000110010111100111010001000100111010100011010010011001011111101010',
    '111101010011001011011010010111010100001000100100111010110010111101001000011101011',
    '001000110011101001100001001101110110000101000100011010110001011101000010011101100',
    '101000111101101010001001100110011000000101000111110100011001110011000001111101101',
    '011101110101000101110000011100000001000000000100000001110000011101000101011101110',
    '111101111011000110011000110111101111000000000111101111011000110011000110111101111',
    '110011101001101001100010010111010110110000011110010010111111100101101000011110000',
    '010011100111101010001010111100111000110000011101111100010111001011101011111110001',
    '100110101111000101110011000110100001110101011110001001111110100101101111011110010',
    '000110100001000110011011101101001111110101011101100111010110001011101100111110011',
    '110011000001100100100000110111101101111000111101000111010101101011100110111110100',
    '010011001111100111001000011100000011111000111110101001111101000101100101011110101',
    '100110000111001000110001100110011010111101111101011100010100101011100001111110110',
    '000110001001001011011001001101110100111101111110110010111100000101100010011110111',
    '100010111110110010100011000000010110100101001111111001111011110101110100011111000',
    '000010110000110001001011101011111000100101001100010111010011011011110111111111001',
    '110111111000011110110010010001100001100000001111100010111010110101110011011111010',
    '010111110110011101011010111010001111100000001100001100010010011011110000111111011',
    '100010010110111111100001100000101101101101101100101100010001111011111010111111100',
    '000010011000111100001001001011000011101101101111000010111001010101111001011111101',
    '110111010000010011110000110001011010101000101100110111010000111011111101111111110',
    '010111011110010000011000011010110100101000101111011001111000010101111110011111111']

    min_board = b
    min_clicks = b.count("1")
    
    for null_board in null_space:
        nulled_board = add(b, null_board)

        clicks = nulled_board.count("1")
        if clicks < min_clicks:
            min_clicks = clicks
            min_board = nulled_board
    
    return (min_clicks, min_board)

def str_to_arr(arr_str):
    """Converts an 81 character string of 0's and 1's to a 9x9 numpy array."""
    return np.array(list(arr_str)).reshape((9,9))

def arr_to_str(mat):
    """Converts a 9x9 number array to an 81 character string of 0's and 1's."""
    return ''.join(mat.flatten().tolist())

def get_combinations(mat):
    """Given a 9x9 numpy array, this function returns all rotations and
    reflection combinations of that array. The result is returned as a set of
    81-character strings.
    """
    perms = [mat]
    for i in range(3):  # mat + 3 rotated ways
        perms.append(np.rot90(perms[-1]))

    l = len(perms)
    for i in range(l):  # reflected versions
        perms.append(np.flip(perms[i], 0))
        perms.append(np.flip(perms[i], 1))
        perms.append(np.flip(perms[i], (0,1)))
        perms.append(np.flip(perms[i], (1,0)))  # this line might be redundant

    return set([arr_to_str(perm) for perm in perms])

# Finds perms of boards that use the same number of clicks
def find_equal_combinations(n, filename="37Clicks.txt"):
    """Given a file of boards as 81-character strings: 1 per line, this function
    finds the equivalent boards that require at least some given number of clicks
    to solve. The result is returned as a set of 81-character strings.
    """
    
    equiv_combos = set()  # A set: {} is a dict
    
    with open(filename) as f:
        for l in f:
            line = l[:-1]  # gets rid of newline character
            
            combos = get_combinations(str_to_arr(line))
            for rotated_board in combos:
                min_clicks, min_board = min_clicks_9x9(rotated_board)
                
                if min_clicks >= n:
                    equiv_combos.add(min_board)

    for equiv_combo in equiv_combos:
        print(equiv_combo)

    return equiv_combos

# Finds boards one click away that use more clicks
def find_bigger_boards(n, size=81, filename="37Clicks.txt"):
    """Given a file of boards as 81-character strings: 1 per line, this function
    finds boards that require one more click to solve than the one in the file
    by attempting to replace a single 0 in the board with a 1. The result is
    returned as a set of 81 character strings.
    """
    better_boards = set()  # A set: {} is a dict
    
    with open(filename) as f:
        for l in f:
            line = l[:-1]  # gets rid of new line character

            for i,character in enumerate(line):
                if character == '0': #replace a single 0 with a 1
                    my_line = line[:i] + "1" + line[i+1:]

                    min_clicks, min_board = min_clicks_9x9(my_line)
                    if min_clicks >= n:
                        better_boards.add(min_board)

    for board in better_boards:
        print(board)

    return better_boards

def wolfram_visual_code(board_str):
    """Given an 81 character string of 0's and 1's this function prints code in
    the Wolfram Language (similar to Mathematica) that can be used to visualize
    the board.
    """
    print("MatrixPlot[ArrayReshape[{", end="")
    for char in board_str:
        print(char+",", end="")
    print("},{9,9}]]")
