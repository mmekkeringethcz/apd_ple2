# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:55:30 2020

@author: rober
"""

from pymeasure.instruments.agilent import Agilent33500

generator = Agilent33500('USB0::0x0957::0x1607::MY50003120::0::INSTR')

generator.shape = 'TRI'

generator.frequency = 200

generator.amplitude = 0.5

generator.output = 'off'
