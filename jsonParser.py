# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:56:01 2019

@author: rglabor07
"""
import numpy as np
import json as js

import os

class Parser:
    
    def __init__(self, filePath):
        self.filePath = filePath
        with open(filePath) as json_file:
            self.file = js.load(json_file)
            
    def get_2D_data(self):
        values2D = []
        for value in self.file['2dBoundingBox']:
            values2D.append(value)
        return values2D
    
    def get_3D_data(self):
        values3D = []
        for value in self.file['3dBoundingBox']:
            values3D.append(value)
        return values3D
    
    def get_centerPoint(self):
        centPoint = []
        for value in self.file['centerPoint']:
            centPoint.append(value)
        return centPoint

