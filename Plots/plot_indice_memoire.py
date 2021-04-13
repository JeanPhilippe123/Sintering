# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:57:18 2021

@author: jplan58
"""
import matplotlib.font_manager
from IPython.core.display import HTML

def make_html(fontname):
    return "<p>{font}: <span style='font-family:{font}; font-size: 24px;'>{font}</p>".format(font=fontname)

code = "\n".join([make_html(font) for font in sorted(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))])

HTML("<div style='column-count: 2;'>{}</div>".format(code))

import numpy as np
import matplotlib.pyplot as plt
import os
import csv 
from matplotlib.font_manager import FontProperties

def change_font(fontname,ax):
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(fontname) for label in labels]
    return
font = FontProperties()
font.set_name('Cambria')
font.set_size(20)

path = 'Warren-2008.csv'

myreader = csv.reader(open(path), delimiter=',')
data = []
for i, row in enumerate(myreader):
    try:
        try:
            row = [[float(row[0]),float(row[1])]]
            data += row
        except IndexError:
            #Sketch mais fonctionne
            index_real=np.array(data).transpose()
            wl = index_real[0]
            index_real = index_real[1]
            data = []
            pass
    except ValueError:
        pass
    
hfont = {'fontname':'Cambria'}
index_imag=np.array(data).transpose()
index_imag = index_imag[1]
# fitler longueur d'onde
filt = (wl>=0.3)&(wl<=2.8)
fig,ax=plt.subplots(figsize=(12,8))
ax.plot(wl[filt],index_real[filt],'k',linewidth=3)
ax2 = ax.twinx()
l = ax2.semilogy(wl[filt],index_imag[filt],'red',linewidth=3)
ax.set_xlabel('Longueur d\'onde ($\mu m$)',fontsize=20,fontproperties=font)
ax.set_ylabel('Indice de réfracion réelle',fontsize=20,fontproperties=font)
ax2.set_ylabel('Indice de réfracion imaginaire',fontsize=20,fontproperties=font)
ax.set_xticklabels([0,0.5,1.0,1.5,2.0,2.5],fontproperties=font)
ax.set_yticklabels([1.05,1.1,1.15,1.2,1.25,1.3],fontproperties=font)
ax.set_xlim([0.3,2.75])
change_font('Cambria',ax2)
# ax2.set_yticklabels(ax2.get_yticks(),fontproperties=font,fontsize=20)
ax2.tick_params(colors='red',labelsize=20)
ax2.spines['right'].set_color('red')
fig.tight_layout()
# 