import bokeh.plotting as bkp
import numpy as np
import sys, os
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from plotting import *


plot_reverse_kl = True
size_x_axis = False
trials = np.arange(10)
Ms = np.arange(30)
nms = [('SVI1', 'SparseVI-1'), ('SVIF', 'SparseVI-Full'), ('GIGAT', 'GIGA (Truth)'), ('GIGAN', 'GIGA (Noisy)'), ('IH', 'HOPS'), ('RAND', 'Uniform')]


#plot the KL figure
fig = bkp.figure(y_axis_type='log', plot_width=750, plot_height=750, x_axis_label=('Coreset Size' if size_x_axis else 'Coreset Size'), y_axis_label=('Reverse KL' if plot_reverse_kl else 'Forward KL') )
preprocess_plot(fig, '32pt', False, True)

for i, nm in enumerate(nms):
  kl = []
  klopt = []
  for t in trials:
    res = np.load('results/results_'+nm[0]+'_' + str(t)+'.npz')
    if plot_reverse_kl:
      klt = res['rklw']
      kloptt = res['rklw_opt']
    else:
      klt = res['fklw']
      kloptt = res['fklw_opt']
    sz = (res['w'] > 0).sum(axis=1) 
    szopt = (res['w_opt'] > 0).sum(axis=1) 
    if size_x_axis:
      kl.append(np.interp(Ms, sz, klt))
      klopt.append(np.interp(Ms, szopt, kloptt))
    else:
      kl.append(klt)
      klopt.append(kloptt)
    #fig.scatter(sz[-1], kl[-1], color=pal[i], legend=nm) 
    #fig.scatter(szopt[-1], klopt[-1], color=pal[i], legend=nm) 
  if size_x_axis:
    fig.line(Ms, np.maximum(np.percentile(kl, 50, axis=0), 1e-16), color=pal[i], line_width=5, line_dash='dashed', legend=nm[1]) 
    fig.line(Ms, np.maximum(np.percentile(klopt, 50, axis=0), 1e-16), color=pal[i], line_width=5, line_dash='solid', legend=nm[1]) 
  else:
    kl = np.array(kl)
    klopt = np.array(klopt)
    #for j in range(kl.shape[0]):
    #  fig.line(np.arange(kl.shape[1]), kl[j, :], color=pal[i], line_width=5, line_dash='dashed', legend=nm[1])
    #  fig.line(np.arange(kl.shape[1]), klopt[j, :], color=pal[i], line_width=5, line_dash='solid', legend=nm[1])
    fig.line(np.arange(kl.shape[1]), np.maximum(np.percentile(kl, 50, axis=0), 1e-16), color=pal[i], line_width=5, line_dash='dashed', legend=nm[1])
    fig.line(np.arange(kl.shape[1]), np.maximum(np.percentile(klopt, 50, axis=0), 1e-16), color=pal[i], line_width=5, line_dash='solid', legend=nm[1])

postprocess_plot(fig, '22pt', location='bottom_left', glyph_width=40)
fig.legend.background_fill_alpha=0.
fig.legend.border_line_alpha=0.

bkp.show(fig)



