import bokeh.plotting as bkp
from bokeh.io import export_png, export_svgs
import numpy as np
import sys, os
import argparse
import hashlib
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from plotting import *

parser = argparse.ArgumentParser(description="Plots Riemannian linear regression experiments")
parser.add_argument('--X', type = str, default="Iterations", help="The X axis of the plot - one of Iterations/Coreset Size/Forward KL/Reverse KL/CPU Time(s)")
parser.add_argument('--X_scale', type=str, default = "linear", help = "Specifies the scale for the X-axis. Default is \"linear\". The most likely argument other than the default is \"log\" (base 10 logarithmic scaling). If the plot doesn't render properly (showing a memory error or failing to load) it may be because this parameter should be \"log\" but is still currently set to default.")
parser.add_argument('--Y', type = str, default = "Reverse KL", help="The Y axis of the plot - one of Iterations/Coreset Size/Forward KL/Reverse KL/CPU Time(s)")
parser.add_argument('--Y_scale', type=str, default = "log", help = "Specifies the scale for the Y-axis. Default is \"log\" (base 10 logarithmic scaling). The most likely argument other than the default is \"linear\".")

parser.add_argument('height', type=int, default=850, help = "Height of the plot's html canvas, default 850")
parser.add_argument('width', type=int, default=850, help = "Width of the plot's html canvas, default 850")

parser.add_argument('names', type = str, nargs = '+', default = ["SVI", "RAND", "GIGAO", "GIGAR"], help = "a list of which algorithm names to plot results for (Examples: SVI / GIGAO / GIGAR / RAND)")
trials = parser.add_mutually_exclusive_group(required=True)

trials.add_argument('--n_trials', type=int, help="Look for & plot experiments with trial IDs 1 through n_trials (inclusive)")
trials.add_argument('--seeds', type = int, nargs = '+', help="Plot experiments associated with the provided trial numbers (seeds)")

parser.add_argument('--fldr', type=str, default="results/", help="This script will look for & plot experiments in this folder")
parser.add_argument('--plot_every', type=int, default='1', help="Coarseness of the graph - will skip (plot_every-1) points between each plotted point")
parser.add_argument('--d', type=int, default = '200', help="The dimension of the multivariate normal distribution used for these experiments")
parser.add_argument('--M', type=int, default='200', help='The desired maximum coreset size specified for these experiments')
parser.add_argument('--N', type=int, default='1000', help='Dataset size/number of examples for these experiments')
parser.add_argument('--proj_dim', type=int, default = '100', help = "The number of samples taken when discretizing log likelihoods for these experiments")
parser.add_argument('--SVI_opt_itrs', type=int, default = '500', help = '(If using SVI/HOPS) The number of iterations used when optimizing weights.')
parser.add_argument('--optimizing', action='store_const', const = True, default= False, help = 'If this flag is provided, plots results for algorithms that have been optimized by the HOPS optimize() function after the coreset was constructed')
parser.add_argument('--SVI_step_sched', type=str, default = "lambda i : 1./(1+i)", help="Plots code with the associated step schedule (tuning rate) for SVI & HOPS. Default is \"lambda i : 1./(1+i)\", with the quotation marks.")
parser.add_argument('--pihat_noise', type=float, default=.75, help = "(If plotting GIGAR or simulating another realistically tuned Hilbert Coreset) - plots data corresponding to this much noise being introduced to the smoothed pi-hat to make the sampler")

arguments = parser.parse_args()
X = arguments.X
X_scale = arguments.X_scale
Y = arguments.Y
Y_scale = arguments.Y_scale
height = arguments.height
width = arguments.width

names = arguments.names
trials = np.arange(1, arguments.n_trials + 1) if arguments.n_trials else arguments.seeds
fldr = arguments.fldr
plot_every = arguments.plot_every
M = arguments.M
N = arguments.N
d = arguments.d
proj_dim = arguments.proj_dim
SVI_opt_itrs =  arguments.SVI_opt_itrs
optimizing = arguments.optimizing
pihat_noise = arguments.pihat_noise

algs = {'SVIEXACT': 'Sparse VI (Exact Tangent Space)',
        'SVI': 'Sparse VI', 
        'GIGAO': 'GIGA(Optimal)', 
        'GIGAR': "GIGA(Realistic)", 
        'RAND': "Uniform", 
        'HOPS': "HOPS",
        'HOPSEXACT': "HOPS (Exact Tangent Space)",
        'HOPS_full_scaling': "HOPS (Tempering from 0-1)",
        'HOPS_full_scaling_exact': "HOPS (Exact TS, Tempering from 0-1)"}
nms = []
for name in names:
  nms.append((name, algs[name]))

#plot the KL figure
fig = bkp.figure(y_axis_type=Y_scale, x_axis_type=X_scale, plot_width=width, plot_height=height, x_axis_label=X, y_axis_label=Y, toolbar_location=None )
preprocess_plot(fig, '32pt', False, True)

for i, nm in enumerate(nms):
  kl = []
  sz = []
  for tr in trials:
    numTuple = (nm[0], "tr="+str(tr), "N="+str(N), "d="+str(d), "proj_dim="+str(proj_dim), "optimizing="+str(optimizing), "SVI_opt_itrs="+str(SVI_opt_itrs), 'SVI_step_sched_hash_sha1='+hashlib.sha1(arguments.SVI_step_sched.encode('utf-8')).hexdigest(), 'pihat_noise='+str(pihat_noise))
    print(os.path.join(fldr, '_'.join(numTuple)+'.pk'))
    x_, mu0_, Sig0_, Sig_, mup_, Sigp_, w_, p_, muw_, Sigw_, rklw_, fklw_, cputs_, tr_, N_, d_, proj_dim_, optimizing_, SVI_opt_itrs_, SVI_step_sched_, pihat_noise_ = np.load(os.path.join(fldr, '_'.join(numTuple)+'.pk'), allow_pickle=True)
    data = { 'Iterations': [np.arange(1,len(rklw_)+1,plot_every)],
             'Coreset Size': [[np.count_nonzero(a) for a in w_[::plot_every]]],
             'Forward KL': [fklw_[::plot_every]],
             'Reverse KL': [rklw_[::plot_every]],
             'CPU Time(s)': [cputs_[::plot_every]]}
             
  x = np.percentile(data[X], 50, axis=0)
  fig.line(x, np.percentile(data[Y], 50, axis=0), color=pal[i-1], line_width=5, legend=nm[1])
  fig.patch(x = np.hstack((x, x[::-1])), y = np.hstack((np.percentile(data[Y], 75, axis=0), np.percentile(data[Y], 25, axis=0)[::-1])), color=pal[i-1], fill_alpha=0.4, legend=nm[1])

postprocess_plot(fig, '12pt', location='bottom_left', glyph_width=40)
fig.legend.background_fill_alpha=0.
fig.legend.border_line_alpha=0.
fig.legend.visible = True

bkp.show(fig)
