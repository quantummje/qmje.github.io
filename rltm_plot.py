### Finds and sorts individual phase defects [vortices] in 2+1 dimensions.
### Produces (x,y,t) trajectory plot [see Fig. 4 Phys. Rev. A 102, 011303(R) (2020)]
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import sys
#
def vort_sort(dat_in):
    #
    out = np.zeros((dat_in.shape[0],dat_in.shape[1],dat_in.shape[2]))
    #
    for x in range(0,dat_in.shape[1]):
        #
        this_vort = x
        out[0][this_vort,:] = dat_in[0][this_vort,:]
        #
        for t in range(1,dat_in.shape[0]):
            #
            r_tmp = []
            #
            for y in range(0,dat_in.shape[1]):
                #
                rn = np.sqrt((dat_in[t][y,0]-out[t-1][this_vort,0])**2 + (dat_in[t][y,1]-out[t-1][this_vort,1])**2)
                r_tmp.append(rn)
                #
            next_vort = r_tmp.index(min(r_tmp))
            #
            out[t][this_vort,:] = dat_in[t][next_vort,:]
            #
        #
    #
    return out
#
dat = np.load('data/rt_igt_om_0_c_10_lz_0_ep_100.npz', encoding='latin1', allow_pickle=True)
tsc = np.linspace(0,dat['T'],dat['vp_tim'].shape[0])
#
dat_tim = np.zeros((dat['vp_tim'].shape[0],dat['vp_tim'][0].shape[0],dat['vp_tim'][0].shape[1]))
#
for ii in range(0,dat_tim.shape[0]):
    #
    if dat['vp_tim'][ii].shape[0] == dat['vp_tim'][0].shape[0]:
        dat_tim[ii] = dat['vp_tim'][ii]
    else:
        dat_tim[ii] = np.nan
    #
#
vs = vort_sort(dat_tim[:])
#
freeD = plt.figure(figsize=(4,7))
ax = freeD.gca(projection='3d')
for jj in range(0,vs.shape[1]):
    #
    if jj == 0:
        ax.plot(vs[:,jj,0],vs[:,jj,1],tsc,marker='o',label=r'$\Omega=0,\ \tilde{C}=0$')
        #ax.plot(vs1[:,jj,0],vs1[:,jj,1],tsc,marker='o',label=r'$\Omega=0,\ \tilde{C}=10$')
    else:
        ax.plot(vs[:,jj,0],vs[:,jj,1],tsc,marker='o',label=r'$\Omega=0,\ \tilde{C}=0$')
        #ax.plot(vs1[:,jj,0],vs1[:,jj,1],tsc,marker='o',label=r'$\Omega=0,\ \tilde{C}=10$')
    #
#
ax.set_zlim([0,tsc[-1]/1.])
ax.set_xlabel(r'$x/a_x$')
ax.set_ylabel(r'$y/a_x$')
ax.set_zlabel(r'$\omega_x t$')
ax.set_xlim([-dat['Dx'][0],dat['Dx'][0]])
ax.set_ylim([-dat['Dy'][0],dat['Dy'][0]])
ax.view_init(elev=10., azim=45.)
ax.legend()
#plt.tight_layout()
freeD.show()
#
oneD = plt.figure()
col = ['tab:red', 'tab:green', 'tab:blue','tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'gold']
for i,v in enumerate(range(dat_tim.shape[1])):
    v = v + 1
    ax1 = subplot(dat_tim.shape[1],1,v)
    ax1.plot(tsc[0:vs.shape[0]],vs[:,v-1,0],'.-',markersize='8',markeredgecolor=col[v-1],color='gray',label='x')
    ax1.plot(tsc[0:vs.shape[0]],vs[:,v-1,1],'+-',markersize='8',markeredgecolor=col[v-1],color='gray',label='y')
    ax1.grid()
    ax1.set_xlim([0,tsc[-1]])
    ax1.legend()
    #
    if v != dat_tim.shape[1]:
        ax1.set_xticklabels([])
    #
ax1.set_xlabel(r'$t$')
plt.subplots_adjust(hspace=0)
oneD.show()
#
