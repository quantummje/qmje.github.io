### Finds the ground state of a trapped BEC with rigid body + nonlinear roation
### with the imaginary time method. See Phys. Rev A 102, 011303(R) (2020).
import numpy as np
import sys
import time
#
def kinm(psi):
    #
    kin[1:-1,1:-1] = psi[:-2,1:-1] + psi[2:,1:-1] + psi[1:-1,:-2] + psi[1:-1,2:] - 4*psi[1:-1,1:-1]
    kin[:,0] = kin[:,-1] = kin[0,:] = kin[-1,:] = 0.0
    #
    return kin
#
def momm(psi):
    #
    px[1:-1,1:-1] = psi[:-2,1:-1] - psi[2:,1:-1]
    py[1:-1,1:-1] = psi[1:-1,:-2] - psi[1:-1,2:]
    #
    px[:,0] = px[:,-1] = px[0,:] = px[-1,:] = 0.0
    py[:,0] = py[:,-1] = py[0,:] = py[-1,:] = 0.0
    #
    return px, py
#
def phi(xx,yy,r):
    #
    phi_out = np.arctan2(np.float64(yy)-r[1],np.float64(xx)-r[0])
    #
    return phi_out

def get_en(psi,x,y,dx,dy,U,g,Om):
    #
    ke = 0
    te = 0
    ie = 0
    am = 0
    #
    for ii in xrange(1,psi.shape[1]-1):
        #
        for jj in xrange(1,psi.shape[0]-1):
            #
            mom_x = (psi[jj,ii+1] - psi[jj,ii-1])/(2.0*dx)
            mom_y = (psi[jj+1,ii] - psi[jj-1,ii])/(2.0*dy)
            #
            ke = ke + 0.5*(np.abs(mom_x)**2 + np.abs(mom_y)**2)
            #
            te = te + U[jj][ii]*np.abs(psi[jj][ii])**2
            ie = ie + 0.5*g*np.abs(psi[jj][ii])**4
            #
            am = am + 1j*np.conj(psi[jj][ii])*(Om[0] + Om[1]*np.abs(psi[jj][ii])**2)*(x[jj]*mom_y - y[ii]*mom_x)
            #
        #
    #
    nm = (dx*dy)*(psi*np.conj(psi)).sum()
    en = (dx*dy)*(ke + te + ie + np.real(am))/nm
    #
    return en
#
def track(psi,x,y,dx,dy,g,ep):
    #
    ry, rx = (np.sqrt(2)*pow(g*ep/np.pi,0.25),(np.sqrt(2)/ep)*pow(g*ep/np.pi,0.25))
    #
    den = np.abs(psi)**2
    vort_pos = [[],[]]
    #
    for ii in range(0,psi.shape[0]):
        #
        for jj in range(0,psi.shape[1]):
            #
            if (x[ii]/rx)**2 + (y[jj]/ry)**2 <= 1:
                #
                if (den[ii-1][jj] > den[ii][jj] < den[ii+1][jj]) and (den[ii][jj-1] > den[ii][jj] < den[ii][jj+1]):
                    #
                    vort_pos = np.append(vort_pos,[[x[ii]],[y[jj]]],axis=1)
                    #
                #
            #
        #
    #
    return vort_pos
#
def imt(Dt,Dx,Dy,tol,g,Om,ep):
    #
    x = np.arange(-Dx[0],Dx[0],Dx[1])
    y = np.arange(-Dy[0],Dy[0],Dx[1])
    #
    yy, xx = np.complex128(np.meshgrid(x, y))
    dx = Dx[1]; dy = Dy[1];
    #
    Nx = x.shape[0]
    Ny = y.shape[0]
    #
    psi0 = np.ones((Nx,Ny)) + 1e-3*np.random.rand(Nx,Ny) + 1j*1e-3*np.random.rand(Nx,Ny)
    psi = psi0/np.sqrt(dx*dy*(psi0*np.conj(psi0)).sum())
    psi_new = np.zeros((Nx,Ny),dtype=psi.dtype)
    #
    en_er = np.random.rand(1)
    en_old = np.random.rand(1)
    #
    en_store = []
    am_store = []
    ee_store = []
    er_store = []
    nv_store = []
    #
    U = 0.5*(xx**2 + (ep*yy)**2).T
    #
    tt = 0
    tt_min = 5e5
    run_flag = True
    t0 = time.time()
    dd = 0.5/dx**2
    xxT = xx.T
    yyT = yy.T
    psi_phi = np.exp(1j*(phi(xx,yy,[1,0])+phi(xx,yy,[-2,0])))
    #
    while run_flag:
        #
        px, py = momm(psi)
        Lz = 1j*(yyT*py - xxT*px)/(2*dx)
        #
        psi_new = psi -Dt*(-dd*kinm(psi) + U*psi - Om[0]*Lz + (g*psi - Om[1]*Lz)*psi*np.conj(psi))
        #
        if divmod(tt,5e4)[1]==0 and tt != 0:
            #
            en = get_en(psi=psi_new,x=x,y=y,dx=Dx[1],dy=Dy[1],U=U,g=g,Om=Om)
            en_er = np.abs((en-en_old)/en)
            en_old = en
            #
            if tt <= tt_min:
                #
                run_flag = True
                #
            elif tt > tt_min and en_er < tol:
                #
                run_flag = False
                #
            #
            px, py = momm(psi_new)
            #
            Lz_new = 1j*(yyT*py - xxT*px)/(2*dx)
            am = dx*dy*(np.conj(psi_new)*Lz_new).sum()
            vort_pos = track(psi=psi_new,x=x,y=y,dx=Dx[1],dy=Dy[1],g=g,ep=ep)
            #
            sig_x = np.real(np.sqrt(dx*dy*((yyT**2)*psi_new*np.conj(psi_new)).sum()))
            sig_y = np.real(np.sqrt(dx*dy*((xxT**2)*psi_new*np.conj(psi_new)).sum()))
            #
            en_store.append(en)
            am_store.append(am)
            ee_store.append(sig_y/sig_x)
            er_store.append(en_er)
            #
            if (len(vort_pos[0]) != 0) and (len(vort_pos[1]) != 0):
                #
                nv_store.append(vort_pos.shape[1])
                nv_out = vort_pos.shape[1]
                #
            else:
                #
                nv_store.append(0)
                nv_out = 0
                #
            #
            print('-----------------------')
            print('Energy: ' + repr(en))
            print('Error: ' + repr(en_er))
            print('Ang. Mom. :' + repr(am))
            print('There are ' + repr(nv_out) + ' vortices')
            print('Asp. rat. sy/sx: ' + repr(sig_y/sig_x))
            print('Itterations: ' + repr(tt))
            #
        #
        if divmod(tt,1e5)[1]==0 and tt != 0:
            #
            if np.float(sys.argv[2]) >= 0:
                #
                np.savez('data/run_2/om_'+repr(int(sys.argv[1]))+'_c_'+repr(int(sys.argv[2]))+'_ep_'+repr(int(sys.argv[3]))+'/it_'+repr(tt)+'.npz',gnd=psi_new,x=x,y=y,g=g,Om=Om,ep=ep,en=en,vort_pos=vort_pos,en_store=en_store,am_store=am_store,ee_store=ee_store,er_store=er_store,nv_store=nv_store)
                #
            elif np.float(sys.argv[2]) < 0:
                #
                np.savez('data/run_2/om_'+repr(int(sys.argv[1]))+'_c_m'+repr(abs(int(sys.argv[2])))+'_ep_'+repr(int(sys.argv[3]))+'/it_'+repr(tt)+'.npz',gnd=psi_new,x=x,y=y,g=g,Om=Om,ep=ep,en=en,vort_pos=vort_pos,en_store=en_store,am_store=am_store,ee_store=ee_store,er_store=er_store,nv_store=nv_store)
                #
            #
            if tt == 1e5:
                #
                print('-----------------------')
                print('First 1e5 Itterations in: ' + repr(time.time()-t0) + 's')
                #
            else:
                #
                print('-----------------------')
                print('Last 1e5 Itterations in: ' + repr(time.time()-tn) + 's')
                #
            #
            tn = time.time()
            #
        #
        psi_new = psi_new/np.sqrt(dx*dy*(psi_new*np.conj(psi_new)).sum())
        psi_new = np.abs(psi_new)*psi_phi
        #
        psi = psi_new
        #
        tt += 1
        #
    #
    print('Energy: ' + repr(en))
    print('Error: ' + repr(en_er))
    print('Itterations: ' + repr(tt))
    #
    return psi, en, vort_pos, en_store, am_store, ee_store, er_store, nv_store, x, y
#
Dx = (np.float(sys.argv[4]),np.float(sys.argv[6]))
Dy = (np.float(sys.argv[5]),np.float(sys.argv[6]))
Dt = np.float(sys.argv[7])
tol = 1e-10
g = 300
#
Om = (np.float(sys.argv[1])/100.0,np.float(sys.argv[2])) #Om,C
ep = np.float(sys.argv[3])/100.0
#
kin = np.zeros((int(2*Dx[0]/Dx[1]),int(2*Dy[0]/Dy[1])),dtype='complex128')
px = np.zeros((int(2*Dx[0]/Dx[1]),int(2*Dy[0]/Dy[1])),dtype='complex128')
py = np.zeros((int(2*Dx[0]/Dx[1]),int(2*Dy[0]/Dy[1])),dtype='complex128')
#
t0 = time.time()
gnd, en, vort_pos, en_store, am_store, ee_store, er_store, nv_store, x, y = imt(Dt=Dt,Dx=Dx,Dy=Dy,tol=tol,g=g,Om=Om,ep=ep)
print(repr(time.time()-t0))
#
if np.float(sys.argv[2]) >= 0:
    #
    np.savez('data/run_2/dyn_om_'+repr(int(sys.argv[1]))+'_c_'+repr(int(sys.argv[2]))+'_ep_'+repr(int(sys.argv[3]))+'.npz', Dt=Dt, Dx=Dx, Dy=Dy, tol=tol, g=g, Om=Om, ep=ep, gnd=gnd, en=en, en_store=en_store, am_store=am_store, er_store=er_store, nv_store=nv_store, x=x, y=y)
    #
elif np.float(sys.argv[2]) < 0:
    #
    np.savez('data/run_2/dyn_om_'+repr(int(sys.argv[1]))+'_c_m'+repr(abs(int(sys.argv[2])))+'_ep_'+repr(int(sys.argv[3]))+'.npz', Dt=Dt, Dx=Dx, Dy=Dy, tol=tol, g=g, Om=Om, ep=ep, gnd=gnd, en=en, en_store=en_store, am_store=am_store, er_store=er_store, nv_store=nv_store, x=x, y=y)
    #
#
