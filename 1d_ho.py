from scipy.fftpack import fft
import matplotlib.pyplot as plt
import numpy as np
#
def get_en(psi,U,dx,g):
	#
	ke = 0
	te = 0
	ie = 0
	en = 0
	#
	for x in xrange(1,len(psi)-1):
		ke = ke + dx*0.5*np.abs((psi[x+1]-psi[x-1])/(2.0*dx))**2
	#
	te = dx*np.trapz(U*np.abs(psi)**2)
	ie = 0.5*g*dx*np.trapz(np.abs(psi)**4)
	en = ke + te + ie
	en = en/(dx*np.trapz(np.abs(psi)**2))
	#
	return en
#
def ss_gnd(Dx,Dt,tol,g):
	#
	NN = int(1 + 2.0 * Dx[0]/Dx[1])
	xx = np.linspace(-Dx[0],Dx[0],NN)
	Nk = 0.5*(len(xx)-1)
	dk = np.pi/(Dx[1]*Nk)
	kk = np.linspace(-Nk,Nk,NN)*dk
	#
	psi = np.random.rand(NN)
	psi = psi/np.sqrt(Dx[1]*np.trapz(np.abs(psi)**2))
	psi0 = psi
	U = 0.5*xx**2
	mu = np.random.rand(1)
	mu_old = np.random.rand(1)
	mu_e = 1
	ii = 1
	mu_store = []
	ist = psi.reshape(psi.shape[0],1)
	#
	while mu_e > tol:
		#
		psi = psi*np.exp(-0.5*Dt*(U + g*np.abs(psi)**2))
		#
		p_k = np.fft.fftshift(fft(psi))/NN
		p_k = p_k*np.exp(-Dt*0.5*kk**2)
		psi = np.fft.ifft(np.fft.ifftshift(p_k))*NN;
		#
		psi = psi*np.exp(-0.5*Dt*(U + g*np.abs(psi)**2))
		#
		mu = get_en(psi=psi,U=U,dx=Dx[1],g=g)
		mu_e = np.abs((mu-mu_old)/mu)
		#
		if divmod(ii,2.5e2)[1]==0:
			#
			print('Energy diff: ' + repr(mu_e))
			print('Ground state energy ' +repr(mu))
			mu_store = np.append(mu_store,mu)
			ist = np.append(ist,psi.reshape(psi.shape[0],1),axis=1)
			pass
			#
		#
		psi = psi/np.sqrt(Dx[1]*(psi*np.conj(psi)).sum())
		#
		mu_old = mu
		ii += 1
	#
	print('Itterations: ' + repr(ii-1))
	print('Ground state energy: ' + repr(mu))
	return psi, mu, U, g, xx, mu_store, ist
#
def ss_rtm(psi,Dx,Dt,T,g,x0):
	#
	NN = int(1 + 2.0 * Dx[0]/Dx[1])
	xx = np.linspace(-Dx[0],Dx[0],NN)
	Nk = 0.5*(len(xx)-1)
	dk = np.pi/(Dx[1]*Nk)
	kk = np.linspace(-Nk,Nk,NN)*dk
	#
	U = 0.5 * (xx-x0)**2
	NT = T/Dt
	samp = 400
	sptm = np.zeros((samp,NN),dtype=complex)
	ts = np.linspace(0,T,samp)
	jj=1
	c=0
	#
	if T != 0:
		#
		for jj in xrange(1,int(NT)):
			#
			psi = psi*np.exp(-0.5*1j*Dt*(U + g*np.abs(psi)**2))
			#
			p_k = np.fft.fftshift(fft(psi))/NN
			p_k = p_k*np.exp(-Dt*1j*0.5*kk**2)
			psi = np.fft.ifft(np.fft.ifftshift(p_k))*NN;
			#
			psi = psi*np.exp(-0.5*1j*Dt*(U + g*np.abs(psi)**2))
			#
			if divmod(jj,np.floor(NT/samp))[1]==0:
				#
				c += 1
				sptm[c][:] = psi
				#
			#
			if divmod(jj,np.floor(NT/20))[1]==0:
				#
				print('Real time Progress: ' + repr(np.round(100*jj/NT)) + '%')
				#
			#
		#
	#
	return psi, sptm, ts
#
Dx = (10,0.05)
Dt = (1e-3,1e-3)
tol = 1e-10
g = 0
x0 = 2
T = 20
#
gnd, mu, U, g, xx, mu_store, ist = ss_gnd(Dx=Dx, Dt=Dt[0], tol=tol, g=g)
psi, sptm, ts = ss_rtm(psi=gnd, Dx=Dx, Dt=Dt[1], T=T, g=g, x0=x0)
#
f1=plt.figure(facecolor='white')
plt.plot(xx,np.abs(gnd)**2,label=r'Numerical $|\psi|^2$')
plt.plot(xx,(1/np.sqrt(np.pi))*np.exp(-xx**2),'--',label=r'Analytical $|\psi|^2$')
axes = plt.gca()
axes.set_xlim([-Dx[0],Dx[0]])
plt.suptitle('', fontsize=20)
plt.xlabel(r"$x$", fontsize=14)
plt.ylabel(r"", fontsize=14)
plt.legend()
f1.show()
#
f2 = plt.figure()
plt.pcolor(np.linspace(0,250*Dt[0],ist.shape[1]),xx,np.abs(ist)**2,cmap='jet')
plt.xlabel(r'$it$')
plt.ylabel(r'$x/a_x$')
plt.title(r'$|\psi(x,it)|^2$ vs. $x$')
f2.show()
#
if T != 0:
	#
	f3=plt.figure(facecolor='white');
	plt.pcolor(ts,xx,np.abs(np.transpose(sptm))**2, cmap='jet');
	#
	axes = plt.gca();
	axes.set_ylim([-Dx[0],Dx[0]]);
	axes.set_xlim([0,T]);
	plt.suptitle('', fontsize=20)
	plt.xlabel(r"$t$", fontsize=14)
	plt.ylabel(r"$x$", fontsize=14)
	f3.show()
	#
#
