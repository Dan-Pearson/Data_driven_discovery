from __future__ import division # causes, e.g., 1/2=0.5 and 1//2=0  (without this line of code 1/2=0)
import numpy as np
import h5py

def rsym(Qin):
    Qout = np.tril(Qin,-1)+np.tril(Qin,-1).conj().T+np.diag(np.diag(Qin))
    Qout = np.hstack((Qout,np.fliplr(Qout[:,1:-1])))
    Qout = np.vstack((Qout,np.flipud(Qout[1:-1,:])))
    return Qout

def _B21u(N,ksq,dt):
    b21u = np.zeros((int(N/2+1),int(N/2+1)))
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1,int(M+1))-0.5)/M)
    for n in range(int(N/2+1)):
        LR = np.asarray(-dt*np.tile(np.mat(ksq[n:int(N/2+1),n]).T,(1,M)) +\
                        np.tile(r,(int(N/2+1-n),1)))
        b21u[n:,n] = np.real((np.exp(LR/2)-1)/LR).sum(1)/M
    b21u = rsym(b21u)
    return b21u

def _B31u(N,ksq,dt):
    b31u = np.zeros((int(N/2+1),int(N/2+1)))
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1,int(M+1))-0.5)/M)
    for n in range(int(N/2+1)):
        LR = np.asarray(-dt*np.tile(np.mat(ksq[n:int(N/2+1),n]).T,(1,M)) +\
                        np.tile(r,(int(N/2+1-n),1)))
        b31u[n:,n] = np.real(((LR-4)*np.exp(LR/2)+LR+4)/(LR**2)).sum(1)/M
    b31u = rsym(b31u)
    return b31u

def _B32u(N,ksq,dt):
    b32u = np.zeros((int(N/2+1),int(N/2+1)))
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1,int(M+1))-0.5)/M)
    for n in range(int(N/2+1)):
        LR = np.asarray(-dt*np.tile(np.mat(ksq[n:int(N/2+1),n]).T,(1,M)) +\
                        np.tile(r,(int(N/2+1-n),1)))
        b32u[n:,n] = np.real((4*np.exp(LR/2)-2*LR-4)/(LR**2)).sum(1)/M
    b32u = rsym(b32u)
    return b32u

def _B41u(N,ksq,dt):
    b41u = np.zeros((int(N/2+1),int(N/2+1)))
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1,int(M+1))-0.5)/M)
    for n in range(int(N/2+1)):
        LR = np.asarray(-dt*np.tile(np.mat(ksq[n:int(N/2+1),n]).T,(1,M)) +\
                        np.tile(r,(int(N/2+1-n),1)))
        b41u[n:,n] = np.real(((LR-2)*np.exp(LR)+LR+2)/(LR**2)).sum(1)/M
    b41u = rsym(b41u)
    return b41u

def _B43u(N,ksq,dt):
    b43u = np.zeros((int(N/2+1),int(N/2+1)))
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1,int(M+1))-0.5)/M)
    for n in range(int(N/2+1)):
        LR = np.asarray(-dt*np.tile(np.mat(ksq[n:int(N/2+1),n]).T,(1,M)) +\
                        np.tile(r,(int(N/2+1-n),1)))
        b43u[n:,n] = np.real((2*np.exp(LR)-2*LR-2)/(LR**2)).sum(1)/M
    b43u = rsym(b43u)
    return b43u

def _C1u(N,ksq,dt):
    c1u = np.zeros((int(N/2+1),int(N/2+1)))
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1,int(M+1))-0.5)/M)
    for n in range(int(N/2+1)):
        LR = np.asarray(-dt*np.tile(np.mat(ksq[n:int(N/2+1),n]).T,(1,M)) +\
                        np.tile(r,(int(N/2+1-n),1)))
        c1u[n:,n] = np.real((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3).sum(1)/M
    c1u = rsym(c1u)
    return c1u

def _C23u(N,ksq,dt):
    c23u = np.zeros((int(N/2+1),int(N/2+1)))
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1,int(M+1))-0.5)/M)
    for n in range(int(N/2+1)):
        LR = np.asarray(-dt*np.tile(np.mat(ksq[n:int(N/2+1),n]).T,(1,M)) +\
                        np.tile(r,(int(N/2+1-n),1)))
        c23u[n:,n] = 2*(np.real((2+LR+np.exp(LR)*(-2+LR))/LR**3)).sum(1)/M
    c23u = rsym(c23u)
    return c23u

def _C4u(N,ksq,dt):
    c4u = np.zeros((int(N/2+1),int(N/2+1)))
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1,int(M+1))-0.5)/M)
    for n in range(int(N/2+1)):
        LR = np.asarray(-dt*np.tile(np.mat(ksq[n:int(N/2+1),n]).T,(1,M)) +\
                        np.tile(r,(int(N/2+1-n),1)))
        c4u[n:,n] = np.real((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3).sum(1)/M
    c4u = rsym(c4u)
    return c4u

    
#Initial Condition
def initial(N,X,Y):
    u = 1e-3*np.random.rand(N,N)
    return u

def nonlinear(uhat,dx,c):
    u = np.real(np.fft.ifft2(uhat))
    ux, uy = np.gradient(u, dx)
    return np.fft.fft2(c*(ux**2+uy**2))


#Exponential time differencing, 4th order Runge-Kutta
#Cox and Matthews
def stepper(uhat,B21u,B31u,B32u,B41u,B43u,C1u,C23u,C4u,Eu,Eu2,dt,dx,c):
    k1u = dt*nonlinear(uhat,dx,c)   # Nv
    u2hat = Eu2*uhat + B21u*k1u  # a
    k2u = dt*nonlinear(u2hat,dx,c)  # Na
    u3hat = Eu2*uhat + B31u*k1u + B32u*k2u # b
    k3u = dt*nonlinear(u3hat,dx,c)  # Nb
    u4hat = Eu*uhat + B41u*k1u + B43u*k3u # c
    k4u = dt*nonlinear(u4hat,dx,c)  # Nc
    return np.fft.fft2(np.real(np.fft.ifft2(Eu*uhat + k1u*C1u + (k2u+k3u)*C23u + k4u*C4u))) # v update, returns uhat
    

def main(N=128,Nfinal=2000,dt=0.1,ckeep=20,L=50.): 
    c = 0.5
    kx = (np.pi/L)*np.hstack((np.arange(0,N/2+1),np.arange(-N/2+1,0)))
    kxx,kyy = np.meshgrid(kx,kx)
    ksq = kxx**2 + kyy**2
    LL = ksq-ksq**2
    Eu2 = np.exp(dt*LL/2.)
    Eu = np.exp(dt*LL) 

    B21u = _B21u(N,ksq,dt)
    B31u = _B31u(N,ksq,dt)
    B32u = _B32u(N,ksq,dt)
    B41u = _B41u(N,ksq,dt)
    B43u = _B43u(N,ksq,dt)
    C1u = _C1u(N,ksq,dt)
    C23u = _C23u(N,ksq,dt)
    C4u = _C4u(N,ksq,dt)
    
    x = (2*L/N)*np.arange(-N//2,N//2)
    dx = float(np.abs(x[1]-x[0]))
    X,Y = np.meshgrid(x, x) # can be used for initial condition

    u = initial(N,X,Y) 
    del X # free up unneeded memory
    del Y #
    uhat = np.fft.fft2(u)
    Nkeep = int(1+Nfinal/ckeep)
    ukeepshape = (N,N,Nkeep)
    outName ='ddd_ks_data.hdf5'
    out = h5py.File(outName, 'w') # output data
    out.create_dataset('t',(Nkeep,),dtype='float') # create time dataset and append simulation attributes
    out['t'][...] = dt*np.arange(0,Nfinal+1,ckeep)     
    out['t'].attrs['L'] = L
    out['t'].attrs['N'] = N
    out['t'].attrs['Nfinal'] = Nfinal
    out['t'].attrs['ckeep'] = ckeep
    out.create_dataset('x',x.shape,dtype='float')
    out['x'][...] = x
    out.create_dataset('kx',kx.shape,dtype='float')
    out['kx'][...] = kx
    out.create_dataset('ukeep',ukeepshape,dtype='float')
    out['ukeep'][:,:,0] = u
    out.close()
    #ETD Runge-Kutta


    for n in range(1,Nfinal+1):
        uhat = stepper(uhat,B21u,B31u,B32u,B41u,B43u,C1u,C23u,C4u,Eu,Eu2,dt,dx,c)    
        if n % ckeep == 0:
            print(str(100*n/Nfinal)+'%')
            u = np.real(np.fft.ifft2(uhat))
            #print(np.max(np.sqrt(np.gradient(u,dx)[1]**2)))
            if np.isnan(u[0,0]):
                print('NaN encountered!')
                return 'Error - Nan'
            if np.max(np.abs(u-np.mean(u))) < 1e-6 or np.var(u) == 0.:
                print('flattened')
                return 'error - flattened' 
            out = h5py.File(outName, 'r+')
            out['ukeep'][:,:,n//ckeep] = u
            out.close()


    print('ddd_ks_gen.py output:  ' + outName)

if __name__ == '__main__':
    main()
