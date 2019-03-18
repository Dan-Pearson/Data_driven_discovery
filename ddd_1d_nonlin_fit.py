from __future__ import division # causes, e.g., 1/2=0.5 and 1//2=0  (without this line of code 1/2=0)
import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
 

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
    
def nonlinear(uhat,kxx,dx,r):
    ux = np.real(np.fft.ifft(1j*kxx*uhat,axis=1))
    return np.fft.fft(r*ux**4,axis=1)

#Exponential time differencing, 4th order Runge-Kutta
#Cox and Matthews
def stepper(uhat,B21u,B31u,B32u,B41u,B43u,C1u,C23u,C4u,Eu,Eu2,dt,dx,kxx,r):
    k1u = dt*nonlinear(uhat,kxx,dx,r)   # Nv
    u2hat = Eu2*uhat + B21u*k1u  # a
    k2u = dt*nonlinear(u2hat,kxx,dx,r)  # Na
    u3hat = Eu2*uhat + B31u*k1u + B32u*k2u # b
    k3u = dt*nonlinear(u3hat,kxx,dx,r)  # Nb
    u4hat = Eu*uhat + B41u*k1u + B43u*k3u # c
    k4u = dt*nonlinear(u4hat,kxx,dx,r)  # Nc
    return np.fft.fft(np.real(np.fft.ifft(Eu*uhat + k1u*C1u + (k2u+k3u)*C23u + k4u*C4u,axis=1)),axis=1) # v update, returns uhat

def gen_pop(best_fits, mutations, mut, imm, N, best_rs, r_muts):
    keep = len(best_fits[:,0])
    angles = np.zeros(shape=(N,N))
    count = 0
    r_out = np.zeros_like(r_muts)
    r_muts -= np.mean(r_muts)
    for i in range(keep):
        for j in range(i+1,keep):
            genes = np.random.randint(2, size=(N,))
            angles[count,:] = genes*best_fits[i,:] + (1-genes)*best_fits[j,:]
            r_gene = np.random.randint(2)
            r_out[count] = r_gene*best_rs[i] + (1-r_gene)*best_rs[j]
            count += 1
    angles = (angles + mut*mutations)/(1 + mut)
    r_out = (r_out + mut*r_muts)/(1 + mut)
    n_imm = N-count
    immigrants = 2*np.pi*np.random.rand(n_imm,N)-np.pi
    r_immigrants = imm*(np.random.rand(N-count))# + best_rs[0]
#    n_copies = n_imm//keep
#    n_remain = n_imm - n_copies*keep
#    if n_copies > 0:
#        r_out[count:count+n_copies*keep,0] = np.tile(best_rs, n_copies)
#    if n_remain > 0:
#        r_out[count+n_copies*keep:,0] = best_rs[:n_remain]
    r_out[count:,0] = r_immigrants #best_rs[0]#
    angles[count:,:] = immigrants#(best_fits[0,:] + imm*immigrants)/(1 + imm)
    angles[-1,:] = best_fits[0,:]
    r_out[-1,0] = (best_rs[0] + mut*np.random.rand())/(1 + mut)
    angles[-2,:] = (best_fits[0,:] + mut*(2*np.pi*np.random.rand(N)-np.pi))/(1 + mut)
    r_out[-2,0] = best_rs[0]
    angles[:,N//2+1:] = np.flip(-angles[:,1:N//2],axis=1)
    return angles, r_out
    
def main(N=128,Nfinal=2000,dt=0.1,ckeep=20,L=50.,num_of_datasets=600):
    i = 1
    #j = 10
    inputFileName = 'ddd_ux4_data_'+str(i)+'.hdf5'
    inputFile = h5py.File(inputFileName, 'r')
    ukeep = inputFile['ukeep'][...]
    r_list = inputFile['r'][...]
    t = inputFile['t'][...]
    x = inputFile['x'][...]
    kx = inputFile['kx'][...]
    nmax = len(t)-1
    inputFile.close()
    A = np.fft.fft(ukeep[0,:,nmax-1])#, axis=1)
    B = np.fft.fft(ukeep[0,:,nmax])#, axis=1)
    AA = np.abs(A)
    BB = np.abs(B)

    dx = float(np.abs(x[1]-x[0]))
    kx = (np.pi/L)*np.hstack((np.arange(0,N/2+1),np.arange(-N/2+1,0)))
    kxx,kyy = np.meshgrid(kx,kx)
    ksq = kxx**2 + kyy**2
    LL = kxx**2-kxx**4
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

    keep = 15
    best_fits = np.zeros(shape=(keep,N))
    best_sums = np.zeros(shape=(keep,))
    best_sums = best_sums + 1e12
    best_rs = np.zeros(shape=(keep,))
    final_fit = np.zeros_like(A)
    new_pop = np.zeros(shape=(N,N))
    #r = 0.1
    mut = 0.1 # mutation weighting
    imm = 1 # immigration weighting
    m_max = 200000
    best_fits_sums = np.zeros(shape=(m_max,))
    best_fits_rs = np.zeros(shape=(m_max,))
    best_fits_rs_avg = np.zeros(shape=(m_max,))
    for m in range(m_max):
        uhat = np.repeat(AA[np.newaxis,...],N,axis=0)
        angles = 2*np.pi*np.random.rand(N,N)-np.pi        
        r = np.reshape(np.random.rand(N),(N,1))
        if m == 0:
            angles[:,N//2+1:] = np.flip(-angles[:,1:N//2],axis=1)
            uhat = uhat*np.exp(1j*angles)
        else: #(baby_angles + mut*(best_fit_sum/first_best_fit_sum)*angles)/(1 + mut*best_fit_sum/first_best_fit_sum)
            angles, r = gen_pop(best_fits, angles, mut, imm, N, best_rs, r)
            uhat = uhat*np.exp(1j*angles)
        uhat[:,0] = AA[0]
        for n in range(1,ckeep+1):
            uhat = stepper(uhat,B21u,B31u,B32u,B41u,B43u,C1u,C23u,C4u,Eu,Eu2,dt,dx,kxx,r)    
        uhat_abs = np.abs(uhat)
        fitness = np.sum(np.abs(uhat_abs-BB),axis=1)
        temp_args = np.argsort(fitness)[:keep]
        temp_best_sums = np.append(best_sums, fitness[temp_args])
        temp_best_rs = np.append(best_rs, r[temp_args])
        temp_best_fits = np.append(best_fits, angles[temp_args,:], axis=0)
        best_sum_indices = np.argsort(temp_best_sums)[:keep]
        best_sums = temp_best_sums[best_sum_indices]
        best_rs = temp_best_rs[best_sum_indices]
        best_fits = temp_best_fits[best_sum_indices]
        best_fits_sums[m] = best_sums[0]
        best_fits_rs_avg[m] = np.mean(best_rs)#best_rs[0]
        best_fits_rs[m] = best_rs[0]
        
    print(best_fits_sums[0])
    print(best_fits_sums[m])
    print(best_fits_sums[0]-best_fits_sums[m])
    print(best_fits_rs_avg[m])
    print(best_fits_rs[m])
#    Breal = np.real(np.fft.ifft(B))
#    Bpred = np.real(np.fft.ifft(best_fit))
    plt.plot(best_fits_sums)
#    plt.plot(x,Breal)
#    plt.plot(x,Bpred)
    plt.show()
    plt.clf()
    plt.plot(best_fits_rs_avg)
    plt.show()
    plt.clf()
    plt.plot(best_fits_rs)
    plt.show()
    print('ddd_preprocess_v6.py finished')#output:  ')# + outName)

if __name__ == '__main__':
    main()
