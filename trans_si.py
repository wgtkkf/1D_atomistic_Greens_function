# 1D Atomistic Green's Function
# Transmission
# Si system, no gap
# Coded by Takuro TOKUNAGA
# Last modified: June 06 2019

import math
import numpy as np
import cmath
import time
import sys
from scipy.integrate import quad
from numpy import linalg as LA
start = time.time()

### define variables (start) ###
catoms1 = 2           # number of atoms of left contact1 region
ddatoms = 5           # every real number
datoms = (2*ddatoms)  # number of atoms of device region
catoms2 = 2           # number of atoms of right contact2 region
totatoms = catoms1+datoms+catoms2 # number of atoms of left and right contact region
dif = np.power(10.,1) # initial value for Lopez sancho algorithm

# Unit conversion: electron volt to joule
ucev = 1.602176620898*np.power(10.,-19)
ucnano = 1.0*np.power(10.,-9)
ucangs = 1.0*np.power(10.,-10)
ucTHZ  = 1.0*np.power(10.,-12)

# parameters for AGF method: Frequency
omegamin = 1.0*np.power(10.,12)
omegamax = 1.0*np.power(10.,14)
omega = omegamin
domega = omegamin
omega_division = 10
counter = 0

# parameters: wavevector
nmax = np.power(10.,2.0) # wave vector
lc = 5.43*np.power(10.,-10.0) # [m]
kxmin = -np.pi/lc
kx = kxmin
kxmax = np.pi/lc
dkx = (kxmax-kxmin)/nmax

# AGF parameters
delta = np.power(10.,-7) # from 10E-3 to 10E-9
zeroplus = 0

# temp variables
detgdtemp = 0 # determinant of gd
detgtottemp = 0 # determinant of gtot
trans = 0 # transmissivity

# parameters for atoms
avo = 6.02214085774*np.power(10.,23) # Avogadro constant
msi = (28.085/avo)*np.power(10.,-3) # mass of a silicon atom (kg)
raddi_si = 111*np.power(10.,-12) # silicon
area_si = np.pi*np.power(raddi_si,2) # silicon

# constants
kb = 1.38064852*np.power(10.,-23) # Boltzmann constant
dh = 6.62607004*np.power(10.,-34)/(2*np.pi) # reduced Planck constant
c = 299792458 # light speed

# parameters
conv = np.power(10.,-5)

# function: begin
def begin():
    print ("begin")

# function: end
def end():
    print ("end")

# Matrices
hd=np.zeros((datoms,datoms), dtype=np.complex)
h1=np.zeros((catoms1,catoms1), dtype=np.complex)
h2=np.zeros((catoms2,catoms2), dtype=np.complex)
Imatrix=np.identity(datoms, dtype=np.complex)
sigma1=np.zeros((datoms,datoms), dtype=np.complex)
sigma2=np.zeros((datoms,datoms), dtype=np.complex)
gdtemp=np.zeros((datoms,datoms), dtype=np.complex) # Green's function of device region
gd=np.zeros((datoms,datoms), dtype=np.complex) # Green's function of device region
dl=np.zeros((datoms,datoms), dtype=np.complex) # local density of states, 1 degree of freedom
gamma1=np.zeros((datoms,datoms), dtype=np.complex) # escape rate, P.108
gamma2=np.zeros((datoms,datoms), dtype=np.complex) # escape rate, P.108
dot1=np.zeros((datoms,datoms), dtype=np.complex) # dot products
dot2=np.zeros((datoms,datoms), dtype=np.complex) # dot products
dot3=np.zeros((datoms,datoms), dtype=np.complex) # dot products

Imatrixtot=np.identity(totatoms, dtype=np.complex)
htot=np.zeros((totatoms,totatoms), dtype=np.complex) # total matrix
gtot=np.zeros((totatoms,totatoms), dtype=np.complex) # Green's function of whole region
gtottemp=np.zeros((totatoms,totatoms), dtype=np.complex)
dtot=np.zeros((totatoms,totatoms), dtype=np.complex) # Density of states

# Vectors
tau1 = np.zeros(datoms,dtype=np.complex)
tau2 = np.zeros(datoms,dtype=np.complex)

# scalars
gs1 = 0
gs2 = 0

### define variables (end) ###

# main
begin()

# file open
f1 = open('fc_iap_si.txt', 'r') # read mode (input, fixed value)
f2 = open('transmission_si.txt', 'w') # write mode
f3 = open('dispersion_si.txt', 'w') # write mode
f4 = open('local_density_of_states_si.txt', 'w') # write mode

# read force constant
for line1 in f1: # by inter atomic force, for Si bulk area
    #print(str(line1))
    iaf_si = float(line1) # iaf_si: inter-atomic force

## Matrices Initialization, tau1 & tau2 ##
tau1[0] = (-iaf_si)/np.sqrt(msi*msi)
tau2[datoms-1] = (-iaf_si)/np.sqrt(msi*msi)

# Matrices Initialization, h1
for i in range(0,catoms1):
    # sub diagonal components, upper
    if i<catoms1-1:
        h1[i][i+1] = (-iaf_si)/msi # upper nondiagonal element
        h1[i+1][i] = (-iaf_si)/msi # lower nondiagonal element

    # Diagonal components, over write
    if i==0:
        h1[i][i] = -h1[i][i+1]
    elif i>0 and i<catoms1-1:
        h1[i][i] = -(h1[i][i-1] + h1[i][i+1])
    elif i==catoms1-1:
        h1[i][i] = -(h1[i][i-1]+tau1[0])

# Matrices Initialization, hd
for i in range(0,datoms):
    # sub diagonal components
    # upper row, Pt side
    if i<ddatoms:
        hd[i][i+1] = (-iaf_si)/msi # upper nondiagonal element
        hd[i+1][i] = (-iaf_si)/msi # lower nondiagonal element

    # lower row, Si side
    if i>=(ddatoms-1) and i<datoms-1: # datoms = 2*ddatoms
        hd[i][i+1] = (-iaf_si)/msi
        hd[i+1][i] = (-iaf_si)/msi

    # overwrite for interface
    hd[int(datoms*0.5)-1][int(datoms*0.5)] = -(iaf_si)/msi
    hd[int(datoms*0.5)][int(datoms*0.5)-1] = -(iaf_si)/msi

    # Diagonal components
    if i==0: # [0][0]
        hd[i][i] = -(tau1[0] + hd[i][i+1])
    elif i>0 and i<datoms-1:
        hd[i][i] = -(hd[i][i-1] + hd[i][i+1])
    elif i==(datoms-1): # [datoms-1][datoms-1]
        hd[i][i] = -(hd[i][i-1] + tau2[datoms-1])

# Matrices Initialization, h2
for i in range(0,catoms2):
    # sub diagonal components, upper
    if i<catoms2-1:
        h2[i][i+1] = (-iaf_si)/msi # upper nondiagonal element
        h2[i+1][i] = (-iaf_si)/msi # lower nondiagonal element

    # Diagonal components, over write
    if i==0:
        h2[i][i] = -(tau2[datoms-1]+h2[i][i+1])
    elif i>0 and i<catoms2-1:
        h2[i][i] = -(h2[i][i-1] + h2[i][i+1])
    elif i==catoms2-1:
        h2[i][i] = -h2[i][i-1]

# Total matrix
zmat_2_2 = np.zeros((catoms1,catoms1), dtype=np.complex) # 2*2 size
zmat_10_1 = np.zeros((datoms,catoms1-1), dtype=np.complex) # 10*1 size
zmat_2_14 = np.zeros((catoms1,totatoms-catoms1), dtype=np.complex) # 2*12 size
#print(zmat_2_2.shape)
#print(zmat_10_1.shape)
#print(zmat_2_14.shape)

tau1_r = tau1.reshape(10,1)
tau2_r = tau2.reshape(10,1)
#print(str(tau1_r))
#print(str(tau2_r))

# 1
Htot_1_temp1 = np.concatenate([h1, zmat_2_14], axis=1)
Htot_1 = np.concatenate([h1, zmat_2_14], axis=1)

# Device
Htot_d_temp1 = np.concatenate([zmat_10_1,tau1_r], axis=1)
Htot_d_temp2 = np.concatenate([Htot_d_temp1,hd], axis=1)
Htot_d_temp3 = np.concatenate([Htot_d_temp2,tau2_r], axis=1)
Htot_d_temp4 = np.concatenate([Htot_d_temp3,zmat_10_1], axis=1)
Htot_d = Htot_d_temp4
#print(Htot_d.shape)

# 2
Htot_2 = np.concatenate([zmat_2_14, h2], axis=1)
#print(Htot_2.shape)

# total temp
Htot_temp = np.concatenate([Htot_1, Htot_d], axis=0)
Htot = np.concatenate([Htot_temp, Htot_2], axis=0)

# total final
# input rest of tau1 & tau2 to zmat
Htot[1][2] = tau1[0]
Htot[12][11] = tau2[datoms-1]

# diagonalization
l, P = np.linalg.eig(Htot)
Pinv = np.transpose(P)
#print(str(Htot))
Htot_diagonal = LA.multi_dot([Pinv,Htot,P])
max = np.amax(Htot_diagonal)
Htot_diagonal = (1/max)*Htot_diagonal
for i in range(0,totatoms):
    for j in range(0,totatoms):
        if Htot_diagonal[i][j]<np.power(10.,-3):
            Htot_diagonal[i][j] = 0.0
        else:
            Htot_diagonal[i][j] = Htot_diagonal[i][j]

        #if i==j:
        #    print(str(Htot_diagonal[i][j].real))

# Transmission
def trans(omega):
    # Calculate tranmission & conductance

    # Initialization of difference
    dif = np.power(10.,1)

    # zeroplus
    zeroplus = delta*np.power(omega,2)

    # Decimation technique for gs1
    # n = 0
    hs_gs1_0 = (np.power(omega,2)+1j*zeroplus-h1[catoms1-1][catoms1-1])   # eq (84)
    hb_gs1_0 = (np.power(omega,2)+1j*zeroplus-2*h1[catoms1-2][catoms1-2]) # eq (84)
    tau_gs1_0 = -h1[catoms1-2][catoms1-1]                                 # eq (84)

    # n = 1
    hs_gs1_1 = hs_gs1_0-np.power(tau_gs1_0,2)/hb_gs1_0
    hb_gs1_1 = hb_gs1_0-2*np.power(tau_gs1_0,2)/hb_gs1_0
    tau_gs1_1 = -np.power(tau_gs1_0,2)/hb_gs1_0

    # Initialization:n = 2, 3..
    hs_gs1_n = hs_gs1_1
    hb_gs1_n = hb_gs1_1
    tau_gs1_n = tau_gs1_1

    while(abs(dif)>conv):
        hs_gs1_n1 = hs_gs1_n # hs_n1:n-1
        hb_gs1_n1 = hb_gs1_n # hb_n1:n-1

        hs_gs1_n = hs_gs1_n-np.power(tau_gs1_n,2)/hb_gs1_n # n
        hb_gs1_n = hb_gs1_n-2*np.power(tau_gs1_n,2)/hb_gs1_n # n
        tau_gs1_n = -np.power(tau_gs1_n,2)/hb_gs1_n1 # n
        dif = (hs_gs1_n)-(hs_gs1_n1) # difference of hs_n-hs_n1

        if abs(dif)<conv:
            gs1 = 1/hs_gs1_n
            break

    # reset 'dif' of Lopez sancho algorithm for 'g2'
    dif = np.power(10.,1)

    # Decimation technique for gs2
    # n = 0
    hs_0 = (np.power(omega,2)+1j*zeroplus-h2[0][0])   # eq (84)
    hb_0 = (np.power(omega,2)+1j*zeroplus-2*h2[1][1]) # eq (84)
    tau_0 = -h2[0][1]                                 # eq (84)

    # n = 1
    hs_1 = hs_0-np.power(tau_0,2)/hb_0 # eq (88)
    hb_1 = hb_0-2*np.power(tau_0,2)/hb_0 # eq (88)
    tau_1 = -np.power(tau_0,2)/hb_0 # eq (88)

    # Initialization:n = 2, 3..
    hs_n = hs_1
    hb_n = hb_1
    tau_n = tau_1

    while(abs(dif)>conv):
        hs_n1 = hs_n # hs_n1:n-1
        hb_n1 = hb_n # hb_n1:n-1

        hs_n = hs_n-np.power(tau_n,2)/hb_n   # n, eq(91)
        hb_n = hb_n-2*np.power(tau_n,2)/hb_n # n, eq(91)
        tau_n = -np.power(tau_n,2)/hb_n1     # n, eq(91)
        dif = (hs_n)-(hs_n1) # difference of hs_n-hs_n1

        if abs(dif)<conv:
            gs2 = 1/hs_n
            break

    # sigma1[0][0], sigma2[datoms-1][datoms-1]
    sigma1[0][0] = tau1[0]*gs1*np.conjugate(tau1[0])
    sigma2[datoms-1][datoms-1] = tau2[datoms-1]*gs2*np.conjugate(tau2[datoms-1])

    for i in range(0,datoms):
        for j in range(0,datoms):
            gdtemp[i][j] = np.power(omega,2)*Imatrix[i][j]-hd[i][j]-sigma1[i][j]-sigma2[i][j]

    detgdtemp = np.linalg.det(gdtemp)
    #print(str(detgdtemp))
    if detgdtemp != 0:
        # Green's function for device region
        gd = np.linalg.inv(gdtemp)

    # local density of states
    dl = 1j*(gd-np.conjugate(gd))*omega/(np.pi*lc*0.25)

    # transmissivity calculation
    # gamma1
    gamma1 = 1j*(sigma1-np.conjugate(sigma1.T))
    # gamma2
    gamma2 = 1j*(sigma2-np.conjugate(sigma2.T))
    # dot products
    dot1 = np.dot(gamma1,gd)
    dot2 = np.dot(dot1,gamma2)
    dot3 = np.dot(dot2,np.conjugate(gd.T))
    trans = np.trace(dot3) # transmissivity

    return trans, dl

# Frequency loop
while omega <= omegamax:
    # Transmission calculation
    transmission = trans(omega)[0]

    # Local Density of States calculation
    ldos = trans(omega)[1]
    row,col = ldos.shape

    # output transmission
    f2.write(str(omega)) # change of variable
    f2.write(str(' '))
    f2.write(str(transmission.real))
    f2.write(str(' '))
    f2.write(str(transmission.imag))
    f2.write('\n')

    f4.write(str(omega)) # change of variable
    f4.write(str(' '))
    f4.write(str(ldos[0][0].real))
    f4.write('\n')

    # Frequency update
    omega = omega + domega

    counter += 1
    if counter == omega_division-1:
        domega = domega*omega_division # omega_division: 10
        counter = 0
    elif omega >= np.power(10.,13):
        domega = 0.1*np.power(10.,12)

# phonon dispersion relation
kvector = np.zeros(3,dtype=np.complex)
for i in range(0, 3): # 0~2
    kvector[i] = Htot[1][i]

#print(str(kvector))
# normalization

# wave vector loop
while kx < kxmax:

    # analytical
    #analytical = kvector[1]*(1-np.cos(kx*lc))
    analytical = np.sqrt((2*kvector[1])*np.power(np.sin(abs(kx*lc*0.5)),2.0))  # [1/s]
    #analytical = np.sqrt(analytical)/(2*np.sqrt(0.5*kvector[1])) # Normalized

    dynamical = 0
    for i in range(0, 3): # 0~2
        dynamical = dynamical + kvector[0]*np.exp(-1j*kx*(-1)*lc) +\
        + kvector[1]*np.exp(-1j*kx*0*lc) + kvector[2]*np.exp(-1j*kx*(1)*lc)

    #print(str(dynamical))
    dynamical = np.sqrt(dynamical) # [1/s]

    # Output result
    f3.write(str(kx/(np.pi/lc)*10))
    f3.write(str(' '))
    f3.write(str(analytical.real*ucTHZ))
    f3.write(str(' '))
    f3.write(str(dynamical.real*ucTHZ))
    f3.write('\n')

    # Wave vector loop
    kx = kx + dkx

    #print(str(kx))

# file close
f1.close()
f2.close()
f3.close()
f4.close()

end()

# time display
elapsed_time = time.time()-start
print("elapsed_time:{:.2f}".format(elapsed_time) + "[sec]")
