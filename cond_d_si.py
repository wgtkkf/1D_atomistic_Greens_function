# 1D Atomistic Green's Function
# Heat transfer coefficient for Si system (Symmetric system)
# Decimation technique is used
# Temperature fixed (single temperature) & gap distance loop
# Coded by Takuro TOKUNAGA
# Last modified: November 14 2018
# Correctly running, be careful for the unit of Casimir force constant if HTC is strange

import math
import numpy as np
import cmath
import time
import sys
start = time.time()

sys.path.append('../fc/')
from lifshitz_f import lifshitz_fc # import function
from sw_f import sw_fc             # import function
from tersoff_f import tersoff_fc   # import function
from lj_f import lj_si_fc   # import function

### define variables (start) ###
catoms1 = 2           # number of atoms of left contact1 region
ddatoms = 5           # every real number
datoms = (2*ddatoms)  # number of atoms of device region
catoms2 = 2           # number of atoms of right contact2 region
totatoms = catoms1+datoms+catoms2 # number of atoms of left and right contact region
dif = np.power(10.,1.0) # initial value for Lopez sancho algorithm

# Unit conversion:
ucev = 1.602176620898*np.power(10.,-19.0) # eV to J
ucnano = 1.0*np.power(10.,-9.0) # nano to m
ucangs = 1.0*np.power(10.,-10.0) # angs to m
ucpico = 1.0*np.power(10.,-12.0) # pico to m

# parameters for AGF method: Frequency
omegazero = 1.0*np.power(10.,-8.0)
omegamax = 6.0*np.power(10.,13.0) # 6.0*np.power(10.,13), this is not used, good!
omega = omegazero
delta = np.power(10.,-7.0) # from 10E-3 to 10E-9
zeroplus = 0

# temporary parameters
detgdtemp = 0 # determinant of gd
detgtottemp = 0 # determinant of gtot
trans = 0 # transmissivity

# parameters for atoms
avo = 6.02214085774*np.power(10.,23.0) # Avogadro constant
msi = (28.085/avo)*np.power(10.,-3.0) # mass of a silicon atom (kg)
raddi_si = 111*np.power(10.,-12.0) # silicon
area_si = np.pi*np.power(raddi_si,2.0) # silicon

# constants
kb = 1.38064852*np.power(10.,-23.0) # Boltzmann constant
rh = 6.62607004*np.power(10.,-34.0)/(2*np.pi) # reduced Planck constant
c = 299792458 # light speed, [m/s]
temperature = 300 # temperature, [K]

# parameters
conv = np.power(10.,-5.0)

# parameters for thermal conductance
temperature1 = 301 # [K], high
temperature2 = 300 # [K], low

# gap related
nmax = np.power(10.,2.0) # number of data points
gapmin = 0.01*ucnano # [m]
gapmax = 100*ucnano # [m]
gap = gapmin # [m]
dgap = gapmin # [m], initial value

# parameters for function
cutoff = 708

# de method integral parameters
sn = 1 # n
sh = np.power(10.,-3.0)
conv_de = np.power(10.,-4.0) # convergence criteria
difde = 1

# function: begin
def begin():
    print ("begin")

# function: end
def end():
    print ("end")

def phi(sn):
    y = np.exp(0.5*np.pi*np.sinh(sn*sh))
    return y

def dphi(sn):
    y = 0.5*np.pi*np.cosh(sn*sh)*np.exp(0.5*np.pi*np.sinh(sn*sh))
    return y

def be(somega, stemperature): # small letter omega & temperature

    # define integrand
    index = rh*somega/(kb*stemperature)

    if index > cutoff:
        fbe = 0
    elif index < cutoff:
        denominator = np.exp(index)-1
        if denominator==0:
            fbe = 0
        else:
            fbe = 1/denominator

    return fbe

# Matrices
hd=np.zeros((datoms,datoms), dtype=np.complex)
h1=np.zeros((catoms1,catoms1), dtype=np.complex)
h2=np.zeros((catoms2,catoms2), dtype=np.complex)
Imatrix=np.identity(datoms, dtype=np.complex)
sigma1=np.zeros((datoms,datoms), dtype=np.complex)
sigma2=np.zeros((datoms,datoms), dtype=np.complex)
gdtemp=np.zeros((datoms,datoms), dtype=np.complex) # Green's function of device region
gd=np.zeros((datoms,datoms), dtype=np.complex) # Green's function of device region
dl=np.zeros((datoms,datoms), dtype=np.complex) # local density of states
gamma1=np.zeros((datoms,datoms), dtype=np.complex) # escape rate, P.108
gamma2=np.zeros((datoms,datoms), dtype=np.complex) # escape rate, P.108
dot1=np.zeros((datoms,datoms), dtype=np.complex) # dot products
dot2=np.zeros((datoms,datoms), dtype=np.complex) # dot products
dot3=np.zeros((datoms,datoms), dtype=np.complex) # dot products

Imatrixtot=np.identity(totatoms, dtype=np.complex)
htot=np.zeros((totatoms,totatoms), dtype=np.complex)
gtot=np.zeros((totatoms,totatoms), dtype=np.complex) # Green's function of whole region
gtottemp=np.zeros((totatoms,totatoms), dtype=np.complex)
dtot=np.zeros((totatoms,totatoms), dtype=np.complex) # Density of states

# vectors
tau1 = np.zeros(datoms,dtype=np.complex)
tau2 = np.zeros(datoms,dtype=np.complex)

# scalars
gs1 = 0
gs2 = 0

### define variables (end) ###

# main
begin()

# file open
f = open('htc.txt', 'w') # write mode (outputs)
f1 = open('fc_iap_si.txt', 'r') # read mode (input, fixed value)

# read force constant
for line1 in f1: # by inter atomic force, for Si bulk area
    #print(str(line1))
    iaf_si = float(line1) # iaf_si: inter-atomic force

# distance loop
while gap < gapmax:

    # force constants update
    cpf = lifshitz_fc(gap) # cpf: Casimir-Polder force
    iff = tersoff_fc(gap) # interfacial force by si & si
    #iff = sw_fc(gap) # interfacial force by si & si
    #iff = lj_si_fc(gap) # interfacial force by si & si

    # for contact calculation
    cpf = 0
    iff = iaf_si

    ## Matrices Initialization, tau1 & tau2 ##
    tau1[0] = (-iaf_si)/np.sqrt(msi*msi)
    tau2[datoms-1] = (-iaf_si)/np.sqrt(msi*msi)

    # Matrices Initialization, h1

    # Matrices Initialization, hd
    for i in range(0,datoms):
        # sub diagonal components
        if i<datoms-1: # datoms = 2*ddatoms
            hd[i][i+1] = (-iaf_si)/msi # upper nondiagonal element
            hd[i+1][i] = (-iaf_si)/msi # lower nondiagonal element

        # overwrite for interface
        hd[int(datoms*0.5)-1][int(datoms*0.5)] = -(iff+cpf)/msi
        hd[int(datoms*0.5)][int(datoms*0.5)-1] = -(iff+cpf)/msi

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
        if i<catoms2-1: # even i
            h2[i][i+1] = (-iaf_si)/msi # upper nondiagonal element
            h2[i+1][i] = (-iaf_si)/msi # lower nondiagonal element

        # Diagonal components, over write
        if i==0:
            h2[i][i] = -(tau2[datoms-1]+h2[i][i+1])
        elif i>0 and i<catoms2-1:
            h2[i][i] = -(h2[i][i-1] + h2[i][i+1])
        elif i==catoms2-1:
            h2[i][i] = -h2[i][i-1]

    # main loop start
    def cond(omega): # Frequency loop
        # Calculate tranmission & conductance

        # Initialization of difference
        dif = np.power(10.,1.0)

        # zeroplus
        zeroplus = delta*np.power(omega,2.0)

        # Decimation technique for gs2
        # n = 0
        hs_0 = (np.power(omega,2.0)+1j*zeroplus-h2[0][0])
        hb_0 = (np.power(omega,2.0)+1j*zeroplus-2*h2[1][1])
        tau_si0 = -h2[0][1]
        tau_c0 = -h2[1][0]

        # n = 1
        hs_1 = hs_0-(tau_si0*tau_c0)/hb_0
        hb_1 = hb_0-2*(tau_si0*tau_c0)/hb_0
        tau_1 = -(tau_si0*tau_c0)/hb_0

        # Initialization:n = 2, 3..
        hs_n = hs_1
        hb_n = hb_1
        tau_n = tau_1

        while(abs(dif)>conv):
            hs_n1 = hs_n # hs_n1:n-1
            hb_n1 = hb_n # hb_n1:n-1

            hs_n = hs_n-np.power(tau_n,2.0)/hb_n # n
            hb_n = hb_n-2*np.power(tau_n,2.0)/hb_n # n
            tau_n = -np.power(tau_n,2.0)/hb_n1 # n
            dif = hs_n-hs_n1 # difference of hs_n-hs_n1

            if abs(dif)<conv:
                gs2 = 1/hs_n
                break

        # gs1 takes same values with gs2
        gs1 = gs2

        # sigma1[0][0], sigma2[datoms-1][datoms-1]
        sigma1[0][0] = tau1[0]*gs1*np.conjugate(tau1[0])
        sigma2[datoms-1][datoms-1] = tau2[datoms-1]*gs2*np.conjugate(tau2[datoms-1])

        for i in range(0,datoms):
            for j in range(0,datoms):
                gdtemp[i][j] = np.power(omega,2.0)*Imatrix[i][j]-hd[i][j]-sigma1[i][j]-sigma2[i][j]

        detgdtemp = np.linalg.det(gdtemp)

        if detgdtemp != 0:
            # Green's function for device region
            gd = np.linalg.inv(gdtemp)

        # transmittance calculation
        # gamma1
        gamma1 = 1j*(sigma1-np.conjugate(sigma1.T))
        # gamma2
        gamma2 = 1j*(sigma2-np.conjugate(sigma2.T))
        # dot products
        dot1 = np.dot(gamma1,gd)
        dot2 = np.dot(dot1,gamma2)
        dot3 = np.dot(dot2,np.conjugate(gd.T))
        trans = np.trace(dot3) # transmissivity

        # define integrand for flux
        behigh = be(omega, temperature1) # omega & temperature, left leads
        below = be(omega, temperature2) # omega & temperature, right leads

        y = trans*np.power(omega,1.0)*(behigh-below)

        return y

    # Initialization of new & old
    new = cond(phi(omegazero))*dphi(omegazero)
    old = 0

    # Thermal conductance calculation
    c1 = rh/(2*np.pi) # coefficient, in case of flux [W/m2K]
    #c1 = np.power(rh,2.0)/(2*np.pi*kb*np.power(temperature,2.0)) # coefficient, in case of conductance [W/K]

    while difde>conv_de: # integral calculation
        new = new + (cond(phi(-sn))*dphi(-sn)+cond(phi(sn))*dphi(sn))
        # conv check
        difde = abs(new-old)
        sn = sn+1
        old = new

        if difde < conv_de:
            break

    # conductance or heat transfer coefficient, do not use name 'cond = ..'
    cd = c1*(sh*new)/(temperature1-temperature2) # [W/K], conductance
    #cd = c1*(sh*new)/area_si # [W/m2K], heat transfer coefficient

    # results
    f.write(str(gap/ucnano))
    f.write(str(' '))
    f.write(str(cd.real))
    f.write('\n')

    # display gap value
    print("gap[nm]:{:.2f}".format(gap/ucnano))

    # matrix hd check
    #for i in range(0,datoms):
    #    for j in range(0,datoms):
    #        print(str(hd[j][i].real))

    # Reset parameters
    new = cond(phi(omegazero))*dphi(omegazero)
    old = 0
    sn = 1
    difde = 1

    # dgap update
    if gap < 0.1*ucnano:
        dgap = 0.01*ucnano
    elif gap>=0.1*ucnano and gap < 1*ucnano:
        dgap = 0.1*ucnano
    elif gap>=1*ucnano and gap < 10*ucnano:
        dgap = 1*ucnano
    elif gap>=10*ucnano:
        dgap = 10*ucnano

    # gap update
    gap = gap + dgap

# file close
f.close()
f1.close()

end()

# time display
elapsed_time = time.time()-start
print("elapsed_time:{:.2f}".format(elapsed_time) + "[sec]")
