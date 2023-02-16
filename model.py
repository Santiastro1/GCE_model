import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma
import scipy as sc
from scipy import integrate
import builtins

#### Parameters defining the SFH shape (constant, single gaussian, two gaussian)

mean=1. # position of the first gaussian's peak (Gyr lookback time)
sig=0.2 # width of the first gaussian (dispersion in Gyr)
mean1=5.4#5. # position of the second gaussian's peak (Gyr lookback time)
sig1=0.15 # width of the second gaussian (dispersion in Gyr)
SFR_base_ini=6 # Initial SFR in Msun/yr, at timemin, Gyr in lookback time
FirstPeak=55. # Max SFR in the first gaussian (first star formation burst)
Relative_peak2peak=FirstPeak/80. # Relative intensity of the second star formation burst with respect to the first

#### Parameters defining the stellar mass range (for IMF integration) 

mass=np.arange(0.005,100,0.01) #lower mass, higher mass and mass step

#### Parameters related with the integration times

bint=0.1 # time steps
timemin=0. # initial time (direct time, not lookback)
timemax=7. # final time (direct time, not lookback)
time=np.arange(timemin, timemax, bint)

#### Parameters related with lithium depletion
Omega_ini=10 #frequency/initial stellar rotation in units of Omega_sun
td=6 #protostellar disk lifetime in Myr (9, 6 or 3Myr)
Li_deplet=[0.0]*len(time)

#### White Dwarfs in the MW at present
WD_MW_t0=1.8e6
#### Nova explosions in the MW at present
Novae_Explosion_MW_t0=50.

#######
# SFH (constant, single gaussian, two gaussian)
#######

def SFR1(t,mean1,sig1):
        return  np.exp(-(t-mean1)**2/sig1)
def SFR(t,mean,sig):
    return np.exp(-(t-mean)**2/sig)

# SFR Msun/yr:
def SFR_base(t,SFR_base_ini):
    return SFR_base_ini-0.7*t
def SFR_total(t,SFR,SFR1,SFR_base):
    normalization_sfr=np.cumsum(SFR(t,mean,sig))
    norm_sfr=normalization_sfr[len(normalization_sfr)-1]
    normalization_sfr1=np.cumsum(SFR1(t,mean1,sig1))
    norm_sfr1=normalization_sfr1[len(normalization_sfr1)-1]
    return SFR_base(t,SFR_base_ini)+FirstPeak*np.array(SFR(t,mean,sig)/norm_sfr)+FirstPeak/Relative_peak2peak*SFR1(t,mean1,sig1)/norm_sfr1
def SFR_total_single_time(t,time,SFR,SFR1,SFR_base):
    normalization_sfr=np.cumsum(SFR(time,mean,sig))
    norm_sfr=normalization_sfr[len(normalization_sfr)-1]
    normalization_sfr1=np.cumsum(SFR1(time,mean1,sig1))
    norm_sfr1=normalization_sfr1[len(normalization_sfr1)-1]    
    return SFR_base(t,SFR_base_ini)+FirstPeak*np.array(SFR(t,mean,sig)/norm_sfr)+FirstPeak/Relative_peak2peak*SFR1(t,mean1,sig1)/norm_sfr1

####
#### Initial metallicity of inflowing gas (Ni/NH) (prepared to be time dependent)
####

def Zli(t,Li_primordial):
    return 10**(Li_primordial-12)

def ZO(t,M_primordial):
    return 0.0161*10**(M_primordial)

def ZFe(t,M_primordial):
    return 0.00178*10**(M_primordial)

####################
    ## IMF subroutines
####################

def slope_imf(x,p1,p2,p3,kn1,kn2):

#Is calculating a three slope IMF
#INPUT:
#  x = An array of masses for which the IMF should be calculated
#  p1..p3 = the slopes of the power law
# kn1, kn2 = Where the breaks of the power law are
#OUTPUT:
#   An array of frequencies matching the mass base array x
    if(x > kn2):
        t = (pow(kn2,p2)/pow(kn2,p3))*pow(x,p3+1)
    elif (x < kn1):
        t = (pow(kn1,p2)/pow(kn1,p1))*pow(x,p1+1)
    else:
        t = pow(x,p2+1)
    return t


def lifetime(m,Z):

#here we will calculate the MS lifetime of the star after Argast et al., 2000, A&A, 356, 873
#INPUT:¡
#   m = mass in Msun
#   Z = metallicity in Zsun

#OUTPUT:
#   returns the lifetime of the star in Gyrs

    lm = np.log10(m)
    a0 =  3.79 + 0.24*Z
    a1 = -3.10 - 0.35*Z
    a2 =  0.74 + 0.11*Z
    tmp = a0 + a1*lm + a2*lm*lm
    return np.divide(np.power(10,tmp),1000)


class IMF(object):

#This class represents the IMF normed to 1 in units of M_sun.
#Input for initialisation:

#   mmin = minimal mass of the IMF

#   mmax = maximal mass of the IMF

#   intervals = how many steps inbetween mmin and mmax should be given

#Then one of the IMF functions can be used

#   self.x = mass base

#   self.dn = the number of stars at x

#   self.dm = the masses for each mass interval x

    def __init__(self, mmin = 0.08 , mmax = 100., intervals = 5000):
        self.mmin = mmin
        self.mmax = mmax
        self.intervals = intervals
        self.x = np.linspace(mmin,mmax,intervals)
        self.dx = self.x[1]-self.x[0]

    def normed_3slope(self,paramet = (-1.3,-2.2,-2.7,0.5,1.0)):

#        Three slope IMF, Kroupa 1993 as a default

        s1,s2,s3,k1,k2 = paramet
        u = np.zeros_like(self.x)
        v = np.zeros_like(self.x)
        for i in range(len(self.x)):
            u[i] = slope_imf(self.x[i],s1,s2,s3,k1,k2)
        v = np.divide(u,self.x)
        self.dm = np.divide(u,sum(u))
        self.dn = np.divide(self.dm,self.x)
        return(self.dm,self.dn)


    def Chabrier_1(self, paramet = (0.69, 0.079, -2.3)):

#        Chabrier IMF from Chabrier 2003 equation 17 field IMF with variable high mass slope and automatic normalisation

        sigma, m_c, expo = paramet
        dn = np.zeros_like(self.x)
        for i in range(len(self.x)):
            if self.x[i] <= 1:
                index_with_mass_1 = i
                dn[i] = (1. / float(self.x[i])) * np.exp(-1*(((np.log10(self.x[i] / m_c))**2)/(2*sigma**2)))
            else:
                dn[i] = (pow(self.x[i],expo))
        # Need to 'attach' the upper to the lower branch
        derivative_at_1 = dn[index_with_mass_1] - dn[index_with_mass_1 - 1]
        target_y_for_m_plus_1 = dn[index_with_mass_1] + derivative_at_1
        rescale = target_y_for_m_plus_1 / dn[index_with_mass_1 + 1]
        dn[np.where(self.x>1.)] *= rescale
        # Normalizing to 1 in mass space
        self.dn = np.divide(dn,sum(dn))
        dm = dn*self.x
        self.dm = np.divide(dm,sum(dm))
        self.dn = np.divide(self.dm,self.x)
        return(self.dm,self.dn)


    def Chabrier_2(self,paramet = (22.8978, 716.4, 0.25,-2.3)):

#        Chabrier IMF from Chabrier 2001, IMF 3 = equation 8 parameters from table 1


        A,B,sigma,expo = paramet
        expo -= 1. ## in order to get an alpha index normalisation
        dn = np.zeros_like(self.x)
        for i in range(len(self.x)):
            dn[i] = A*(np.exp(-pow((B/self.x[i]),sigma)))*pow(self.x[i],expo)
        self.dn = np.divide(dn,sum(dn))
        dm = dn*self.x
        self.dm = np.divide(dm,sum(dm))
        self.dn = np.divide(self.dm,self.x)
        return(self.dm,self.dn)


    def salpeter(self, alpha = (2.35)):

#        Salpeter IMF

#        Input the slope of the IMF

        self.alpha = alpha
        temp = np.power(self.x,-self.alpha)
        norm = sum(temp)
        self.dn = np.divide(temp,norm)
        u = self.dn*self.x
        self.dm = np.divide(u,sum(u))
        self.dn = np.divide(self.dm,self.x)
        return (self.dm,self.dn)


    def BrokenPowerLaw(self, paramet):
        breaks,slopes = paramet
        if len(breaks) != len(slopes)-1:
            print("error in the precription of the power law. It needs one more slope than break value")
        else:
            dn = np.zeros_like(self.x)
            self.breaks = breaks
            self.slopes = slopes
            self.mass_range = np.hstack((self.mmin,breaks,self.mmax))
            for i,item in enumerate(self.slopes):
                cut = np.where(np.logical_and(self.x>=self.mass_range[i],self.x<self.mass_range[i+1]))
                dn[cut] = np.power(self.x[cut],item)
                if i != 0:
                    renorm = np.divide(last_dn,dn[cut][0])
                    dn[cut] = dn[cut]*renorm
                last_dn = dn[cut][-1]
                last_x = self.x[cut][-1]
            self.dn = np.divide(dn,sum(dn))
            u = self.dn*self.x
            self.dm = np.divide(u,sum(u))
            self.dn = np.divide(self.dm,self.x)
        return (self.dm,self.dn)


    def imf_mass_fraction(self,mlow,mup):

#        Calculates the mass fraction of the IMF sitting between mlow and mup

        norm = sum(self.dm)
        cut = np.where(np.logical_and(self.x>=mlow,self.x<mup))
        fraction = np.divide(sum(self.dm[cut]),norm)
        return(fraction)

    def imf_number_fraction(self,mlow,mup):

#        Calculating the number fraction of stars of the IMF sitting between mlow and mup

        norm = sum(self.dn)
        cut = np.where(np.logical_and(self.x>=mlow,self.x<mup))
        fraction = np.divide(sum(self.dn[cut]),norm)
        return(fraction)

    def imf_number_stars(self,mlow,mup):
        cut = np.where(np.logical_and(self.x>=mlow,self.x<mup))
        number = sum(self.dn[cut])
        return(number)

    def stochastic_sampling(self, mass):

#        The analytic IMF will be resampled according to the mass of the SSP.
#        The IMF will still be normalised to 1

#        Stochastic sampling is realised by fixing the number of expected stars and then drawing from the probability distribution of the number density
#        Statistical properties are tested for this sampling and are safe: number of stars and masses converge.

        number_of_stars = int(round(sum(self.dn) * mass))
        self.dm_copy = np.copy(self.dm)
        self.dn_copy = np.copy(self.dn)

        #self.dn_copy = np.divide(self.dn_copy,sum(self.dn_copy))
        random_number = np.random.uniform(low = 0.0, high = sum(self.dn_copy), size = number_of_stars)
        self.dm = np.zeros_like(self.dm)
        self.dn = np.zeros_like(self.dn)

        ### This could be favourable if the number of stars drawn is low compared to the imf resolution
#        for i in range(number_of_stars):
            ### the next line randomly draws a mass according to the number distribution of stars
#            cut = np.where(np.abs(np.cumsum(self.dn_copy)-random_number[i])== np.min(np.abs(np.cumsum(self.dn_copy)-random_number[i])))
#            x1 = self.x[cut][0]
            #self.dn[cut] += 0.5
#            self.dn[cut[0]] += 1
#            self.dm[cut[0]] += x1 + self.dx/2.
#            t.append(x1 + self.dx/2.)

        counting = np.cumsum(self.dn_copy)
        for i in range(len(counting)-1):
            if i == 0:
                cut = np.where(np.logical_and(random_number>0.,random_number<=counting[i]))
            else:
                cut = np.where(np.logical_and(random_number>counting[i-1],random_number<=counting[i]))
            number_of_stars_in_mass_bin = len(random_number[cut])
            self.dm[i] = number_of_stars_in_mass_bin * self.x[i]
        if number_of_stars:
            self.dm = np.divide(self.dm,sum(self.dm))
        else:
            self.dm = np.divide(self.dm, 1)
        self.dn = np.divide(self.dm,self.x)
        return (self.dm,self.dn)
########
    ## End IMF subroutines
########

## INITIALIZE IMF

imf=IMF(mmin=0.08, mmax=100.0, intervals=5000)
final=imf.normed_3slope() #Select IMF function normed_3slope=Kroupa, Chabrier_1, Chabrier_2, salpeter and BrokenPowerLaw are available

###########
# Evolution of metallicity FeH from observations (non-self consistent with other routines)
# (Used to compute lifetime of stars in the MS, something that is metallicity dependent)
# (In a perfect world this should be uptated at each timestep instead of being an imposed function)
###########
def FeH_obs(t):
    return -0.2+0.04*t #From our sample of N stars and HARPS-GTO/AMBRE-HARPS (Minchev et al. 2018)


###########
# Novae
###########

    #White Dwarf Formation Rate
deltaWD=0. # usually 2Gyr from the WD formation and the first Nova explosion as in Matteucci et al. 2003, 0 if nova lifetime is used 
Nova_lifetime=10**np.arange(7.6, 10.25, 0.1)/1e9
#Nova lifetime before SNIa from https://ui.adsabs.harvard.edu/abs/2005A%26A...441.1055G/abstract
Nova_lifetime_fraction=[0.0,0.045,0.07135,0.081,0.0819,0.081,0.08,0.0773,0.0739,0.0689,0.06,0.0535,0.045,0.04012,0.03576,0.031872,0.02256,0.016,0.0113,0.008,0.0057,0.0038,0.0028,0.002,0.0007,0.00036,0.000178]
Nova_lifetime_fraction_cum=np.cumsum(Nova_lifetime_fraction)
#print(len(log10_Nova_lifetime),len(Nova_lifetime_fraction),np.sum(Nova_lifetime_fraction))
#print(Nova_lifetime)

massbins=np.arange(0.8, 8.0, 0.01)
Nstars_WD=[0.0]*len(massbins)
for i in range(0,len(massbins)-1):
    Nstars_WD[i]=imf.imf_number_stars(massbins[i],massbins[i+1])
Nstars_WD_time=[0.0]*len(time)
SFR_to_use=SFR_total(time,SFR,SFR1,SFR_base)
for i in range(0,len(massbins)-1):
    for t in time:
        massWD=massbins[i]+(massbins[i+1]-massbins[i])/2.0
        Nstars_WD_time_temp=Nstars_WD[i]*SFR_to_use[np.where(time == t)]*bint*1e9
        WD_MSlife=lifetime(massWD,FeH_obs(t))
#        print(Nstars_WD_time_temp,massWD,WD_MSlife,timemax-t)
        if WD_MSlife <= timemax-t-deltaWD:
            ind=builtins.min(enumerate(time), key=lambda x: np.min([abs(WD_MSlife-x[1]+deltaWD+t),bint/2.]))
            index=ind[0]
            Nstars_WD_time[index]=Nstars_WD_time[index]+Nstars_WD_time_temp
ind1=builtins.min(enumerate(time), key=lambda x: abs(deltaWD+mean-2*sig-x[1]))
Ratio_Nova=[0.0]*len(Nstars_WD_time)
Novae=[0.0]*len(Nstars_WD_time)
Novae_O=[0.0]*len(Nstars_WD_time)
Novae_total=[0.0]*len(Nstars_WD_time)
for i in range(0,len(time)):
    if time[i] < deltaWD+mean-2*sig:
        Nstars_WD_time[i]=Nstars_WD_time[np.int(ind1[0])]
    # assuming that stars of any mass between 0.8 and 8 Msun in a binary system can suffer a Nova event
    # we take that the ratio of the number of WD at t=0 vs the observed Nova is mantained in the Galaxy evolution

# We only count WD systems that have not undergone a SNIa explosion (Nova_lifetime_fraction = fraction that undergone a SNIa)    
Nstars_WD_time_all=[0.0]*len(Nstars_WD_time)
#print(Nova_lifetime)
for i in range(0,len(time)):
    ind2=[builtins.min(enumerate(Nova_lifetime), key=lambda x: abs(timemax-time[j]-x[1])) for j in range(0,i+1)]
#    print(ind2[i][0])
#    print(ind2,time[i])
    Nstars_WD_time_all[i]=np.sum([Nstars_WD_time[j]*(1.-Nova_lifetime_fraction_cum[ind2[j][0]]) for j in range(0,i)])
    
#Nstars_WD_time_all=np.cumsum(Nstars_WD_time)
for i in range(0,len(time)):
    t=time[i]
    Ratio_Nova[i]=Novae_Explosion_MW_t0*Nstars_WD_time_all[i]/np.float(WD_MW_t0)#np.float(Nstars_WD_time_all[-1])
    print(np.float(Nstars_WD_time_all[-1]))
    Novae_O[i]=Ratio_Nova[i]*bint*1e9* 8.75e-6# https://arxiv.org/pdf/2201.10125.pdf or for the MW: (5e-4+2e-4+5e-5) #From https://ui.adsabs.harvard.edu/abs/2005ASPC..330..265H/abstract
    Novae[i]=Ratio_Nova[i]*bint*1e9*2.5e-10 #0.3-4.6e-10 value from Izzo et al. 2015 observations, and Starrfield2020 simulations #20-70 Nova/yr in the current MW (we scaled it with the SFR) Cescutti and Molaro 2009
    Novae_total[i]=bint*Ratio_Nova[i]*1e9*7.51e-4

# Models for novae ejecta -> https://ui.adsabs.harvard.edu/abs/2015MNRAS.446.1924H/abstract
#def Novae_total(t,SFR_total,timemax):  #each burst ejects 7x10^-5Msun of gas (Shara et al. 2010, Yaron et al. (2005), Kawanaka et al 2018 https://ui.adsabs.harvard.edu/abs/2018PhRvL.120d1103K/abstract)
#    return SFR_total(t,SFR,SFR1,SFR_base)/SFR_total(timemax,SFR,SFR1,SFR_base)*bint*50*1e9*7e-5 #20-70 Nova/yr in the current MW (we scaled it with the SFR) Cescutti and Molaro 2009
#def Novae(t,SFR_total,timemax): 
#    return SFR_total(t,SFR,SFR1,SFR_base)/SFR_total(timemax,SFR,SFR1,SFR_base)*bint*50*1e9*1.8e-9 #20-70 Nova/yr in the current MW (we scaled it with the SFR) Cescutti and Molaro 2009
#Novae1=Novae(time,SFR_total,timemax)
#print(Novae2,Novae1)

###########
# AGB/LIMS: Tabulated Lithium, oxygen, carbon and nitrogen, and total mass from AGB and LIMS
###########  

Mstar=[0.01,0.081,0.501,1.251,1.751,2.251,2.751,3.251,3.751,4.251,4.751,5.251,5.751,6.251,6.751,7.251,7.751,8.01,17.01]
Mstar1=[0.08,0.5,1.25,1.75,2.25,2.75,3.25,3.75,4.25,4.75,5.25,5.75,6.25,6.75,7.25,7.75,8.0,17.0,100.]
Tdelay=[2.44,1.06,0.7,0.41,0.27,0.19,0.14,0.11,0.09,0.07,0.06,0.05,0.04,0.04]
Mst=[1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0]
Mcore=[0.64,0.70,0.73,0.76,0.82,0.86,0.88,0.91,0.94,0.98,1.04,1.14,1.22,1.32]
ALi_AGB=[-2.73,-2.51,1.95,1.31,2.90,2.71,2.58,2.35,2.59,2.63,2.93,3.27,2.98,3.91]
O_AGB=[0.21,0.22,0.28,0.25,0.85,0.81,0.76,0.71,0.67,0.65,0.65,0.66,0.58,0.74]
C_AGB=[0.36,0.72,1.02,0.86,-0.32,-0.44,-0.97,-1.00,-0.83,-1.06,-0.95,-1.06,-1.07,-1.08]
N_AGB=[0.01,0.01,0.46,0.42,1.11,1.06,1.05,1.04,1.03,1.00,0.99,0.95,0.96,0.93]

#
# In this version we only compute oxygen and lithium
#

Mli_unitmass=10**(np.array(ALi_AGB)-12.)
MO_unitmass=10**(np.array(O_AGB)-12.)+10**(np.array(C_AGB)-12.)+10**(np.array(N_AGB)-12.)
Massfraction_in_ejecta=(np.array(Mst)-np.array(Mcore))/np.array(Mst)

Mbins=[imf.imf_mass_fraction(Mstar[i],Mstar1[i]) for i in range(0,len(Mstar))]
M_LiMS_AGB=np.array(Mbins[3:len(Mbins)-2])

#
#Total stellar mass, lithium and oxygen, in each mass range, created at each instant by LIMS and AGB
#
def LIMS_AGB_total(t,time,M_LiMS_AGB,SFR_total_single_time,bint,Massfraction_in_ejecta):
#    print(SFR_total(t,SFR,SFR1,SFR_base))
    return SFR_total_single_time(t,time,SFR,SFR1,SFR_base)*bint*1e9*M_LiMS_AGB*Massfraction_in_ejecta
def LIMS_AGB_li(t,time,M_LiMS_AGB,SFR_total_single_time,bint,Massfraction_in_ejecta,Mli_unitmass):
    return SFR_total_single_time(t,time,SFR,SFR1,SFR_base)*bint*1e9*M_LiMS_AGB*Massfraction_in_ejecta*Mli_unitmass
def LIMS_AGB_O(t,time,M_LiMS_AGB,SFR_total_single_time,bint,Massfraction_in_ejecta,MO_unitmass):
    return SFR_total_single_time(t,time,SFR,SFR1,SFR_base)*bint*1e9*M_LiMS_AGB*Massfraction_in_ejecta*MO_unitmass

M_t_LIMS_AGB_O=[0.0]*len(time)
M_t_LIMS_AGB_li=[0.0]*len(time)
M_t_LIMS_AGB_tot=[0.0]*len(time)
for j in range(0,len(time)):
    tini=time[j]   
    mO_agb=LIMS_AGB_O(tini,time,M_LiMS_AGB,SFR_total_single_time,bint,Massfraction_in_ejecta,MO_unitmass)
    mli_agb=LIMS_AGB_li(tini,time,M_LiMS_AGB,SFR_total_single_time,bint,Massfraction_in_ejecta,Mli_unitmass)
    mtot_agb=LIMS_AGB_total(tini,time,M_LiMS_AGB,SFR_total_single_time,bint,Massfraction_in_ejecta)
    indexs=[min(range(len(time)), key=lambda i: abs(time[i]-(Tdelay[k]+tini))) for k in range(0,len(Tdelay))]
    for k in range(0,len(mli_agb)):
        M_t_LIMS_AGB_O[indexs[k]]=M_t_LIMS_AGB_O[indexs[k]]+mO_agb[k]
        M_t_LIMS_AGB_li[indexs[k]]=M_t_LIMS_AGB_li[indexs[k]]+mli_agb[k]
        M_t_LIMS_AGB_tot[indexs[k]]=M_t_LIMS_AGB_tot[indexs[k]]+mtot_agb[k]

        
###########
# SNII: Only stars between 8-17 will contribute as SNe, more massive stars fails to produce SNe
###########

Stars_8Msun_above_fraction=imf.imf_mass_fraction(8.0,30.0)
Stars_8Msun_above_fraction_num=imf.imf_number_stars(8.0,30.0)
Stars_89Msun_above_fraction=imf.imf_mass_fraction(8.0,9.0)
Stars_912Msun_above_fraction=imf.imf_mass_fraction(9.001,12.0)
Stars_1215Msun_above_fraction=imf.imf_mass_fraction(12.001,15.0)
Stars_15Msun_above_fraction=imf.imf_mass_fraction(15.001,30.0)
SNII_ejecta=Stars_8Msun_above_fraction*0.9 # 3/4 to 90% of mass in ejecta from Fields,Daigne&Cassé, Stockinger et al. 2020 https://academic.oup.com/mnras/article/496/2/2039/5857660
SNII_li=Stars_8Msun_above_fraction_num*5e-10
SNII_Fe=Stars_8Msun_above_fraction_num*7.4e-2
SNII_O=Stars_8Msun_above_fraction_num*1.55
SNII_li_89=Stars_89Msun_above_fraction*1.5e-10 # ~5*10e-10Msun from (Nomoto et al. 2013) https://star.herts.ac.uk/~chiaki/works/YIELD_CK13.DAT https://www.annualreviews.org/doi/10.1146/annurev-astro-082812-140956
SNII_li_912=Stars_8Msun_above_fraction_num*5e-10
SNII_li_1215=Stars_89Msun_above_fraction*1e-7
SNII_li_15=Stars_89Msun_above_fraction*5e-10
# Also ~ from Nakamura et al. 2010 http://articles.adsabs.harvard.edu/pdf/2010IAUS..268..463N
# Modified to follow Meynet et al. 2014 models 

def SNII(t,SFR_total,SNII_ejecta): 
    return SFR_total(t,SFR,SFR1,SFR_base)*bint*1e9*SNII_ejecta
def SNIIli_0(t,SFR_total,SNII_li89,SNII_li_912,SNII_li_1215,SNII_li_15):
    return SFR_total(t,SFR,SFR1,SFR_base)*bint*1e9*SNII_li_89+SFR_total(t,SFR,SFR1,SFR_base)*bint*1e9*SNII_li_912+SFR_total(t,SFR,SFR1,SFR_base)*bint*1e9*SNII_li_1215+SFR_total(t,SFR,SFR1,SFR_base)*bint*1e9*SNII_li_15
def SNIIli(t,SFR_total,SNII_li):
    return SFR_total(t,SFR,SFR1,SFR_base)*bint*1e9*SNII_li
def SNIIFe(t,SFR_total,SNII_Fe):
    return SFR_total(t,SFR,SFR1,SFR_base)*bint*1e9*SNII_Fe
def SNIIO(t,SFR_total,SNII_O):
    return SFR_total(t,SFR,SFR1,SFR_base)*bint*1e9*SNII_O
    
    
###########    
# SNIa:¡¡do not contribute!! Most of mass in other elements # Novae eruption is the main cotributor
###########

def SNIa(t,SFR_total):
    return SFR_total(t,SFR,SFR1,SFR_base)*bint*1e9*0.002*1.4 #0.002SNIa/Msun from Greggio&Capellaro 2009, 1.4Msun each SNIa (Chandrasekar) and/or Mateucci 2006
def SNIali(t,SFR_total):
    return SFR_total(t,SFR,SFR1,SFR_base)*bint*1e9*0.002*1.4*0.1e-6 #0.1e-6 Msun/SNe from Fields, Daigne and Cassé
def SNIaFe(t,SFR_total):
    return SFR_total(t,SFR,SFR1,SFR_base)*bint*1e9*0.002*1.4*0.25 #25% (0.4Msun (Chiapinni 1997), or 0.7Msun per supernovae in https://ui.adsabs.harvard.edu/abs/2006MNRAS.372..265M/abstract, pag 271, from Nomoto et al. 1997
def SNIaO(t,SFR_total):
    return SFR_total(t,SFR,SFR1,SFR_base)*bint*1e9*0.002*1.4*0.1 #10% in alpha from https://ui.adsabs.harvard.edu/abs/2018ApJ...857...97M/abstract


def SNIa_t(t,SNIa):
    indexs=[min(range(len(t)), key=lambda i: abs(t[i]-(t[k]-2))) for k in range(0,len(t))]
#    for k in range(0,len(t)):
#    sNIa_t=[0.0]*len(t)
    sNIa_0=SNIa(t,SFR_total)
    SNIa_t=[sNIa_0[i] if i != 0 else 0.0 for i in indexs]
    return SNIa_t ## do not contribute!! Most of mass in other elements # Novae eruption is the main cotributor
def SNIali_t(t,SNIali):
    indexs=[min(range(len(t)), key=lambda i: abs(t[i]-(t[k]-2))) for k in range(0,len(t))]
#    for k in range(0,len(t)):
#    sNIali_t=[0.0]*len(t)
    sNIali_0=SNIali(t,SFR_total)
    SNIali_t=[sNIali_0[i] if i != 0 else 0.0 for i in indexs]
    return 0. #SNIali_t ## do not contribute!! Most of mass in other elements # Novae eruption is the main cotributor
def SNIaFe_t(t,SNIaFe):
    indexs=[min(range(len(t)), key=lambda i: abs(t[i]-(t[k]-2))) for k in range(0,len(t))]
#    for k in range(0,len(t)):
#    sNIaFe_t=[0.0]*len(t)
    sNIaFe_0=SNIaFe(t,SFR_total)
    SNIaFe_t=[sNIaFe_0[i] if i != 0 else 0.0 for i in indexs]
#    print(sNIaFe_t,sNIaFe_0,sNIaFe_0[indexs],indexs)
    return SNIaFe_t 
def SNIaO_t(t,SNIaO):
    indexs=[min(range(len(t)), key=lambda i: abs(t[i]-(t[k]-2))) for k in range(0,len(t))]
#    for k in range(0,len(t)):
#    sNIaO_t=[0.0]*len(t)
    sNIaO_0=SNIaO(t,SFR_total)
    SNIaO_t=[sNIaO_0[i] if i != 0 else 0.0 for i in indexs]
    return SNIaO_t 


###########
# Pristine Gas inflow function
###########

def inflow_func(t):
    return np.exp(-(t)**2/6)


###########
# Li depletion mechanisms
###########

# Li depletion in Pre-Main Sequence (Eggenberger 2012 - https://ui.adsabs.harvard.edu/abs/2012A%26A...539A..70E/abstract)
def PMS_deltaLi(Omega_ini,td): # Omega_sun=27.5 (Xie et al. 2017 https://iopscience.iop.org/article/10.3847/1538-3881/aa6199/pdf)
    if td == 3:
        if Omega_ini==20:
            delta_Li=-0.85
        elif Omega_ini==10:
            delta_Li=-0.8
        elif Omega_ini==5:
            delta_Li=-0.775
    elif td == 6:
        if Omega_ini==20:
            delta_Li=-0.95
        elif Omega_ini==10:
            delta_Li=-0.9
        elif Omega_ini==5:
            delta_Li=-0.875
    elif td == 9:
        if Omega_ini==20:
            delta_Li=-1.0
        elif Omega_ini==10:
            delta_Li=-0.95
        elif Omega_ini==5:
            delta_Li=-0.925
    return delta_Li

#  Li depletion in Main Sequence (Dumont et al. 2021 https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..48D/abstract)
# Model x_nu_R1_a_t6.425 in  Dumont et al. 2021   
def Depletion_func(t): 
    return 3.507*np.exp(-0.015*(t+3.9)**2)
def Depletion_func1(t):
    return 2.904-0.565*t+0.069*t**2-0.008*t**3


##################################################################
##################################################################
#              Main computations and initial parameters
##################################################################
##################################################################


normalization_inflow=np.cumsum(inflow_func(time))
norm_inf=normalization_inflow[len(normalization_inflow)-1]
M_tot_ini=inflow_func(time)/norm_inf*3.7e10 #Msun

Li_primordial=2.7
O_primordial=-1.6 #-0.1
Fe_primordial=-1.5 #-0.2
M_li_ini=10**(Li_primordial-12.)*M_tot_ini #Using primordial abundance (2.7) and that A(Li)=log10(Nli/NH)+12
M_Fe_ini=0.00178*10**(Fe_primordial)*M_tot_ini
M_O_ini=0.0161*10**(O_primordial)*M_tot_ini

# Mgas total:
def Mgas_total_polluted(t,SFR_total,bint,M_t_LIMS_AGB_tot,SNII):
    return np.cumsum(Novae_total)+np.cumsum(SNII(t,SFR_total,SNII_ejecta))+np.cumsum(M_t_LIMS_AGB_tot)+np.cumsum(SNIa_t(t,SNIa))#+
def Mgas_total(t,SFR_total,bint,M_t_LIMS_AGB_tot,SNII):
    return np.cumsum(M_tot_ini)-np.cumsum(SFR_total(t,SFR,SFR1,SFR_base)*bint*1e9)+Mgas_total_polluted(t,SFR_total,bint,M_t_LIMS_AGB_tot,SNII)

# mass in Iron peak elements:
def MFe_total(t,SFR_total,bint,SNIIFe):
#    print(SNIaFe_t(t,SNIaFe),SNIIFe(t,SFR_total,SNII_Fe))
    return np.cumsum(M_Fe_ini)-np.cumsum(SFR_total(t,SFR,SFR1,SFR_base)*bint*1e9*ZFe(t,Fe_primordial))+np.cumsum(SNIaFe_t(t,SNIaFe))+np.cumsum(SNIIFe(t,SFR_total,SNII_Fe))

# mass in Alpha elements total:
def MO_total(t,SFR_total,bint,M_t_LIMS_AGB_O,SNIIO):
    return np.cumsum(M_O_ini)-np.cumsum(SFR_total(t,SFR,SFR1,SFR_base)*bint*1e9*ZO(t,O_primordial))+np.cumsum(M_t_LIMS_AGB_O)+np.cumsum(SNIaO_t(t,SNIaO))+np.cumsum(Novae_O)+np.cumsum(SNIIO(t,SFR_total,SNII_O))


# total number of stars formed at each timestep
Nstars_time=[]
for t in time:
    mlow=0.5 # Lower mass of K dwarf stars as in https://ui.adsabs.harvard.edu/abs/2013ApJS..208....9P/abstract
    mup=np.min([np.min([mass[i] if lifetime(mass[i],FeH_obs(t)) < timemax-t else np.max(mass) for i in range(0,len(mass))]),1.4])
    Nstars=imf.imf_number_stars(mlow,mup)
    Nstars_time.append(Nstars)
NStars=Nstars_time*SFR_total(time,SFR,SFR1,SFR_base)*bint*1e9

Nstars_time_all=[]
for t in time:
    mlow=0.5 # Lower mass of K dwarf stars as in https://ui.adsabs.harvard.edu/abs/2013ApJS..208....9P/abstract
    mup=np.min([np.min([mass[i] if lifetime(mass[i],FeH_obs(t)) < timemax-t else np.max(mass) for i in range(0,len(mass))]),1.4])
    Nstars_all=imf.imf_number_stars(mlow,mup)
    Nstars_time_all.append(Nstars_all)
NStars_all=Nstars_time_all*SFR_total(time,SFR,SFR1,SFR_base)*bint*1e9

# Li depletion in PMS
Li_deplet=[Li_deplet[i] if time[-1]-time[i]>0.01 else Li_deplet[i]+PMS_deltaLi(Omega_ini,td) for i in range(0,len(time))]

# Li depletion in MS
Li_deplet=[Li_deplet[i]+Depletion_func1(time[-1])-Depletion_func1(time[i]) for i in range(0,len(time))]

# M_Fe 
MFe_tot=MFe_total(time,SFR_total,bint,SNIIFe)

# M_O 
MO_tot=MO_total(time,SFR_total,bint,M_t_LIMS_AGB_O,SNIIO)

MH_t=np.log10((MO_tot+MFe_tot)/Mgas_total(time,SFR_total,bint,M_t_LIMS_AGB_tot,SNII)/(0.0161+0.00178))
MFe_t=np.log10((MFe_tot)/Mgas_total(time,SFR_total,bint,M_t_LIMS_AGB_tot,SNII)/(0.00178))
MAlpha_t=np.log10((MO_tot)/Mgas_total(time,SFR_total,bint,M_t_LIMS_AGB_tot,SNII)/(0.0161))
Alpha_Fe_t=np.log10(MO_tot/MFe_tot/(0.0161/0.00178))


##########
# Cosmic Rays spallation: needs to be computed here when the total gas mass, and metallicity has been computed
#########
def GCR(t,MFe_t,Mgas_total):
    return 0.5*Mgas_total(t,SFR_total,bint,M_t_LIMS_AGB_tot,SNII)*10**(-9.982+1.24*MFe_t) #Observational relation from Be9 that is only generated by CRs Grisoni et al. 2019, Smiljanic et al. (2009)
# The GCR explanation is wrong in Grisoni et al 2019. They use 7Li/9Be=7.6, but this is only valid for cold stars that depleted 6Li (Molaro et al 1997 New Beryllium observations ...). The value
# that should be used is (6Li+7Li)/9Be=13.1 that is for hotter stars than 6300 K. If we consider 7Li/6Li=2 (Theoretical 
# work for CRs spallation + alpha-alpha fusion in the ISM by Meneguzzi, Audouze & Reeves 1971, also see 
#https://cds.cern.ch/record/511768/files/0107492.pdf?version=1 )
# The Grisoni et al. 2019 equation for Lithium vs. [fe/H] becomes 
## https://ui.adsabs.harvard.edu/abs/2016APh....81...21M/abstract ->> 7Li/9Be crossections = 2.5 ->
#We assume that only 50% of gas is wellmixed and has the mean metallicity of the galaxy


# Mli total:
def Mli_total(t,SFR_total,bint,M_t_LIMS_AGB_li,SNIIli):
    return np.cumsum(M_li_ini)-np.cumsum(SFR_total(t,SFR,SFR1,SFR_base)*bint*1e9*Zli(t,Li_primordial))+np.cumsum(Novae)+np.cumsum(GCR(t,MFe_t,Mgas_total))+np.cumsum(SNIIli(t,SFR_total,SNII_li))+np.cumsum(M_t_LIMS_AGB_li)
#print(np.cumsum(M_t_LIMS_AGB_li),np.cumsum(M_t_LIMS_AGB_tot))

# M_Li non-depleted
Mli_tot=Mli_total(time,SFR_total,bint,M_t_LIMS_AGB_li,SNIIli)

# M_Li depleted
Mli_dep=[Mli_tot[i]*10**(Li_deplet[i]) for i in range(0,len(time))]


##print(MFe_t,MAlpha_t)
##print(Alpha_Fe_t)

x=Mgas_total(time,SFR_total,bint,M_t_LIMS_AGB_tot,SNII)
#x0=Mgas_total_polluted(time,SFR_total,bint,M_t_LIMS_AGB_tot,SNII)
#print(Mgas_total_polluted(time,SFR_total,bint,M_t_LIMS_AGB_tot,SNII)/(Mgas_total(time,SFR_total,bint,M_t_LIMS_AGB_tot,SNII)+np.cumsum(SFR_total(t,SFR,SFR1,SFR_base)*bint*1e9)))
#x1=np.cumsum(Novae(time,SFR_total,timemax))
#x11=np.cumsum(GCR(time,FeH_obs,Mgas_total))
#x2=Mli_total(time,SFR_total,bint,M_t_LIMS_AGB_li,SNIIli)
x21=SFR_total(time,SFR,SFR1,SFR_base)
xli=np.log10(Mli_total(time,SFR_total,bint,M_t_LIMS_AGB_li,SNIIli)/Mgas_total(time,SFR_total,bint,M_t_LIMS_AGB_tot,SNII))+12
xli_deplet=np.log10(Mli_dep/Mgas_total(time,SFR_total,bint,M_t_LIMS_AGB_tot,SNII))+12#[np.max([0,xli[i]+Li_deplet[i]]) for i in range(0,len(time))]
#print(xli_deplet)
#x4=M_t_LIMS_AGB_li
#x0=np.cumsum(SNIali_t(time,SNIali))
x1=np.log10(np.cumsum(Novae)/Mgas_total(time,SFR_total,bint,M_t_LIMS_AGB_tot,SNII))+12
x2=np.log10(np.cumsum(GCR(time,MFe_t,Mgas_total))/Mgas_total(time,SFR_total,bint,M_t_LIMS_AGB_tot,SNII))+12
x3=np.log10(np.cumsum(SNIIli(time,SFR_total,SNII_li))/Mgas_total(time,SFR_total,bint,M_t_LIMS_AGB_tot,SNII))+12
x4=np.log10(np.cumsum(M_t_LIMS_AGB_li)/Mgas_total(time,SFR_total,bint,M_t_LIMS_AGB_tot,SNII))+12

#plt.plot(time,xli_deplet,'g')
plt.plot(time,x1,'b')
plt.plot(time,x2,'r')
plt.plot(time,x3,'c')
#plt.plot(time,x30,'c')
plt.plot(time,x4,'g')
plt.plot(time,xli,'k')
