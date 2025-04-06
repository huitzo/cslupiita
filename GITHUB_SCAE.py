#!/usr/bin/env python3
import matplotlib.pyplot as plt
import math, gudhi, sys
import numpy as np
#########################################################################################################################################################
def math_com(N, k):
    """
    This function computes the combinations C(N,k)=N!/(k!(N-k)!)
    """
    return math.comb(N, k)
#########################################################################################################################################################
def vr_complex(points, max_dim, epsilon):
    """
    This function computes Vietoris-Rips Simplicial Complex of a point cloud.
    --> points: array-like of shape (N,m) with the point cloud embedded in a m-dimensional space.
    --> max_dim: maxumun dimenson of simplicial complex.
    --> epsilon: maximum distance to which an edge is created in the simplicial complex.
    <-- return: a gudhi generator skeleton containing the tuples ([v_1, v_2...], birth) where v_i are vertices of each simplex of dimension i-1.
    """
    rips_complex = gudhi.RipsComplex(points=points, max_edge_length=epsilon)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
    skeleton = simplex_tree.get_skeleton(max_dim)
    return skeleton
#########################################################################################################################################################
def simplex_number(skeleton):
    """
    This function computes the number of simplexes for each dimension.
    --> skeleton: a gudhi generator skeleton containing the tuples ([v_1, v_2...], birth) where v_i are vertices of each simplex of dimension i-1.
    <-- return: a dictionary with the number of the simplices for each dimension in the simplicial complex.
    """
    simplexes_counter = {dim:0 for dim in range(max_dim+1)}
    for simplex, _ in skeleton:
        simplexes_counter[len(simplex)-1] += 1
    return simplexes_counter
#########################################################################################################################################################
def simplicial_ratio(simplexes_counter, N):
    """
    This function, for each given dimension, computes the ratio between simplexes formed in a simplicial complex and all the possible simplexes that
        could be formed in the point cloud.
    --> simplexes_counter: a dictionary containing the number of simplexes formed in the simplicial complex for each dimension.
    --> N: number of points in the point cloud over the simplicial complex is formed with.
    <-- return: a dictionary with the ratio of the simplexes formed in the simplicial complex respecto to the all possible simplexes that could be formed 
        for each dimension.
    """
    ratios = {}
    for dim in simplexes_counter:
        tks = math_com(N, dim+1)
        ratios[dim] = simplexes_counter[dim]/tks
    return ratios
#########################################################################################################################################################
def scae0(ratios):
    """
    This function, for each given dimension, computes the simplicial complex aproximate entropy.
    --> ratios: a dictionary containing the ratios of simplexes formed in the simplicial complex for each dimension.
    <-- return: the simplicial complex aproximate entropy value SCAE0.
    """
    
    if  ratios[0]>0 and ratios[1]>0:
        s = -np.log(ratios[1]/ratios[0])
    else:
        s = 0
    return s
#########################################################################################################################################################
def scae1(ratios):
    """
    This function, for each given dimension, computes the simplicial complex aproximate entropy.
    --> ratios: a dictionary containing the ratios of simplexes formed in the simplicial complex for each dimension.
    <-- return: the simplicial complex aproximate entropy value SCAE1.
    """
    
    if  ratios[1]>0 and ratios[2]>0:
        s = -np.log(ratios[2]/ratios[1])
    else:
        #s = np.nan
        s = 0
    return s
#########################################################################################################################################################
def coarse_graining(series, tau, r):
    """ 
    This function calculates the coarse-graining procedure.
    --> series: is the time series (this is a list or a numpy array),
    --> tau: is the scale parameter at which the coarse-graining procedure will be applied. Illustrative example of tau=2 and tau=3 are presented in Fig.1(a) of the paper,
    --> r: is the initial point at which the coarse-graining procedure begins (remember minimum value of r is zero (first position in python), and the maximum value is r=tau-1),
    <-- return: the resulting coarse-grained time series.
    """

    i = r; f = i+tau; New_Series=[]
    while f <= len(series):
        New_Series.append(np.mean(series[i:f]))
        i = i+tau; f = i+tau

    return New_Series
#########################################################################################################################################################
max_dim = 2 #defines maximum homology dimension for TDA
epsilon_max=0.1 #defines filtration radii for Vietoris-Rips
k=max_dim-1 #defines subindex k of the SCAE_k
#########################################################################################################################################################
#########################################################################################################################################################
#Example1
White_Noise = np.random.normal(0,1,10000) #generates a white noise of len N=10,000
Signal = coarse_graining(White_Noise, 2, 0) #calculates the coarse-graining of the white noise with parameters (tau=2) and (r=0)

#Now we will calculate the SCAE0 and SCAE1
point_cloud = np.array([Signal[1:], Signal[:-1]]).T; N = len(point_cloud) #constructs the point cloud to apply the Topological Data Analysis
skeleton = vr_complex(point_cloud, max_dim, epsilon_max) #computes the Vietoris-Rips
simplexes_counter = simplex_number(skeleton) #computes the number of simplexes for each dimension
ratios = simplicial_ratio(simplexes_counter, N) #computes the ratio between simplexes formed in a simplicial complex and all the possible simplexes that could be formed in the point cloud.

scae_0 = scae0(ratios) #calculates SCAE0
scae_1 = scae1(ratios) #calculates SCAE1

print('Example 1')
print('SCAE0='+str(scae_0)+', tau=2 and r=0')
print('SCAE1='+str(scae_1)+', tau=2 and r=0')
#########################################################################################################################################################
#########################################################################################################################################################
#Example2. Lets reproduce a curve of Figure4
Scale_paremeters= np.arange(1,21,1) #generates scale parameters used un Fig4
Average_SCAE0_per_tau=[] #generates void list for saving mean of SCAE0 for each scale parameter tau
Average_SCAE1_per_tau=[] #generates void list for saving mean of SCAE1 for each scale parameter tau
Std_SCAE0_per_tau=[] #generates void list for saving standard deviation of SCAE1 for each scale parameter tau
Std_SCAE1_per_tau=[] #generates void list for saving standard deviation of SCAE1 for each scale parameter tau


White_Noise = np.random.normal(0,1,10000) #generates a white noise of len N=10,000
for tau in Scale_paremeters:
    For_Average_SCAE_0=[] #saves SCAE0 for a fixed tau while r varies from 0 until tau-1
    For_Average_SCAE_1=[] #saves SCAE1 for a fixed tau while r varies from 0 until tau-1
    for r in range(tau):
        Signal = coarse_graining(White_Noise, tau, r) #we calculate the coarse-graining of the white noise with (1<=tau<=20) and (r=0)

        #Now we will calculate the SCAE0 and SCAE1
        point_cloud = np.array([Signal[1:], Signal[:-1]]).T; N = len(point_cloud) #constructs the point cloud to apply the Topological Data Analysis
        skeleton = vr_complex(point_cloud, max_dim, epsilon_max) #computes the Vietoris-Rips
        simplexes_counter = simplex_number(skeleton) #computes the number of simplexes for each dimension
        ratios = simplicial_ratio(simplexes_counter, N) #computes the ratio between simplexes formed in a simplicial complex and all the possible simplexes that could be formed in the point cloud.

        For_Average_SCAE_0.append(scae0(ratios)) #calculates and saves SCAE0
        For_Average_SCAE_1.append(scae1(ratios)) #calculates and saves SCAE1

    Average_SCAE0_per_tau.append(np.mean(For_Average_SCAE_0))
    Std_SCAE0_per_tau.append(np.std(For_Average_SCAE_0))
    Average_SCAE1_per_tau.append(np.mean(For_Average_SCAE_1))
    Std_SCAE1_per_tau.append(np.std(For_Average_SCAE_1))


plt.errorbar(x=Scale_paremeters, y=Average_SCAE0_per_tau, yerr=Std_SCAE0_per_tau, label=r'$SCAE_0$')
plt.errorbar(x=Scale_paremeters, y=Average_SCAE1_per_tau, yerr=Std_SCAE1_per_tau, label=r'$SCAE_1$')
plt.title('Example 2')
plt.ylabel(r'$SCAE$')
plt.xlabel(r'scale parameter $\tau$')
plt.legend()

plt.show()
#########################################################################################################################################################
#########################################################################################################################################################
#Example3. Lets reproduce a curve of Figure2
Scale_paremeters= np.arange(1,21,1) #generates scale parameters used un Fig4

Average_SCAE0_per_tau_Healthy=[] #generates void list for saving mean of SCAE0 for each scale parameter tau
Average_SCAE1_per_tau_Healthy=[] #generates void list for saving mean of SCAE1 for each scale parameter tau
Std_SCAE0_per_tau_Healthy=[] #generates void list for saving standard deviation of SCAE0 for each scale parameter tau
Std_SCAE1_per_tau_Healthy=[] #generates void list for saving standard deviation of SCAE1 for each scale parameter tau
#################################################################################################################
A = np.loadtxt('Data_Healthy.dat') #loads sample data of a healthy patient
data = A/np.std(A) #normalizes time-series
for tau in Scale_paremeters:
    For_Average_SCAE_0=[] #saves SCAE0 for a fixed tau while r varies from 0 until tau-1
    For_Average_SCAE_1=[] #saves SCAE1 for a fixed tau while r varies from 0 until tau-1
    for r in range(tau):
        Signal = coarse_graining(data, tau, r) #we calculate the coarse-graining of the white noise with parameters (1<=tau<=20) and (r<=tau)
        #Now we will calculate the SCAE0 and SCAE1
        point_cloud = np.array([Signal[1:], Signal[:-1]]).T; N = len(point_cloud) #constructs the point cloud to apply the Topological Data Analysis
        skeleton = vr_complex(point_cloud, max_dim, epsilon_max) #computes the Vietoris-Rips
        simplexes_counter = simplex_number(skeleton) #computes the number of simplexes for each dimension
        ratios = simplicial_ratio(simplexes_counter, N) #computes the ratio between simplexes formed in a simplicial complex and all the possible simplexes that could be formed in the point cloud.
        For_Average_SCAE_0.append(scae0(ratios)) #calculates and saves SCAE0
        For_Average_SCAE_1.append(scae1(ratios)) #calculates and saves SCAE1

    Average_SCAE0_per_tau_Healthy.append(np.mean(For_Average_SCAE_0))
    Std_SCAE0_per_tau_Healthy.append(np.std(For_Average_SCAE_0))
    Average_SCAE1_per_tau_Healthy.append(np.mean(For_Average_SCAE_1))
    Std_SCAE1_per_tau_Healthy.append(np.std(For_Average_SCAE_1))
#################################################################################################################
Average_SCAE0_per_tau_CHF=[] #generates void list for saving mean of SCAE0 for each scale parameter tau
Average_SCAE1_per_tau_CHF=[] #generates void list for saving mean of SCAE1 for each scale parameter tau
Std_SCAE0_per_tau_CHF=[] #generates void list for saving standard deviation of SCAE0 for each scale parameter tau
Std_SCAE1_per_tau_CHF=[] #generates void list for saving standard deviation of SCAE1 for each scale parameter tau
B = np.loadtxt('Data_Congestive_Heart_Failure.dat') #loads sample data of a congestive heart failure patient
data = B/np.std(B) #normalizes time-series
for tau in Scale_paremeters:
    For_Average_SCAE_0=[] #saves SCAE0 for a fixed tau while r varies from 0 until tau-1
    For_Average_SCAE_1=[] #saves SCAE1 for a fixed tau while r varies from 0 until tau-1
    for r in range(tau):
        Signal = coarse_graining(data, tau, r) #we calculate the coarse-graining of the white noise with parameters (tau=2) and (r=0)
        #Now we will calculate the SCAE0 and SCAE1
        point_cloud = np.array([Signal[1:], Signal[:-1]]).T; N = len(point_cloud) #constructs the point cloud to apply the Topological Data Analysis
        skeleton = vr_complex(point_cloud, max_dim, epsilon_max) #computes the Vietoris-Rips
        simplexes_counter = simplex_number(skeleton) #computes the number of simplexes for each dimension
        ratios = simplicial_ratio(simplexes_counter, N) #computes the ratio between simplexes formed in a simplicial complex and all the possible simplexes that could be formed in the point cloud.
        For_Average_SCAE_0.append(scae0(ratios)) #calculates and saves SCAE0
        For_Average_SCAE_1.append(scae1(ratios))  #calculates and saves SCAE1

    Average_SCAE0_per_tau_CHF.append(np.mean(For_Average_SCAE_0))
    Std_SCAE0_per_tau_CHF.append(np.std(For_Average_SCAE_0))
    Average_SCAE1_per_tau_CHF.append(np.mean(For_Average_SCAE_1))
    Std_SCAE1_per_tau_CHF.append(np.std(For_Average_SCAE_1))
#################################################################################################################
Average_SCAE0_per_tau_AF=[] #generates void list for saving mean of SCAE0 for each scale parameter tau
Average_SCAE1_per_tau_AF=[] #generates void list for saving mean of SCAE1 for each scale parameter tau
Std_SCAE0_per_tau_AF=[] #generates void list for saving standard deviation of SCAE0 for each scale parameter tau
Std_SCAE1_per_tau_AF=[] #generates void list for saving standard deviation of SCAE1 for each scale parameter tau
C = np.loadtxt('Data_Atrial_Fibrilation.dat') #loads sample data of an atrial fibrilation patient
data = C/np.std(C) #normalizes time-series
for tau in Scale_paremeters:
    For_Average_SCAE_0=[] #saves SCAE0 for a fixed tau while r varies from 0 until tau-1
    For_Average_SCAE_1=[] #saves SCAE1 for a fixed tau while r varies from 0 until tau-1
    for r in range(tau):
        Signal = coarse_graining(data, tau, r) #we calculate the coarse-graining of the white noise with parameters (tau=2) and (r=0)
        #Now we will calculate the SCAE0 and SCAE1
        point_cloud = np.array([Signal[1:], Signal[:-1]]).T; N = len(point_cloud) #constructs the point cloud to apply the Topological Data Analysis
        skeleton = vr_complex(point_cloud, max_dim, epsilon_max) #computes the Vietoris-Rips
        simplexes_counter = simplex_number(skeleton) #computes the number of simplexes for each dimension
        ratios = simplicial_ratio(simplexes_counter, N) #computes the ratio between simplexes formed in a simplicial complex and all the possible simplexes that could be formed in the point cloud.
        For_Average_SCAE_0.append(scae0(ratios))  #calculates and saves SCAE0
        For_Average_SCAE_1.append(scae1(ratios)) #calculates SCAE1

    Average_SCAE0_per_tau_AF.append(np.mean(For_Average_SCAE_0))
    Std_SCAE0_per_tau_AF.append(np.std(For_Average_SCAE_0))
    Average_SCAE1_per_tau_AF.append(np.mean(For_Average_SCAE_1))
    Std_SCAE1_per_tau_AF.append(np.std(For_Average_SCAE_1))
#################################################################################################################
plt.subplot(1,2,1)
MEANs_SCAE0 = [Average_SCAE0_per_tau_Healthy, Average_SCAE0_per_tau_CHF, Average_SCAE0_per_tau_AF]
STDs_SCAE0 = [Std_SCAE0_per_tau_Healthy, Std_SCAE0_per_tau_CHF, Std_SCAE0_per_tau_AF]
for mean_SCAE0, std_SCAE0, name in zip(MEANs_SCAE0, STDs_SCAE0, ['Healthy', 'CHF', 'AF']):
    plt.errorbar(x=Scale_paremeters, y=mean_SCAE0, yerr=std_SCAE0, label=name)
plt.title('Example 3')
plt.ylabel(r'$SCAE_0$')
plt.xlabel(r'scale parameter $\tau$')
plt.legend()

plt.subplot(1,2,2)
MEANs_SCAE1 = [Average_SCAE1_per_tau_Healthy, Average_SCAE1_per_tau_CHF, Average_SCAE1_per_tau_AF]
STDs_SCAE1 = [Std_SCAE1_per_tau_Healthy, Std_SCAE1_per_tau_CHF, Std_SCAE1_per_tau_AF]
for mean_SCAE1, std_SCAE1, name in zip(MEANs_SCAE1, STDs_SCAE1, ['Healthy', 'CHF', 'AF']):
    plt.errorbar(x=Scale_paremeters, y=mean_SCAE1, yerr=std_SCAE1, label=name)
plt.title('Example 3')
plt.ylabel(r'$SCAE_1$')
plt.xlabel(r'scale parameter $\tau$')
plt.legend()
plt.tight_layout()
plt.show()
#########################################################################################################################################################