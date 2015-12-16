__author__ = 'Jiyang Zhou'
#THIS CODE USES MH METHOD TO ESTIMATE 3D POINTS FROM 2D SPACE DATA
#Final assignment of F2015 ISTA 521 by Dr.Clayton T. Morrison
#ALL YOU NEED TO CHANGE IS THE MATRIX IN projection AND r data!!!!
#Time:2015-12-11 14:04:00

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def read_data(filepath, d = ','):
    """ returns an np.array of the data """
    return np.genfromtxt(filepath, delimiter=d, dtype=None)
def projection(p):

    M = np.matrix([[1, 0, 0, 0,],[0,1,0,0],[0,0,1,0]])       #for Q1-Q4

    #M = np.matrix([[0, 0, 1, -5,],[0,1,0,0],[-1,0,0,5]])      #FOR Q5


    p_temp = np.append(p,1)
    q_temp = M.dot(p_temp).T
    q = np.array([q_temp[0,0]/q_temp[2,0],q_temp[1,0]/q_temp[2,0]])
    return  q

def get_qs( q_start, q_end ,t):
    return q_start + t*(q_end-q_start)


def target(p_i,p_f,t,size):
    prior = multivariate_normal(mu,sigma_1)
    likelihood = prior.logpdf(p_i.T) + prior.logpdf(p_f.T)

    q_i = projection(p_i)
    q_f = projection(p_f)
    for i in range(size):
        f = multivariate_normal(get_qs(q_i,q_f,t[i]),sigma)
        likelihood = likelihood + f.logpdf(r[i,:])
    return likelihood

def projection_verify(p):
    #This fuction is only used for verify Q5
    M = np.matrix([[1, 0, 0, 0,],[0,1,0,0],[0,0,1,0]])


    p_temp = np.append(p,1)
    q_temp = M.dot(p_temp).T
    q = np.array([q_temp[0,0]/q_temp[2,0],q_temp[1,0]/q_temp[2,0]])
    return  q

N = 10000 #number of iteration


t_data = 'project_option_A_data_release/inputs.csv'
t = read_data(t_data, ',')



r_data = 'project_option_A_data_release/points_2d.csv'    #data for Q1-Q4
#r_data = 'project_option_A_data_release/points_2d_2.csv'  #for Q5
r = read_data(r_data, ',')

size = r.shape[0]
r_i = (r[0,0],r[0,1])
r_f = (r[size-1,0],r[size-1,1])


sigma = np.identity(2)*np.power(0.05,2)
sigma_1 = np.identity(3)*10
mu = np.array([0,0,4]).T



#initial pi and pf around [0,0,4]
p= np.random.multivariate_normal(mu, sigma_1 ,2).T
p_i = p[:,0]
p_f =  p[:,1]



q_i = projection(p_i)
q_f = projection(p_f)


avg_pi = 0
avg_pf = 0

record_i = p_i.reshape(1,3)
record_f = p_f.reshape(1,3)


#change the value of random walk step for precision
rand_walk_sigma = np.identity(3)*0.5



for kkk in range(N):

    p_i_new = np.random.multivariate_normal(p_i, rand_walk_sigma).T
    p_f_new = np.random.multivariate_normal(p_f, rand_walk_sigma).T
    avg_pi += p_i_new
    avg_pf += p_f_new

    rho = min(0,target(p_i_new,p_f_new,t,size)-target(p_i,p_f,t,size))

    u = np.random.uniform()
    record_i = np.append(record_i,p_i_new.reshape(1,3),axis=0)
    record_f = np.append(record_f,p_f_new.reshape(1,3),axis=0)


    if np.log(u) <= rho:
        p_i = p_i_new
        p_f = p_f_new



avg_pi = avg_pi/N
avg_pf = avg_pf/N

print("The final result of MAP  \n   pi {0}\n   pf {1} ".format(p_i,p_f))
print("Related 2d projection\n   qi {0}\n   qf {1}".format(projection(p_i),projection(p_f)))

print("\nMonte carlo result: \n   pi_avg {0}\n   pf_avg {1}".format(avg_pi,avg_pf))


print("\nUse the monte carlo result, we predict the point at t = 1.5 in 2d:\n    p ={0}"\
      .format(get_qs(projection(avg_pi),projection(avg_pf),1.5)))

#Verify
if(r_data == 'project_option_A_data_release/points_2d_2.csv'):
    print("\nFor Q5 ONLY!\nThe results projected into first Matrix [1, 0, 0, 0,],[0,1,0,0],[0,0,1,0] \n   qi {0}\n   qf {1}"\
      .format(projection_verify(p_i),projection_verify(p_f)))



plt.figure(1)
plt.plot(r[:,0],r[:,1])
plt.plot(projection(p_i)[0],projection(p_i)[1],'o')
plt.plot(projection(p_f)[0],projection(p_f)[1],'x')
plt.plot(get_qs(projection(avg_pi),projection(avg_pf),1.5)[0],get_qs(projection(avg_pi),projection(avg_pf),1.5)[1],\
         'D',label='p at t = 1.5')






plt.figure(2)
plt.suptitle('x,y,z for p_i(up) & p_f(down) with iteration {0}'.format(N),fontsize=18)
plt.subplot(2,3,1)
plt.plot(np.linspace(0,record_i.shape[0],record_i.shape[0]),record_i[:,0])
plt.subplot(2,3,2)
plt.plot(np.linspace(0,record_i.shape[0],record_i.shape[0]),record_i[:,1])
plt.subplot(2,3,3)
plt.plot(np.linspace(0,record_i.shape[0],record_i.shape[0]),record_i[:,2])
plt.subplot(2,3,4)
plt.plot(np.linspace(0,record_i.shape[0],record_i.shape[0]),record_f[:,0])
plt.subplot(2,3,5)
plt.plot(np.linspace(0,record_i.shape[0],record_i.shape[0]),record_f[:,1])
plt.subplot(2,3,6)
plt.plot(np.linspace(0,record_i.shape[0],record_i.shape[0]),record_f[:,2])
plt.show()

