import matplotlib.pyplot as plt
import numpy as np
import phconvert

measurement = '405Laser1MHz_Power7.ptu'
file=phconvert.pqreader.load_ptu(r'E:/LAB_DATA/Robert/g2again/'+measurement)

hist = np.histogram(file[0],bins=61)[0]
plt.plot(hist)
plt.show()

print (file[3])

reprate = 1.0*10**(-6)
res = 6.4*10**(-11) 

b_width = 10**(-9)
maxbin = 4*10**(-6)
n_bins = int((2*maxbin)/b_width)   
histodat =  np.zeros(n_bins)
xdat= [(-maxbin + (b_width/2)+ b_width*i) for i in range(n_bins)]  
b_list = [-maxbin + b_width*i for i in range(n_bins+1)]  

delays = []
macrodat=file[0]
marker=file[1]
microdat=file[2]
length = len(macrodat)

print (marker[0:50])
print (macrodat[0:50])
print (microdat[0:50])


for j in range(length):
    if marker[j] == 1:
        c_1 = 1
        c_2 = 0
        while c_2 <= 3 and c_1 <= 3:
            if j - (c_2 + c_1) < 0:
                break
            elif marker[j - (c_2  + c_1)] == 0:
                n_delay = (macrodat[j - (c_2  + c_1)]*reprate + microdat[j - (c_2  + c_1)]*res) - (macrodat[j]*reprate + microdat[j]*res)
                #print (n_delay)
                if abs(n_delay) > maxbin:
                    break
                else:
                    delays.append(n_delay)
                    c_2 += 1
            else:
                c_1 += 1
        c_1 = 1
        c_2 = 0
        while c_2 <= 3 and c_1 <= 3:
            if j + (c_2 + c_1) > length-1:
                break
            elif marker[j + (c_2 + c_1)] == 0:
                p_delay = (macrodat[j + (c_2  + c_1)]*reprate + microdat[j + (c_2  + c_1)]*res) - (macrodat[j]*reprate + microdat[j]*res)  
            
                if p_delay > maxbin:
                    break
                else:
                    delays.append(p_delay)
                    c_2 += 1
            else:
                c_1 += 1               
    if j%10**5==0:
        print (j)
        
        
print (max(delays),len(delays))
print (np.mean(delays))
histo = np.histogram(delays, bins = b_list)[0]
           

#print (delays[0:100])

plt.plot(histo)
#plt.xlim((4000,6000))
#plt.yscale('Log')
plt.show()

np.savetxt('g2_'+measurement+'.dat', histo, fmt = '%f')  


   
