import matplotlib.pyplot  as plt
import numpy as np
a=np.load('FinalResults5.npy')
b=np.mean(a[:,:,0],axis=0)
c=np.mean(a[:,:,1],axis=0)
d=np.mean(a[:,:,2],axis=0)
e=np.mean(a[:,:,3],axis=0)
print(b.shape)

dataKM={'RGB':b[0],'lab':b[1],'hsv':b[2],'RGB+xy':b[3],'lab+xy':b[4],'hsv+xy':b[5]}
dataGMM={'RGB':c[0],'lab':c[1],'hsv':c[2],'RGB+xy':c[3],'lab+xy':c[4],'hsv+xy':c[5]}
dataHR={'RGB':d[0],'lab':d[1],'hsv':d[2],'RGB+xy':d[3],'lab+xy':d[4],'hsv+xy':d[5]}
dataWSH={'RGB':e[0],'lab':e[1],'hsv':e[2],'RGB+xy':e[3],'lab+xy':e[4],'hsv+xy':e[5]}


names=list(dataKM.keys())
valuesKM=list(dataKM.values())
valuesGMM=list(dataGMM.values())
valuesHR=list(dataHR.values())
valuesWSH=list(dataWSH.values())

fig=plt.figure()
ax1=fig.add_subplot(111)

ax1.scatter(names,valuesKM,label='K-means', color='xkcd:green')
ax1.scatter(names,valuesGMM,label='GMM', color='xkcd:blue')
ax1.scatter(names,valuesHR,label='Hierarchical', color='xkcd:red')
ax1.scatter(names,valuesWSH,label='Watersheds', color='xkcd:azure')


plt.title('Average Jaccard index per feature space')
plt.xlabel('Feature Space')
plt.ylabel('Average Jaccard Index')

plt.legend(loc='upper left')
plt.savefig('Jaccard 5 clus eq')
plt.show()

