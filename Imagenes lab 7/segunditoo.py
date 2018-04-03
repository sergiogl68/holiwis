from segmentByClustering import segmentByClustering
import os
import os.path as osp
import numpy as np
import scipy.io as sio
from PIL import Image
count = 1
ks = range(3, 31, 3)
for root, dirs, files in os.walk('.', topdown=False):
    print(root)
    if(root.startswith('./BSR/BSDS500/data/images/val') or root.startswith('./BSR/BSDS500/data/images/test') or root.startswith('./BSR/BSDS500/data/images/train')):
        for f in files:
            if f.endswith('.jpg'):
                s = root + '/' + f
                im = Image.open(s)
                im.load()
                img_act = np.asarray(im, dtype='uint8')
                print('********')
                print(f)
                print(count)
                nsf = np.zeros((len(ks,)), dtype= np.object)
                nss = np.zeros((len(ks),), dtype= np.object)
                i = 0
                for pos in ks:
                    print((i+1))
                    print('----------')
                    nsf[i] = segmentByClustering(img_act, 'lab+xy', 'kmeans', pos)
                    nss[i] = segmentByClustering(img_act, 'rgb+xy', 'kmeans', pos)
                    i+=1
                name = f.split('.')
                dv = 'results_lab_val_kmeans'
                ddv = 'results_rgb_val_kmeans'
                dt = 'results_lab_test_kmeans'
                ddt = 'results_rgb_test_kmeans'
                dr = 'results_lab_train_kmeans'
                ddr = 'results_rgb_train_kmeans'
                if not osp.exists(dv):
                    os.mkdir(dv)
                if not osp.exists(ddv):
                    os.mkdir(ddv)
                if not osp.exists(dt):
                    os.mkdir(dt)
                if not osp.exists(ddt):
                    os.mkdir(ddt)
                if not osp.exists(dr):
                    os.mkdir(dr)
                if not osp.exists(ddr):
                    os.mkdir(ddr)
                n1 = dv + '/' + name[0] + '.mat'
                n2 = ddv + '/' + name[0] + '.mat'
                n3 = dt + '/' + name[0] + '.mat'
                n4 = ddt + '/' + name[0] + '.mat'
                n5 = dr + '/' + name[0] + '.mat'
                n6 = ddr + '/' + name[0] + '.mat'
                h = root.split('/')
                if h[-1] == 'val':
                    sio.savemat(n1, {'segs': nsf})
                    sio.savemat(n2, {'segs': nss})
                elif h[-1].startswith('tr'):
                    sio.savemat(n5, {'segs': nsf})
                    sio.savemat(n6, {'segs': nss})
                else:
                    sio.savemat(n3, {'segs': nsf})
                    sio.savemat(n4, {'segs': nss})
                count += 1
