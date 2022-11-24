import matplotlib.pyplot as plt

import numpy as np

plt.annotate('Top Max',xy = (4.3,1),xytext = (4.6,0.25),arrowprops = dict(facecolor='m',shrink=0.01),fontsize = 10)

t1 = np.arange(0.0,5.0,0.02)

plt.plot(t1,np.sin(2*np.pi*t1),color = 'g',marker= '+',linestyle='-.')

plt.show()