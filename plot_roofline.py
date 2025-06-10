from roofline_data import *
import numpy as np
from matplotlib import pyplot as plt


print("===========================loading data=========================")
print(inputs)

print("================================================================")



rooflinePoints = rnn_roofline(inputs)


plt.rcParams["figure.figsize"] = [6,4]
plt.rcParams["figure.autolayout"] = True

peak = 607.2
textLoc = 450


def f(x):
   return np.minimum(6.9*x, peak)

x = np.linspace(0, 1800, 1800, dtype=np.float16)


x100 = np.linspace(0, 100, 100, dtype=np.float16)
plt.plot(x100, 6.9*x100, '--',color='orange',label='Memory Bandwidth')
plt.plot(x, 0*x + peak, '--',color='teal', label = 'Peak Performance')

plt.plot(x, f(x), '-')
for p in rooflinePoints:
    plt.plot([p[0]], [p[1]], marker="s", markersize=4, label=p[2] + ': '+str(round(p[1],1))+' GOPS')

for d in dspStat:
    localPeak = round(d[0],1)
    plt.plot(x, 0*x + localPeak, '--', color ='grey')
    plt.annotate(str(localPeak) + ' GOPS ' , #+ d[1]
        xy     = (     x[-1], localPeak),
        xytext = (textLoc,  localPeak+10),
        color = 'grey'
    )

plt.xlabel("OP/Byte")
plt.ylabel("GOPS")
plt.legend( loc='lower right')


plt.annotate('6.9 GB/sec',
  xy     = (     x100[-1],  6.5*x100[-1]),
  xytext = (0.85*x100[-1],  6.9*x100[-1]),
  color  = 'orange',
)

plt.annotate(str(peak) + ' GOPS', # - 1518 DSPs',
  xy     = (     x[-1], peak),
  xytext = (textLoc,  peak+10),
  color  = 'teal',
)

# plt.annotate('28.6 GOPS',
#   xy     = (     x[-1], 30),
#   xytext = (x[-1]-10,  40),
#   color  = 'hotpink',
# )

# plt.title('Roofline model for Various RNNs')
plt.savefig("roofline_plot.pdf", format="pdf")
plt.show()