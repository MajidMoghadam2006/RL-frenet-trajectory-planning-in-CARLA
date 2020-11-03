import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

# cte
max_cte = 2
cte = np.arange(0, max_cte, 0.01)
r_cte0 = -cte/max_cte # 1 + np.sin(2 * np.pi * t)
# r_cte1 = np.exp(-cte**2/max_cte)-1
w = 10
r_cte2 = np.exp(-cte**2/max_cte*w)-1
fig, ax = plt.subplots()
ax.plot(cte, r_cte0)
# ax.plot(cte, r_cte1)
ax.plot(cte, r_cte2)
ax.set(xlabel='cte', ylabel='rew',
       title='Reward functions')
ax.grid()
# plt.show()


# THETA
max_theta = math.pi/2
theta = np.arange(0, max_theta, 0.01)
r_theta0 = -theta/max_theta # 1 + np.sin(2 * np.pi * t)
# r_theta1 = np.exp(-theta**2/max_theta)-1
w = 12
r_theta2 = np.exp(-theta**2/max_theta*w)-1
fig, ax2 = plt.subplots()
ax2.plot(theta, r_theta0)
# ax2.plot(theta, r_theta1)
ax2.plot(theta, r_theta2)
ax2.set(xlabel='theta', ylabel='rew',
       title='Reward functions')
ax2.grid()


# angular velocity norm
max_w = math.sqrt(2 * 180 ** 2) / 4
w = np.arange(0, max_w, 0.01)
r_w0 = -w/max_w # 1 + np.sin(2 * np.pi * t)
# r_w1 = np.exp(-w**2/max_w)-1
ww = 1/5
r_w2 = np.exp(-w**2/max_w*ww)-1
fig, ax3 = plt.subplots()
ax3.plot(w, r_w0)
# ax3.plot(w, r_w1)
ax3.plot(w, r_w2)
ax3.set(xlabel='w', ylabel='rew',
       title='Reward functions')
ax3.grid()

# speed
max_v = 90
e_v = np.arange(0, max_v, 0.01)
r_v0 = -e_v/max_v # 1 + np.sin(2 * np.pi * t)
w = 10
r_v2 = np.exp(-e_v**2/max_v*w)
fig, ax4 = plt.subplots()
ax4.plot(e_v, r_v0)
# ax3.plot(w, r_w1)
ax4.plot(e_v, r_v2)
ax4.set(xlabel='e_v', ylabel='rew',
       title='Reward functions')
ax4.grid()
plt.show()