from math import sqrt
from math import atan2
from math import cos
from math import sin
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Frames = 100

fig = plt.figure()
ax1 = fig.add_subplot(212, projection="3d")
ax1.set(xlim=[-3, 3], ylim=[-3, 3], zlim=[-3, 3])
ax2 = fig.add_subplot(221)
ax3 = fig.add_subplot(222)


# Set plate parametrs
LenghtOfPlate = 4
WidthOfPlate = 4
HeightOfStand = -2

# Set movements of plate and point
Diagonal = sqrt(LenghtOfPlate**2 + WidthOfPlate**2)/2
alpha = atan2(LenghtOfPlate, WidthOfPlate)
t = sp.Symbol('t')
s = Diagonal * sp.cos(t)
phi = t / 2
x = s*sp.cos(phi)*cos(alpha)
y = s*sp.sin(phi)*cos(alpha)
z = s*sin(alpha)
vx = sp.diff(x, t)
vy = sp.diff(y, t)
vz = sp.diff(y, t)
wx = sp.diff(vx, t)
wy = sp.diff(vy, t)
wz = sp.diff(vy, t)

T = np.linspace(0, 4*np.pi, Frames)
phi_def = sp.lambdify(t, phi)
s_def = sp.lambdify(t, s)
x_def = sp.lambdify(t, x)
y_def = sp.lambdify(t, y)
z_def = sp.lambdify(t, z)
VX_def = sp.lambdify(t, vx)
VY_def = sp.lambdify(t, vy)
VZ_def = sp.lambdify(t, vz)
WX_def = sp.lambdify(t, wx)
WY_def = sp.lambdify(t, wy)
WZ_def = sp.lambdify(t, wz)

Phi = phi_def(T)
S = s_def(T)
X = x_def(T)
Y = y_def(T)
Z = z_def(T)
VX = VX_def(T)
VY = VY_def(T)
VZ = VZ_def(T)
WX = WX_def(T)
WY = WY_def(T)
WZ = WZ_def(T)
V = (VX**2+VY**2+VZ**2)**0.5
W = (WX**2+WX**2+WZ**2)**0.5

ax1.plot(X, Y, Z)
ax2.plot(T, V)
ax2.set_xlabel('T')
ax2.set_ylabel('V')
ax3.plot(T, W)
ax3.set_xlabel('T')
ax3.set_ylabel('W')

# Movement of plate
XLD = LenghtOfPlate/2 * np.cos(Phi)
YLD = LenghtOfPlate/2 * np.sin(Phi)
ZLD = HeightOfStand

XRD = -XLD
YRD = -YLD
ZRD = HeightOfStand

XLU = XLD
YLU = YLD
ZLU = ZLD + WidthOfPlate

XRU = XRD
YRU = YRD
ZRU = ZRD + WidthOfPlate


#plottting
Point, = ax1.plot(X[0], Y[0], Z[0], marker='o', markersize='3')
axis = ax1.plot([0, 0], [0, 0], [-WidthOfPlate/2-2, -WidthOfPlate/2], color='#000', linewidth='2')
axis1 = ax1.plot([0, 0], [0, 0], [WidthOfPlate/2, WidthOfPlate/2+2], color='#000', linewidth='2')

DownLine, = ax1.plot([XLD[0], XRD[0]], [YLD[0], YRD[0]], [ZLD, ZRD], color="#000", linewidth='5')
LeftLine, = ax1.plot([XLD[0], XLU[0]], [YLD[0], YLU[0]], [ZLD, ZLU], color="#000", linewidth='5')
RightLine, = ax1.plot([XRD[0], XRU[0]], [YRD[0], YRU[0]], [ZRD, ZRU], color="#000", linewidth='5')
UpLine, = ax1.plot([XLU[0], XRU[0]], [YLU[0], YRU[0]], [ZLU, ZRU], color="#000", linewidth='5')
DiagLine, = ax1.plot([XRD[0], XLU[0]], [YRD[0], YLU[0]], [ZRD, ZLU], color="#000", linewidth='4', alpha=0.3)

def anima(i):
    Point.set_data_3d(X[i], Y[i], Z[i])
    DownLine.set_data_3d([XLD[i], XRD[i]], [YLD[i], YRD[i]], [ZLD, ZRD])
    LeftLine.set_data_3d([XLD[i], XLU[i]], [YLD[i], YLU[i]], [ZLD, ZLU])
    RightLine.set_data_3d([XRD[i], XRU[i]], [YRD[i], YRU[i]], [ZRD, ZRU])
    UpLine.set_data_3d([XLU[i], XRU[i]], [YLU[i], YRU[i]], [ZLU, ZRU])
    DiagLine.set_data_3d([XRD[i], XLU[i]], [YRD[i], YLU[i]], [ZRD, ZLU])
    return Point, DownLine, LeftLine, RightLine, UpLine, DiagLine


anim = FuncAnimation(fig, anima, frames=Frames, interval=1000/60)
plt.show()
