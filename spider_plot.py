import numpy as np
import matplotlib.pyplot as plt
import math

plt.rcParams.update({'font.size': 16})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


def _invert(x, limits):
    """inverts a value x on a scale from limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])


def _scale_data(data, ranges):
    """scales data[1:] to ranges[0], inverts if the scale is reversed"""
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = data[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d - y1) / (y2 - y1)
                     * (x2 - x1) + x1)
    return sdata


class ComplexRadar:
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=3):
        angles = np.arange(0, 360, 360. / len(variables))

        axes = [fig.add_axes([0.5, 0.5, 1, 1], polar=True,
                             label="axes{}".format(i))
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles,
                                         labels=variables)
        [txt.set_rotation(angle - 90) for txt, angle
         in zip(text, angles)]
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i],
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(math.floor(round(x, 2)))
                         for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1]  # hack to invert grid
                # gridlabels aren't reversed
            gridlabel[0] = ""  # clean up origin
            ax.set_rgrids(grid, labels=gridlabel,
                          angle=angles[i])
            # ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]

    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)


# parameters
variables = (
    "             Area \n              Coverage \n             (max: 360\N{DEGREE SIGN})", "User orientation \n (max: 360\N{DEGREE SIGN})\n \n \n",
    "Multi User \n \n ",
    "Micro            \n    activities                ", "\n \n \n \n Macro     \n activities    \n",
    "\n \n \n \n Monitoring paradigm \n(opportunistic: 1, continuous: 2)")
ranges = [(360, 0), (360, 0), (10, 0),
          (5, 0), (8, 0), (2, 0)]
# data
radhaar = (120, 120, 1,
           0, 5, 1)
mactivity = (120, 120, 1,
             0, 5, 2)
yu = (120, 120, 1,
      0, 7, 1)
mmpoint = (120, 120, 1,
           4, 5, 1)
wu = (120, 120, 1,
      5, 7, 1)
mmsense = (360, 360, 5,
           2, 1, 2)
mmgaitnet = (360, 360, 5,
             0, 1, 1)
ourmethod = (360, 360, 10,
             5, 8, 2)

# plotting
fig1 = plt.figure(figsize=(10, 6))
radar = ComplexRadar(fig1, variables, ranges)
radar.plot(radhaar, "-", lw=2, color="b", alpha=0.9, marker = 'o', ms = 9, label="RadHAR [12]")
radar.fill(radhaar, alpha=0.1)
radar.plot(mactivity, "-", lw=2, color="r", alpha=0.9, marker = 's', ms = 9, label="m-Activity")
radar.fill(mactivity, alpha=0.2)
radar.plot(yu, "-", lw=2, color="m", alpha=0.9, marker = '+', ms = 9, label="DVCNN")
radar.fill(yu, alpha=0.2)
radar.plot(mmpoint, "-", lw=2, color="c", alpha=0.9, marker = 'x', ms = 9, label="MMPoint-GNN")
radar.fill(mmpoint, alpha=0.1)
radar.plot(wu, "-", lw=2, color="y", alpha=0.9, marker = 'h', ms = 9, label="3D Radar")
radar.fill(wu, alpha=0.1)
radar.plot(mmsense, "-", lw=2, color='mediumseagreen', alpha=0.9, marker = 'v', ms = 9, label="mmSense")
radar.fill(mmsense, alpha=0.1)
radar.plot(mmgaitnet, "-", lw=2, color='aquamarine', alpha=0.9, marker = 'D', ms = 9, label="mmGaitNet")
radar.fill(mmgaitnet, alpha=0.1)
radar.plot(ourmethod, "-", lw=2, color='0', alpha=0.9, marker = '*', ms = 9, label="MARS")
radar.fill(ourmethod, alpha=0.1)
radar.ax.legend(bbox_to_anchor=(1.1, 1.1), loc='upper left', borderaxespad=0)
plt.savefig('sample_dimensions.pdf',bbox_inches='tight')
plt.show()
