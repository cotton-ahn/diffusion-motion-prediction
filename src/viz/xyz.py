import numpy as np
import pickle
import matplotlib.pyplot as plt
import imageio
import os
import os.path as osp

def visualize_xyz(vid_path, out_path, n_prefix, row, col, prefix_len, pred_len):
    xyz_pose = pickle.load(open(out_path, 'rb')) # n_prefix, n_samples, len, 16, 3

    fig = plt.figure(figsize=(col*3, row*4))
    axes = [[None for _ in range(col)] for _ in range(row)]
    obes = [[None for _ in range(col)] for _ in range(row)]

    for r in range(row):
        for c in range(col):
            new_ax = fig.add_subplot(row, col, r+row*c+1, projection='3d')
            new_ob = Ax3DPose(new_ax)
            axes[r][c] = new_ax
            obes[r][c] = new_ob

    for idx in range(n_prefix):
        curr_xyz = xyz_pose[idx]  # 50 125 16 3

        # === Plot and animate ===
        plt.suptitle('{}th prefix'.format(idx), fontsize=13)
        cnt = 0
        for i in range(prefix_len):
            for r in range(row):
                for c in range(col):
                    obes[r][c].update(curr_xyz[r+row*c, i, :])
            plt.show(block=False)
            fig.canvas.draw()
            plt.pause(0.001)
            plt.savefig('tmp_imgs/{}.png'.format(cnt))
            cnt += 1

        for i in range(pred_len):
            for r in range(row):
                for c in range(col):
                    obes[r][c].update(curr_xyz[r+row*c, prefix_len+i, :], lcolor="#9b59b6", rcolor="#2ecc71" )
            plt.show(block=False)
            fig.canvas.draw()
            plt.savefig('tmp_imgs/{}.png'.format(cnt))
            plt.pause(0.001)
            cnt += 1

        images = []
        for t in range(prefix_len+pred_len):
            images.append(imageio.v2.imread('tmp_imgs/{}.png'.format(t)))
        imageio.v2.mimsave(osp.join(vid_path, '{}.gif'.format(idx)), images, duration=0.02)




class Ax3DPose(object):
  def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Create a 3d pose visualizer that can be updated with new poses.

    Args
      ax: 3d axis to plot the 3d pose on
      lcolor: String. Colour for the left part of the body
      rcolor: String. Colour for the right part of the body
    """

    # Start and endpoints of our representation
    self.I = np.array([6, 0, 1, 6, 3, 4, 6, 7, 8, 7, 10, 11, 7, 13, 14])
    self.J = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10,11, 12,13, 14, 15])
    # Left / right indicator
    self.LR  = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    self.ax = ax

    vals = np.zeros((16, 3))

    # Make connection matrix
    self.plots = []
    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots.append(self.ax.plot(x, y, z, lw=2, c=lcolor if self.LR[i] else rcolor))

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_xaxis.line.set_lw(0.)
    ax.w_yaxis.line.set_lw(0.)
    ax.w_zaxis.line.set_lw(0.)


  def update(self, vals, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Update the plotted 3d pose.

    Args
      channels: 96-dim long np array. The pose to plot.
      lcolor: String. Colour for the left part of the body.
      rcolor: String. Colour for the right part of the body.
    Returns
      Nothing. Simply updates the axis with the new pose.
    """

    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots[i][0].set_xdata(x)
      self.plots[i][0].set_ydata(y)
      self.plots[i][0].set_3d_properties(z)
      self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)

    self.ax.set_xlim3d([-0.75, 0.75])
    self.ax.set_ylim3d([-0.75, 0.75])
    self.ax.set_zlim3d([-1.0, 1.0])
    
    self.ax.set_aspect('auto')

