from cmath import exp
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from src.utils.euler import expmap2rotmat
import pickle
import matplotlib.pyplot as plt
import imageio
import os.path as osp
import glob
import copy
import matplotlib
import os
matplotlib.use('agg')

def video_euler(vid_path, out_path, n_prefix, prefix_len, pred_len, row=5, stride=3):
  expmap_data = pickle.load(open(out_path, 'rb'))
  denoise_process = expmap_data['denoise_process']
  sample = expmap_data['sample']

  parent, offset, rotInd, expmapInd = some_variables()

  for k in sample.keys():
    curr_sample = sample[k]
    curr_denoise = denoise_process[k]
    for idx in range(n_prefix):
      cnt = 0
      fig = plt.figure(figsize=(2*(pred_len+1)//stride, 2*row))
      axes = [[None for _ in range((pred_len)//stride+2)] for _ in range(row)]
      obes = [[None for _ in range((pred_len)//stride+2)] for _ in range(row)]
      
      for r in range(row):
        for c in range((pred_len)//stride+2):
          new_ax = fig.add_subplot(row, (pred_len)//stride+2, r*(2+(pred_len)//stride)+c+1, projection='3d')
          new_ob = Ax3DPose(new_ax, plane=False)
          axes[r][c] = new_ax
          obes[r][c] = new_ob

      for di, dp in enumerate(curr_denoise):                  
        final_prefix = []
        final_pred = []
        for r in range(row):
          xyz_prefix = []
          xyz_pred = []         

          for ii in range(prefix_len+pred_len):
            if ii < prefix_len:
              xyz = fkl(curr_sample[r, idx, ii], parent, offset, rotInd, expmapInd)
              xyz_prefix.append(xyz)
            else:
              xyz = fkl(dp[r, idx, ii-prefix_len], parent, offset, rotInd, expmapInd)
              xyz_pred.append(xyz)

          # === Plot ===
          xyz_prefix = np.asarray(xyz_prefix)
          xyz_pred = np.asarray(xyz_pred)

          if di == len(curr_denoise)-1:
            final_prefix.append(xyz_prefix)
            final_pred.append(xyz_pred)
          
          obes[r][0].update(xyz_prefix[-1], r=350)
          for ci, cc in enumerate(range(0, pred_len, stride)):
              obes[r][ci+1].update(xyz_pred[cc], lcolor="#4b3976", rcolor="#168c41", r=350)
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.001)
        plt.suptitle('[{}], Diffusion Step: {}'.format(k.upper(), di))
        plt.savefig('tmp_imgs/{}.png'.format(cnt))
        cnt += 1
      
      images = []
      for t in range(cnt):
        images.append(imageio.v2.imread('tmp_imgs/{}.png'.format(t)))
      for i in range(10):
        images.append(imageio.v2.imread('tmp_imgs/{}.png'.format(cnt-1)))
      imageio.v2.mimsave(osp.join(vid_path, 'denoise:{}:{}.gif'.format(k, idx)), images, duration=0.2)
      for fp in glob.glob('./tmp_imgs/*'):
        os.remove(fp)

      fig = plt.figure(figsize=(2*row, 2))
      axes = [None for _ in range(row)]
      obes = [None for _ in range(row)]
      
      for r in range(row):
        new_ax = fig.add_subplot(1, row, r+1, projection='3d')
        new_ob = Ax3DPose(new_ax, plane=False)
        axes[r] = new_ax
        obes[r] = new_ob

      for t in range(pred_len+prefix_len):
        for r in range(row):        
          if t < prefix_len:
            obes[r].update(final_prefix[r][t], r=350)
          else:
            obes[r].update(final_pred[r][t-prefix_len], lcolor="#4b3976", rcolor="#168c41", r=350)
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.001)
        plt.suptitle('[{}]'.format(k.upper()))
        plt.savefig('tmp_imgs/{}.png'.format(t))

      images = []
      for t in range(pred_len+prefix_len):
        images.append(imageio.v2.imread('tmp_imgs/{}.png'.format(t)))
      for i in range(25):
        images.append(imageio.v2.imread('tmp_imgs/{}.png'.format(pred_len+prefix_len-1)))
      imageio.v2.mimsave(osp.join(vid_path, 'result:{}:{}.gif'.format(k, idx)), images, duration=0.04)
      plt.close()
      plt.clf()
      for fp in glob.glob('./tmp_imgs/*'):
        os.remove(fp)


def figure_first_euler(fig_path, out_path, n_prefix, prefix_len, pred_len, row=8, stride=2):
  expmap_pose = pickle.load(open(out_path, 'rb'))

  parent, offset, rotInd, expmapInd = some_variables()
  
  fig = plt.figure(figsize=(2*pred_len//stride, (row+2)*1.5))
  axes = [[None for _ in range(pred_len//stride+2)] for _ in range(row+2)]
  obes = [[None for _ in range(pred_len//stride+2)] for _ in range(row+2)]
  for r in range(row+2):
    for c in range(pred_len//stride+2):
      new_ax = fig.add_subplot(row+2, pred_len//stride+2, r*(pred_len//stride+2)+c+1, projection='3d')
      new_ob = Ax3DPose(new_ax, plane=False)
      axes[r][c] = new_ax
      obes[r][c] = new_ob

  for k in expmap_pose['samples'].keys():
      for idx in range(n_prefix):  
        far_index = np.arange(0, row)
        for r in range(row+2):
          xyz_prefix = []
          xyz_pred = []         
          dist = []

          for ii in range(prefix_len+pred_len):
            if r > 1:
              xyz = fkl(expmap_pose['samples'][k][far_index[r-2], idx, ii], parent, offset, rotInd, expmapInd)
            elif r == 0:
              xyz = fkl(expmap_pose['gts'][k][idx, ii], parent, offset, rotInd, expmapInd)
            elif r == 1:
              xyz = fkl(expmap_pose['deters'][k][r, idx, ii], parent, offset, rotInd, expmapInd)
              
            dist.append(np.linalg.norm(xyz-fkl(expmap_pose['deters'][k][r, idx, ii], parent, offset, rotInd, expmapInd)))

            if ii < prefix_len:
              xyz_prefix.append(xyz)
            else:
              xyz_pred.append(xyz)

          # === Plot ===
          xyz_prefix = np.asarray(xyz_prefix)
          xyz_pred = np.asarray(xyz_pred)
          
          obes[r][0].update(xyz_prefix[-1], r=350)
          for cc in range(0, pred_len, stride):
            if r == 0:
              obes[r][cc//stride+1].update(xyz_pred[cc], r=350)
            elif r == 1:
              obes[r][cc//stride+1].update(xyz_pred[cc], lcolor="#4b3976", rcolor="#168c41", r=350)
            elif r > 1:
              obes[r][cc//stride+1].update(xyz_pred[cc], lcolor="#9b59b6", rcolor="#2ecc71", r=350)
            plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.001)
        plt.savefig(osp.join(fig_path, '{}_{}.png'.format(k, idx)))

def visualize_euler(vid_path, out_path, n_prefix, row, col, prefix_len, pred_len):
  expmap_pose = pickle.load(open(out_path, 'rb'))

  fig = plt.figure(figsize=(col*3, row*4))
  axes = [[None for _ in range(col)] for _ in range(row)]
  obes = [[None for _ in range(col)] for _ in range(row)]
  for r in range(row):
      for c in range(col):
          new_ax = fig.add_subplot(row, col, r+row*c+1, projection='3d')
          new_ob = Ax3DPose(new_ax)
          axes[r][c] = new_ax
          obes[r][c] = new_ob

  parent, offset, rotInd, expmapInd = some_variables()

  for k in expmap_pose.keys():
      for idx in range(n_prefix):

          xyz_prefix = np.zeros((row*col, prefix_len, 96))
          xyz_pred = np.zeros((row*col, pred_len, 96))

          for i in range(row*col):
              for ii in range(prefix_len+pred_len):
                  xyz = fkl(expmap_pose[k][i, idx, ii], parent, offset, rotInd, expmapInd)
                  if ii < prefix_len:
                      xyz_prefix[i, ii] = xyz
                  else:
                      xyz_pred[i, ii-prefix_len] = xyz

          # === Plot and animate ===

          cnt = 0
          for i in range(prefix_len):
              for r in range(row):
                  for c in range(col):
                      obes[r][c].update(xyz_prefix[r+row*c, i, :])
              plt.suptitle('{}:{}\n frame: {}'.format(k, idx, cnt), fontsize=13)
          
              plt.show(block=False)
              fig.canvas.draw()
              plt.pause(0.001)
              plt.savefig('tmp_imgs/{}.png'.format(cnt))
              cnt += 1

          for i in range(pred_len):
              for r in range(row):
                  for c in range(col):
                      obes[r][c].update(xyz_pred[r+row*c, i, :], lcolor="#9b59b6", rcolor="#2ecc71")
              plt.suptitle('{}:{}\n frame: {}'.format(k, idx, cnt), fontsize=13)
              
              plt.show(block=False)
              fig.canvas.draw()
              plt.savefig('tmp_imgs/{}.png'.format(cnt))
              plt.pause(0.001)
              cnt += 1

          images = []
          for t in range(prefix_len+pred_len):
            images.append(imageio.v2.imread('tmp_imgs/{}.png'.format(t)))
          for t in range(12):
            images.append(images[-1])
            
          imageio.v2.mimsave(osp.join(vid_path, '{}:{}.gif'.format(k, idx)), images, duration=0.04)
          for fp in glob.glob('./tmp_imgs/*'):
            os.remove(fp)


class Ax3DPose(object):
  def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c", plane=True):
    """
    Create a 3d pose visualizer that can be updated with new poses.

    Args
      ax: 3d axis to plot the 3d pose on
      lcolor: String. Colour for the left part of the body
      rcolor: String. Colour for the right part of the body
    """

    # Start and endpoints of our representation
    self.I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1
    self.J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1
    # Left / right indicator
    self.LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    self.ax = ax

    vals = np.zeros((32, 3))

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
    if not plane:
      ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
      
    ax.w_xaxis.line.set_lw(0.)
    ax.w_yaxis.line.set_lw(0.)
    ax.w_zaxis.line.set_lw(0.)


  def update(self, channels, lcolor="#3498db", rcolor="#e74c3c", r=750):
    """
    Update the plotted 3d pose.

    Args
      channels: 96-dim long np array. The pose to plot.
      lcolor: String. Colour for the left part of the body.
      rcolor: String. Colour for the right part of the body.
    Returns
      Nothing. Simply updates the axis with the new pose.
    """

    assert channels.size == 96, "channels should have 96 entries, it has %d instead" % channels.size
    vals = np.reshape( channels, (32, -1) )

    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots[i][0].set_xdata(x)
      self.plots[i][0].set_ydata(y)
      self.plots[i][0].set_3d_properties(z)
      self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)

    self.ax.set_xlim3d([-r, r])
    self.ax.set_zlim3d([-r, r])
    self.ax.set_ylim3d([-500, 500])

    self.ax.set_aspect('auto')



def fkl( angles, parent, offset, rotInd, expmapInd ):
  """
  Convert joint angles and bone lenghts into the 3d points of a person.
  Based on expmap2xyz.m, available at
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m

  Args
    angles: 99-long vector with 3d position and 3d joint angles in expmap format
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
  Returns
    xyz: 32x3 3d points that represent a person in 3d space
  """

  assert len(angles) == 99

  # Structure that indicates parents for each joint
  njoints   = 32
  xyzStruct = [dict() for x in range(njoints)]

  for i in np.arange( njoints ):

    if not rotInd[i] : # If the list is empty
      xangle, yangle, zangle = 0, 0, 0
    else:
      xangle = angles[ rotInd[i][0]-1 ]
      yangle = angles[ rotInd[i][1]-1 ]
      zangle = angles[ rotInd[i][2]-1 ]

    r = angles[ expmapInd[i] ]

    thisRotation = expmap2rotmat(r)
    thisPosition = np.array([xangle, yangle, zangle])

    if parent[i] == -1: # Root node
      xyzStruct[i]['rotation'] = thisRotation
      xyzStruct[i]['xyz']      = np.reshape(offset[i,:], (1,3)) + thisPosition
    else:
      xyzStruct[i]['xyz'] = (offset[i,:] + thisPosition).dot( xyzStruct[ parent[i] ]['rotation'] ) + xyzStruct[ parent[i] ]['xyz']
      xyzStruct[i]['rotation'] = thisRotation.dot( xyzStruct[ parent[i] ]['rotation'] )

  xyz = [xyzStruct[i]['xyz'] for i in range(njoints)]
  xyz = np.array( xyz ).squeeze()
  xyz = xyz[:,[0,2,1]]
  # xyz = xyz[:,[2,0,1]]


  return np.reshape( xyz, [-1] )


def some_variables():
  """
  We define some variables that are useful to run the kinematic tree

  Args
    None
  Returns
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
  """

  parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9,10, 1,12,13,14,15,13,
                    17,18,19,20,21,20,23,13,25,26,27,28,29,28,31])-1

  offset = np.array([0.000000,0.000000,0.000000,-132.948591,0.000000,0.000000,0.000000,-442.894612,0.000000,0.000000,-454.206447,0.000000,0.000000,0.000000,162.767078,0.000000,0.000000,74.999437,132.948826,0.000000,0.000000,0.000000,-442.894413,0.000000,0.000000,-454.206590,0.000000,0.000000,0.000000,162.767426,0.000000,0.000000,74.999948,0.000000,0.100000,0.000000,0.000000,233.383263,0.000000,0.000000,257.077681,0.000000,0.000000,121.134938,0.000000,0.000000,115.002227,0.000000,0.000000,257.077681,0.000000,0.000000,151.034226,0.000000,0.000000,278.882773,0.000000,0.000000,251.733451,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999627,0.000000,100.000188,0.000000,0.000000,0.000000,0.000000,0.000000,257.077681,0.000000,0.000000,151.031437,0.000000,0.000000,278.892924,0.000000,0.000000,251.728680,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999888,0.000000,137.499922,0.000000,0.000000,0.000000,0.000000])
  offset = offset.reshape(-1,3)

  rotInd = [[5, 6, 4],
            [8, 9, 7],
            [11, 12, 10],
            [14, 15, 13],
            [17, 18, 16],
            [],
            [20, 21, 19],
            [23, 24, 22],
            [26, 27, 25],
            [29, 30, 28],
            [],
            [32, 33, 31],
            [35, 36, 34],
            [38, 39, 37],
            [41, 42, 40],
            [],
            [44, 45, 43],
            [47, 48, 46],
            [50, 51, 49],
            [53, 54, 52],
            [56, 57, 55],
            [],
            [59, 60, 58],
            [],
            [62, 63, 61],
            [65, 66, 64],
            [68, 69, 67],
            [71, 72, 70],
            [74, 75, 73],
            [],
            [77, 78, 76],
            []]

  expmapInd = np.split(np.arange(4, 100)-1,32)

  return parent, offset, rotInd, expmapInd

