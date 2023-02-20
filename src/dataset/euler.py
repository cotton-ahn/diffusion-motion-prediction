import numpy as np
from src.utils.euler import unNormalizeData, rotmat2euler, expmap2rotmat

def get_srnn_gts( actions, test_set, data_mean, data_std, dim_to_ignore, pose_dim, prefix_len, pred_len, one_hot=False, to_euler=True):
  """
  Below code borrowed from Martinez et al: https://github.com/una-dinosauria/human-motion-prediction

  Get the ground truths for srnn's sequences, and convert to Euler angles.
  (the error is always computed in Euler angles).

  Args
    actions: a list of actions to get ground truths for.
    model: training model we are using (we only use the "get_batch" method).
    test_set: dictionary with normalized training data.
    data_mean: d-long vector with the mean of the training data.
    data_std: d-long vector with the standard deviation of the training data.
    dim_to_ignore: dimensions that we are not using to train/predict.
    one_hot: whether the data comes with one-hot encoding indicating action.
    to_euler: whether to convert the angles to Euler format or keep thm in exponential map

  Returns
    srnn_gts_euler: a dictionary where the keys are actions, and the values
      are the ground_truth, denormalized expected outputs of srnns's seeds.
  """
  srnn_gts_euler = {}

  for action in actions:

    srnn_gt_euler = []
    _, srnn_expmap = get_batch_srnn( test_set, pose_dim, prefix_len, pred_len, action )

    # expmap -> rotmat -> euler
    for i in np.arange( srnn_expmap.shape[0] ):
      denormed = unNormalizeData(srnn_expmap[i,:,:], data_mean, data_std, dim_to_ignore, actions, one_hot )

      if to_euler:
        for j in np.arange( denormed.shape[0] ):
          for k in np.arange(3,97,3):
            denormed[j,k:k+3] = rotmat2euler( expmap2rotmat( denormed[j,k:k+3] ))

      srnn_gt_euler.append( denormed )

    # Put back in the dictionary
    srnn_gts_euler[action] = srnn_gt_euler

  return srnn_gts_euler


def get_batch(data, batch_size, pose_dim, prefix_len, pred_len, actions ):
        """
        Below code borrowed from Martinez et al: https://github.com/una-dinosauria/human-motion-prediction

        Get a random batch of data from the specified bucket, prepare for step.

        Args
        data: a list of sequences of size n-by-d to fit the model to.
        Returns
        The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
        the constructed batches have the proper format to call step(...) later.
        """

        # Select entries at random
        all_keys    = list(data.keys())
        chosen_keys = np.random.choice( len(all_keys), batch_size )

        # How many frames in total do we need?
        total_frames = prefix_len + pred_len
        
        prefix = np.zeros((batch_size, prefix_len, pose_dim), dtype=float)
        gt_out = np.zeros((batch_size, pred_len, pose_dim), dtype=float)
        valid_data = np.zeros((batch_size), dtype=bool)
        for i in range( batch_size ):
            the_key = all_keys[ chosen_keys[i] ]

            # Get the number of frames
            n, _ = data[ the_key ].shape

            action_name = the_key[1]
            if action_name in actions:
                valid_data[i] = True

                # Sample somewherein the middle
                idx = np.random.randint( 16, n-total_frames )

                # Select the data around the sampled points
                data_sel = data[ the_key ][idx:idx+total_frames ,:]

                # Add the data
                prefix[i,:,:] = data_sel[:prefix_len, :]
                gt_out[i,:,:] = data_sel[prefix_len:, :]
            else:
                valid_data[i] = False

        prefix = prefix[valid_data] 
        gt_out = gt_out[valid_data]

        return prefix, gt_out


def find_indices_srnn( data, action ):
        """
        Below code borrowed from Martinez et al: https://github.com/una-dinosauria/human-motion-prediction

        Find the same action indices as in SRNN.
        See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
        """

        # Used a fixed dummy seed, following
        # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
        SEED = 1234567890
        rng = np.random.RandomState( SEED )

        subject = 5 # fixed for testing
        subaction1 = 1
        subaction2 = 2

        T1 = data[ (subject, action, subaction1, 'even') ].shape[0]
        T2 = data[ (subject, action, subaction2, 'even') ].shape[0]
        prefix, suffix = 50, 100

        idx = []
        idx.append( rng.randint( 16,T1-prefix-suffix ))
        idx.append( rng.randint( 16,T2-prefix-suffix ))
        idx.append( rng.randint( 16,T1-prefix-suffix ))
        idx.append( rng.randint( 16,T2-prefix-suffix ))
        idx.append( rng.randint( 16,T1-prefix-suffix ))
        idx.append( rng.randint( 16,T2-prefix-suffix ))
        idx.append( rng.randint( 16,T1-prefix-suffix ))
        idx.append( rng.randint( 16,T2-prefix-suffix ))
        return idx


def get_batch_srnn( data, pose_dim, prefix_len, pred_len, action ):
    """
    Below code borrowed from Martinez et al: https://github.com/una-dinosauria/human-motion-prediction

    Get a random batch of data from the specified bucket, prepare for step.

    Args
    data: dictionary with k:v, k=((subject, action, subsequence, 'even')),
        v=nxd matrix with a sequence of poses
    action: the action to load data from
    Returns
    The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
    the constructed batches have the proper format to call step(...) later.
    """

    actions = ["directions", "discussion", "eating", "greeting", "phoning",
                "posing", "purchases", "sitting", "sittingdown", "smoking",
                "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

    if not action in actions:
        raise ValueError("Unrecognized action {0}".format(action))

    frames = {}
    frames[ action ] = find_indices_srnn( data, action )

    batch_size = 8 # we always evaluate 8 seeds
    subject    = 5 # we always evaluate on subject 5

    seeds = [( action, (i%2)+1, frames[action][i] ) for i in range(batch_size)]

    prefix = np.zeros( (batch_size, prefix_len, pose_dim), dtype=float )
    gt_out = np.zeros( (batch_size, pred_len, pose_dim), dtype=float )

    # Compute the number of frames needed
    total_frames = prefix_len + pred_len
    
    # Reproducing SRNN's sequence subsequence selection as done in
    # https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
    for i in range( batch_size ):
        _, subsequence, idx = seeds[i]
        idx = idx + 50

        data_sel = data[ (subject, action, subsequence, 'even') ]

        data_sel = data_sel[(idx-prefix_len):(idx+pred_len) ,:]

        prefix[i, :, :] = data_sel[:prefix_len, :]
        gt_out[i, :, :] = data_sel[prefix_len:, :]

    return prefix, gt_out


