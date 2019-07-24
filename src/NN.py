''' On NERSC
    module load python/3.6-anaconda-4.4 
    salloc -N 1 -q interactive -C haswell -t 15:00
    srun -n 4 python NN.py 
'''
import tensorflow as tf                    # NN stuff
import numpy as np                         # numerical python 
import os

# imac branch
class preprocess(object):
    def __init__(self, datai, axfit=None):
        np.random.seed(12345)
        data   = datai.copy()      # copy input data
        np.random.shuffle(data)    # shuffle input data
        self.X = data['features']
        self.Y = data['label'][:,np.newaxis]
        self.P = data['hpind']
        self.W = data['fracgood'][:, np.newaxis]#**2
        if len(self.X.shape) == 1:
            self.X = self.X[:,np.newaxis]
        if axfit is not None:
            self.X = self.X[:, axfit]
            
        #self.Xs = None
        #self.Ys = None
        
    
class Netregression(object):
    """
        class for a general regression
    """
    def __init__(self, train, valid, test, axfit=None):
        # data (Traind, Testd) should have following attrs,
        #
        # features i.e.  X
        # label    i.e. Y = f(X) + noise ?
        # hpix     i.e. healpix indices to keep track of data
        # fracgood i.e. weight associated to each datapoint eg. pixel
        #
        # train
        self.train = preprocess(train, axfit)
        # test
        self.test  = preprocess(test,  axfit)
        # validation
        self.valid = preprocess(valid, axfit)
        
        #
        # one feature or more
        self.nfeatures = self.train.X.shape[1]
        
        
    def train_evaluate(self, learning_rate=0.001,
                       batchsize=100, nepoch=10, nchain=5,
                      Units=[10,10], tol=1.e-5, scale=0.0,
                       actfunc=tf.nn.relu, patience=10):
        #
        #from tensorflow.python.framework import ops

        nfeature = self.nfeatures
        #print('nfeature : ', nfeature)
        nclass   = 1            # for classification, you will have to change this
        #
        
        #
        #
        train = self.train
        valid = self.valid
        test  = self.test
        
        train_size = train.X.shape[0]
        #
        # using training label/feature mean and std
        # to normalize training/testing label/feature
        meanX      = np.mean(train.X, axis=0)
        stdX       = np.std(train.X, axis=0)
        meanY      = np.mean(train.Y, axis=0)
        stdY       = np.std(train.Y, axis=0)
        #
        # convert 0 stds to 1.0s
        assert np.all(stdX != 0.0)
        assert (stdY != 0.0)        
        #stdX[stdX==0.0] = 1.0

        
        self.Xstat = (meanX, stdX)
        self.Ystat = (meanY, stdY)
        
        train.X = (train.X - meanX) / stdX
        train.Y = (train.Y - meanY) / stdY
        test.X  = (test.X - meanX) / stdX
        test.Y  = (test.Y - meanY) / stdY
        valid.X = (valid.X - meanX) / stdX
        valid.Y = (valid.Y - meanY) / stdY
        #
        # compute the number of training updates
        if np.mod(train_size, batchsize) == 0:
            nep = (train_size // batchsize)
        else:
            nep = (train_size // batchsize) + 1
        #
        # initialize empty lists to store MSE 
        # and prediction at each training epoch for each chain
        self.epoch_MSEs = []
        self.chain_y     = []
        global_seed = 12345
        np.random.seed(global_seed)
        seeds = np.random.randint(0, 4294967295, size=nchain)
        for ii in range(nchain): # loop on chains
            tf.set_random_seed(seeds[ii]) # set the seed
            # set up the model x [input] -> y [output]
            x   = tf.placeholder(tf.float32, [None, nfeature])
            #
            # linear, one hidden layer or 2 hidden layers 
            # need to modify this if more layers are desired
            # tf.layers.dense works like f(aX+b) where f is activation
            if (len(Units) == 1) and Units[0]==0:    # linear
                kernel_init = tf.random_normal_initializer(stddev=np.sqrt(1./(nfeature)), seed=seeds[ii])
                y  = tf.layers.dense(x, units=nclass, activation=None, kernel_initializer=kernel_init,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=scale))
            elif len(Units) == 1 and Units[0]!=0: # 1 hidden layer
                kernel_init0 = tf.random_normal_initializer(stddev=np.sqrt(1./(nfeature)), seed=seeds[ii])
                kernel_init  = tf.random_normal_initializer(stddev=np.sqrt(2./(Units[0])), seed=seeds[ii]) 
                y0 = tf.layers.dense(x,  units=Units[0], activation=actfunc, kernel_initializer=kernel_init0,
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=scale)) 
                y  = tf.layers.dense(y0, units=nclass, activation=None, kernel_initializer=kernel_init,
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=scale)) 
            elif len(Units) == 2:                                    # 2 hidden layers
                kernel_init0 = tf.random_normal_initializer(stddev=np.sqrt(1./(nfeature)), seed=seeds[ii])
                kernel_init1 = tf.random_normal_initializer(stddev=np.sqrt(2./(Units[0])), seed=seeds[ii]) 
                kernel_init  = tf.random_normal_initializer(stddev=np.sqrt(2./(Units[1])), seed=seeds[ii]) 
                y0 = tf.layers.dense(x,  units=Units[0], activation=actfunc, kernel_initializer=kernel_init0,
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=scale))
                y1 = tf.layers.dense(y0, units=Units[1], activation=actfunc, kernel_initializer=kernel_init1,
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=scale))
                y  = tf.layers.dense(y1, units=nclass,   activation=None,  kernel_initializer=kernel_init,
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=scale))
            elif len(Units) == 3:
                kernel_init0 = tf.random_normal_initializer(stddev=np.sqrt(1./(nfeature)), seed=seeds[ii])
                kernel_init1 = tf.random_normal_initializer(stddev=np.sqrt(2./(Units[0])), seed=seeds[ii]) 
                kernel_init2 = tf.random_normal_initializer(stddev=np.sqrt(2./(Units[1])), seed=seeds[ii]) 
                kernel_init  = tf.random_normal_initializer(stddev=np.sqrt(2./(Units[2])), seed=seeds[ii]) 
                y0 = tf.layers.dense(x,  units=Units[0], activation=actfunc, kernel_initializer=kernel_init0,
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=scale))
                y1 = tf.layers.dense(y0, units=Units[1], activation=actfunc,  kernel_initializer=kernel_init1,
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=scale))
                y2 = tf.layers.dense(y1, units=Units[2], activation=actfunc,  kernel_initializer=kernel_init2,
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=scale))
                y  = tf.layers.dense(y2, units=nclass,   activation=None,  kernel_initializer=kernel_init, 
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=scale))
            elif len(Units) == 4:
                kernel_init0 = tf.random_normal_initializer(stddev=np.sqrt(1./(nfeature)), seed=seeds[ii])
                kernel_init1 = tf.random_normal_initializer(stddev=np.sqrt(2./(Units[0])), seed=seeds[ii]) 
                kernel_init2 = tf.random_normal_initializer(stddev=np.sqrt(2./(Units[1])), seed=seeds[ii]) 
                kernel_init3 = tf.random_normal_initializer(stddev=np.sqrt(2./(Units[2])), seed=seeds[ii]) 
                kernel_init  = tf.random_normal_initializer(stddev=np.sqrt(2./(Units[3])), seed=seeds[ii]) 
                y0 = tf.layers.dense(x,  units=Units[0], activation=actfunc, kernel_initializer=kernel_init0,
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=scale))
                y1 = tf.layers.dense(y0, units=Units[1], activation=actfunc, kernel_initializer=kernel_init1,
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=scale))
                y2 = tf.layers.dense(y1, units=Units[2], activation=actfunc, kernel_initializer=kernel_init2,
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=scale))
                y3 = tf.layers.dense(y2, units=Units[3], activation=actfunc, kernel_initializer=kernel_init3,
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=scale))
                y  = tf.layers.dense(y3, units=nclass,   activation=None, kernel_initializer=kernel_init,
                                    kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=scale))
            else:
                raise ValueError('Units should be either None, [M], [M,N] ...')
            #
            # placeholders for the input errorbar and label
            y_  = tf.placeholder(tf.float32, [None, nclass])
            w   = tf.placeholder(tf.float32, [None, nclass])

            #
            # objective function
            mse = tf.losses.mean_squared_error(y_, y, weights=w)
            l2_loss = tf.losses.get_regularization_loss()
            mse_w_l2 = mse + l2_loss
            #
            # see https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer   = tf.train.AdamOptimizer(learning_rate)
            train_step  = optimizer.minimize(mse_w_l2, global_step=global_step)            
            #print('chain ',ii)
            mse_min = 10000000.
            last_improvement = 0
            mse_list = []
            # 
            # initialize the NN
            #config = tf.ConfigProto()
            #config.intra_op_parallelism_threads = 1
            #config.inter_op_parallelism_threads = 1
            #sess = tf.InteractiveSession(config=config)
            sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()            
            for i in range(nepoch+1): # loop on training epochs
                #
                # save train & test MSE at each epoch
                train_loss = mse.eval(feed_dict={x:train.X, y_:train.Y, w:train.W}) # June 7th 2:20pm - change self.train.X to train.X
                valid_loss = mse.eval(feed_dict={x:valid.X, y_:valid.Y, w:valid.W})                                
                #test_loss = mse.eval(feed_dict={x:test.X, y_:test.Y, w:test.W}) # to evaluate test MSE
                mse_list.append([i, train_loss, valid_loss])
                #mse_list.append([i, train_loss, valid_loss, test_loss])  # to save test MSE
                #
                #  Early Stopping
                if (np.abs(valid_loss/mse_min -1.0) > tol) and (valid_loss < mse_min):
                    mse_min = valid_loss
                    last_improvement = 0
                else:
                    last_improvement += 1
                
                if last_improvement > patience:
                    #print("No improvement found during the {} last iterations at {}, stopping optimization!!".format(patience, i))
                    print("stopping at {}".format(i))
                    break # stop training by early stopping
                for k in range(nep): # loop on training unpdates
                    ji = k*batchsize
                    jj = np.minimum((k+1)*batchsize, train_size)                    
                    batch_xs, batch_ys, batch_ws = train.X[ji:jj], train.Y[ji:jj], train.W[ji:jj]   # use up to the last element
                    # train NN at each update
                    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys, w:batch_ws})
            #
            
            # save the final test MSE and prediction for each chain 
            y_mse, y_pred  = sess.run((mse,y),feed_dict={x: test.X, y_: test.Y, w:test.W})
            self.chain_y.append([ii, y_pred])
            self.epoch_MSEs.append([ii, y_mse, np.array(mse_list)])
            sess.close()
            tf.reset_default_graph()
        # baseline model is the average of training label
        # baseline mse
        baselineY  = np.mean(train.Y)
        assert np.abs(baselineY) < 1.e-6, 'check normalization!'
        baseline_testmse  = np.mean(test.W  * test.Y**2)
        baseline_validmse = np.mean(valid.W * valid.Y**2)
        baseline_trainmse = np.mean(train.W * train.Y**2)
        #    
        self.optionsdic = {}
        self.optionsdic['baselineMSE']   = (baseline_trainmse, baseline_validmse, baseline_testmse)
        self.optionsdic['learning_rate'] = learning_rate
        self.optionsdic['batchsize']     = batchsize
        self.optionsdic['nepoch']        = nepoch
        self.optionsdic['nchain']        = nchain
        self.optionsdic['Units']         = Units
        self.optionsdic['scale']         = scale
        self.optionsdic['stats']         = {'xstat':self.Xstat, 'ystat':self.Ystat}
            
            
    def savez(self, indir='./', name='regression_2hl_5chain_10epoch'):
        output = {}
        output['train']      = self.train.P, self.train.X, self.train.Y, self.train.W 
        output['test']       = self.test.P, self.test.X, self.test.Y, self.test.W 
        output['valid']      = self.valid.P, self.valid.X, self.valid.Y, self.valid.W         
        output['epoch_MSEs'] = self.epoch_MSEs
        output['chain_y']    = self.chain_y
        output['options']    = self.optionsdic
        if indir[-1] != '/':
            indir += '/'
        if not os.path.exists(indir):
            os.makedirs(indir)
        #if not os.path.isfile(indir+name+'.npz'):   # write w a new name
        np.savez(indir+name, output)
        #else:
        #    print("there is already a file!")
        #    name = name+''.join(time.asctime().split(' '))
        #    np.savez(indir+name, output)
        print('output is saved as {} under {}'.format(name, indir))

def run_nchainlearning(indir, *arrays, **options):
    n_arrays = len(arrays)
    if n_arrays != 3:
        raise ValueError("Three arrays for train, validation  and test are required")
    net = Netregression(*arrays)
    net.train_evaluate(**options) #learning_rate=0.01, batchsize=100, nepoch=10, nchain=5
    #
    batchsize = options.pop('batchsize', 100)
    nepoch = options.pop('nepoch', 10)
    nchain = options.pop('nchain', 5)
    Units  = options.pop('Units', [10,10])
    Lrate  = options.pop('learning_rate', 0.001)
    Scale  = options.pop('scale', 0)
    units  = ''.join([str(l) for l in Units])
    ouname = 'reg-nepoch'+str(nepoch)+'-nchain'+str(nchain)
    ouname += '-batchsize'+str(batchsize)+'units'+units
    ouname += '-Lrate'+str(Lrate)+'-l2scale'+str(Scale)
    #
    net.savez(indir=indir, name=ouname)        
    
def read_NNfolds(files):
    """
        Reading different folds results, 
        `files` is a list holding the paths to different folds
        uses the mean and std of training label
        to scale back the prediction 
    """
    p_true = []
    x_true = []
    y_true = []
    y_pred = []
    y_base = []
    weights = []
    for j,file_i in enumerate(files):
        d = np.load(file_i)              # read the file
        out = d['arr_0'].item()          #
        meanY, stdY = out['options']['stats']['ystat']
        meanX, stdX = out['options']['stats']['xstat']
        p_true.append(out['test'][0])    # read the true pixel id
        x_true.append(stdX * out['test'][1] + meanX)    # features ie. X
        y_true.append(stdY * out['test'][2].squeeze() + meanY)  # true label ie. Y
        weights.append(out['test'][3].squeeze()) # weights
        #
        # loop over predictions from chains 
        # take the average of them
        # and use the mean & std of the training label to scale back
        y_avg = []
        for i in range(len(out['chain_y'])):
            y_avg.append(out['chain_y'][i][1].squeeze().tolist())    
        #print(np.mean(out['train'][2]))
        #print(meanY, stdY)
        # mean of training label as baseline model
        y_base.append(np.ones(out['test'][2].shape[0])*meanY)        
        y_pred.append(stdY*np.mean(np.array(y_avg), axis=0) + meanY)
    
    # combine different folds 
    Ptrue = np.concatenate(p_true)
    Xtrue = np.concatenate(x_true)
    Ytrue = np.concatenate(y_true)
    Ypred = np.concatenate(y_pred)
    Ybase = np.concatenate(y_base)
    Weights = np.concatenate(weights)
    #print(Xtrue.shape, Ytrue.shape, Ypred.shape, Ybase.shape, Weights.shape)
    return Ptrue, Xtrue, Ytrue, Ypred, Ybase, Weights
    
if __name__ == '__main__':    
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    #
    #
    if rank == 0:
        from argparse import ArgumentParser
        ap = ArgumentParser(description='Neural Net regression')
        ap.add_argument('--path',   default='/global/cscratch1/sd/mehdi/dr5_anand/eboss/')
        ap.add_argument('--input',  default='test_train_eboss_dr5-masked.npy')
        ap.add_argument('--output', default='/global/cscratch1/sd/mehdi/dr5_anand/eboss/regression/')
        ap.add_argument('--nchain', type=int, default=10)
        ap.add_argument('--nepoch', type=int, default=1000)
        ap.add_argument('--batchsize', type=int, default=8000)
        ap.add_argument('--units', nargs='*', type=int, default=[10,10])
        ap.add_argument('--learning_rate', type=float, default=0.01)
        ap.add_argument('--scale', type=float, default=0.0)
        ns = ap.parse_args()
        #
        #
        data   = np.load(ns.path+ns.input).item()
        config = {'nchain':ns.nchain,
                  'nepoch':ns.nepoch,
                  'batchsize':ns.batchsize,
                  'Units':ns.units,
                  'learning_rate':ns.learning_rate,
                  'scale':ns.scale}
        oupath = ns.output
    else:
        oupath = None
        data   = None
        config = None
    #
    # bcast
    data   = comm.bcast(data, root=0)
    config = comm.bcast(config, root=0)
    oupath = comm.bcast(oupath, root=0)
    
    #
    # run
    if rank == 0:
        print("bcast finished")
    if rank in [0, 1, 2, 3, 4]:
        print("config on rank %d is: "%rank, config)        
        fold = 'fold'+str(rank)
        print(fold, ' is being processed')
        run_nchainlearning(oupath+fold+'/',
                       data['train'][fold],
                       data['validation'][fold], 
                       data['test'][fold],
                      **config)
