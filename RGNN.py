import re
import time
import numpy as np
import tensorflow as tf


class RGNN(): #2 layers
    """A class of relational graph neural network."""
    
    def __init__(self, args):
        self.args = dict(args._get_kwargs())
        for key, value in self.args.items():
            if type(value) == str:
                exec('self.{} = "{}"'.format(key, value))
            else:
                exec('self.{} = {}'.format(key, value))
                        
        self.data_dir = 'dataset/' + self.dataset + '/'
        if self.dataset == 'aifb':
            self.bans = ['employs', 'affiliation']
            self.keyword = 'instance'
            self.Y_dict = {'1': 0, '2': 1, '3': 2, '4': 3}
            self.loc = [2, -8]
        elif self.dataset == 'mutag':
            self.bans = ['isMutagenic']
            self.keyword = '#d'
            self.Y_dict = {'0.0': 0, '1.0': 1}
            self.loc = [0, 3]
        elif self.dataset == 'bgs':
            self.bans = ['hasLithogenesis']
            self.keyword = 'NamedRockUnit'
            self.Y_dict = {'FLUV': 0, 'GLACI': 1}
            self.loc = [0, 5]
        self.out_dim = len(self.Y_dict)
        
        
    def load_data(self):
        """
        (1) Loads triplet.txt data, only keeps the triplets don't have ban
            words, and contain at least one keyword.
        (2) Get entity dict and relation dict.
        (3) Get label matrix and train/dev/test range.
        """
        
        with open(self.data_dir + 'triplet.txt') as file:
            lines = file.readlines()

        H, R, T = [], [], []
        for line in lines[: -1]:
            line = line.split()
            h, r, t = line[0], line[1], ' '.join(line[2: -1])
            valid = True
            for ban in self.bans:
                if ban in h or ban in t:
                    valid = False
                    break
            if self.keyword not in h and self.keyword not in t:
                valid = False
            if valid:
                H.append(h)
                R.append(r)
                T.append(t)      
                
        E_list = sorted(set(H + T))
        self.n_E = len(E_list)
        self.E_dict = dict(zip(E_list, range(self.n_E)))
        print('    #{:<7} entities.'.format(self.n_E))
        
        R_list = sorted(set(R))
        self.n_R = len(R_list)
        self.R_dict = dict(zip(R_list, range(self.n_R)))
        print('    #{:<7} relations.'.format(self.n_R))
        
        self.KG = []
        for i in range(len(H)):
            self.KG.append([self.E_dict[H[i]], self.R_dict[R[i]],
                            self.E_dict[T[i]]])
        self.KG = np.array(self.KG)
        print('    #{:<7} triplets.'.format(len(self.KG))) 
        
        self.Y = np.zeros((self.n_E, self.out_dim))
        train_range, test_range = [], []
        for key in ['train', 'test']:    
            with open(self.data_dir + key + '.txt') as file:
                lines = file.readlines()    
            for line in lines[1: ]:                    
                e, _, y = line.strip().split('\t')
                if self.dataset == 'bgs':
                    e = _
                e = self.E_dict['<' + e + '>']
                y = self.Y_dict[y.split('/')[-1][self.loc[0]: self.loc[1]]]
                self.Y[e, y] = 1.0
                if key == 'train':
                    train_range.append(e)
                else:
                    test_range.append(e)
        dev_range = train_range[: len(train_range) // 5]
        train_range = train_range[len(train_range) // 5: ]
        print('    #{:<7} train instances.'.format(len(train_range)))
        print('    #{:<7} dev   instances.'.format(len(dev_range)))
        print('    #{:<7} test  instances.'.format(len(test_range)))
        
        self.A = self.get_A()
        self.train_m = self.get_m(train_range)
        self.dev_m = self.get_m(dev_range)
        self.test_m = self.get_m(test_range)
        self.train_y = self.get_y(train_range)
        self.dev_y = self.get_y(dev_range)
        self.test_y = self.get_y(test_range)


    def get_m(self, key_range):
        """Get mask indexes, normalized to mean 1."""
        
        m = np.array([x in key_range for x in range(self.n_E)])
        return m / np.mean(m)
    
    
    def get_y(self, key_range):
        """Get masked label."""
        
        y = self.Y.copy()
        y[list(set(range(self.n_E)) - set(key_range)), :] = 0.0
        return y


    def common_structure(self):        
        """Common structure of R-GCN and R-GAT."""
        
        tf.reset_default_graph()
        self.mask = tf.placeholder(tf.float32, [self.n_E])
        self.label = tf.placeholder(tf.float32, [self.n_E, self.out_dim])
        self.keep = tf.placeholder(tf.float32)
        
        output = self.rgnn_layer()                            

        loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( \
                    logits = output, labels = self.label) * self.mask) + \
                    self.l2 * loss
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output,
                        1), tf.argmax(self.label, 1)), tf.float32) * self.mask)
        self.train_op = tf.train.AdamOptimizer(self.l_r).minimize(self.loss)
    
    
    def _train(self, sess):  
        """Training Process."""
        
        eps = self.epoches
        print('              Train          Dev')
        print('    EPOCH  LOSS   ACC    LOSS   ACC   time   TIME')
        
        temp_kpi, KPI = [], []
        t0 = t1 = time.time()
        for ep in range(eps):
            feed_dict = {self.label: self.train_y, self.mask: self.train_m,
                         self.keep: 1.0 - self.dropout}
            feed_dict.update(self.feed_dict)
            _, train_loss, train_acc = \
                sess.run([self.train_op, self.loss, self.accuracy], feed_dict)
            dev_loss, dev_acc = self._evaluate(sess)
            if (ep + 1) % 10 == 0:
                _t = time.time()
                print('    {:^5} {:^6.4f} {:^5.3f}  {:^6.4f} {:^5.3f}' \
                      ' {:^6.2f} {:^6.2f}'.format(ep + 1, train_loss,
                      train_acc, dev_loss, dev_acc, _t - t1, _t - t0))
                t1 = _t
        
            if ep == 0 or dev_loss < KPI[-1]:
                if len(temp_kpi) > 0:
                    KPI.extend(temp_kpi)
                    temp_kpi = []
                KPI.append(dev_loss)
            else:
                if len(temp_kpi) == self.earlystop:
                    break
                else:
                    temp_kpi.append(dev_loss)
                    
        best_ep = len(KPI)
        if best_ep != eps:
            print('\n    Early stop at epoch of {} !'.format(best_ep))
        

    def _evaluate(self, sess):
        """Validation Process."""
        
        feed_dict = {self.label: self.dev_y, self.mask: self.dev_m,
                     self.keep: 1.0}
        feed_dict.update(self.feed_dict)
        return sess.run([self.loss, self.accuracy], feed_dict)


    def _test(self, sess):
        """Test Process."""
        
        feed_dict = {self.label: self.test_y, self.mask: self.test_m,
                     self.keep: 1.0}
        feed_dict.update(self.feed_dict)
        loss_test, acc_test = sess.run([self.loss, self.accuracy], feed_dict)
        print('\n    Test : [ Loss: {:.4f} ; Acc : {:.3f} ]\n'. \
              format(loss_test, acc_test))
        return acc_test
  
    
    def run(self, N): 
        """Repeat N times run and calculate mean accuracy."""
        
        Acc = []
        for i in range(N):            
            config = tf.ConfigProto() 
            config.gpu_options.allow_growth = True
            self.common_structure()
            if i == 0:                           
                print('\n    *Hidden Dim      : {}'.format(self.h_dim))
                if 'n_B' in self.args:
                    print('    *Number of Basis : {}'.format(self.n_B))
                if 'n_head_1' in self.args:
                    print('    *Head of layer1  : {}'.format(self.n_head_1))
                    print('    *Head of layer2  : {}'.format(self.n_head_2))
                    print('    *Number of Hop   : {}'.format(self.n_hop))
                print('    *Drop Out Rate   : {}'.format(self.dropout))
                print('    *L2 Rate         : {}'.format(self.l2))
                print('    *Learning Rate   : {}'.format(self.l_r))
                print('    *Epoches         : {}'.format(self.epoches))
                print('    *Earlystop Step  : {}\n'.format(self.earlystop))
                    
                shape = {re.match('^(.*):\\d+$', v.name).group(1):
                          v.shape.as_list() for v in tf.trainable_variables()}
                tvs = [re.match('^(.*):\\d+$', v.name).group(1)
                        for v in tf.trainable_variables()]                      
                for v in tvs:
                    print('    -{} : {}'.format(v, shape[v]))
                
            with tf.Session(config = config) as sess:
                tf.global_variables_initializer().run()  
                print('\n>>  {} | {} Training Process.'.format(i + 1, N))
                self._train(sess)
                Acc.append(self._test(sess))
        
        print('\n>>  Result of {} Runs: {:.3f} ({:.3f})'.format(N,
              np.mean(Acc), np.std(Acc)))