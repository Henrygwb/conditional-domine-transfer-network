import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import scipy.io
import scipy.misc
import argparse
from sklearn.utils import shuffle

class CS_DTN(object):
    """
    conditional supervised Domine transfer network
    """
    def __init__(self, mode, learning_rate, num_class, const = 15):
        self.mode = mode
        self.learning_rate = learning_rate
        self.num_class = num_class
        self.const = const

    def content_extractor(self, images, reuse = False):
        # (batch_size, 32, 32, 1)
        #if images.get_shape()[3] == 1:
        #    images = tf.image.grayscale_to_rgb(images)

        with tf.variable_scope('content_extractor', reuse = reuse):
            with slim.arg_scope([slim.conv2d], padding = 'SAME', activation_fn = None,
                                stride = 2, weights_initializer = tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay = 0.95, center = True,
                                    scale = True, activation_fn = tf.nn.relu, is_training = (self.mode == 'train' or self.mode == 'pretrain')):
                    net = slim.conv2d(images, 64, [3, 3], scope='conv1') # (batch_size, 16, 16, 128)
                    net = slim.batch_norm(net, scope='bn1')
                    net = slim.conv2d(net, 128, [3, 3], scope='conv2') # (batch_size, 8, 8, 128)
                    net = slim.batch_norm(net, scope='bn2')
                    net = slim.conv2d(net, 128, [3, 3], scope='conv3') # (batch_size, 4, 4, 128)
                    net = slim.batch_norm(net, scope='bn3')
                    net = slim.conv2d(net, 128, [4, 4], padding = 'VALID', scope = 'conv4') # (batch_size, 1, 1, 128)
                    net = slim.batch_norm(net, activation_fn=tf.nn.tanh, scope='bn4')
                    if self.mode == 'pretrain':
                        net = slim.conv2d(net, 26, [1, 1], padding='VALID', scope='out')
                        net = slim.flatten(net)
        return net

    def generator(self, inputs, y_labels, reuse = False):
        # inputs: (batch_size, 1, 1, 128);
        # labels: (batch_size, 1, 1, num_class)
        inputs = tf.concat([inputs, y_labels], 3)
        with tf.variable_scope('generator', reuse = reuse):
            with slim.arg_scope([slim.conv2d_transpose], padding = 'SAME', activation_fn = None,
                           stride = 2, weights_initializer = tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay = 0.95, center = True, scale = True,
                                    activation_fn = tf.nn.relu, is_training = (self.mode == 'train')):
                    net = slim.conv2d_transpose(inputs, 512, [4, 4], padding='VALID', scope='conv_transpose1')  # (batch_size, 4, 4, 512)
                    net = slim.batch_norm(net, scope='bn1')
                    net = slim.conv2d_transpose(net, 256, [3, 3], scope='conv_transpose2')  # (batch_size, 8, 8, 256)
                    net = slim.batch_norm(net, scope='bn2')
                    net = slim.conv2d_transpose(net, 128, [3, 3], scope='conv_transpose3')  # (batch_size, 16, 16, 128)
                    net = slim.batch_norm(net, scope='bn3')
                    net = slim.conv2d_transpose(net, 1, [3, 3], activation_fn=tf.nn.tanh, scope='conv_transpose4')  # (batch_size, 32, 32, 1)
        return net

    def discriminator(self, images, y_fills, reuse = False):
        # images: (batch_size, 32, 32, 1)
        # y_fills: (batch_size, 32, 32, num_class)
        inputs = tf.concat([images, y_fills], 3)
        with tf.variable_scope('discriminator', reuse=reuse):
            with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=None,
                                stride=2, weights_initializer=tf.contrib.layers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                    activation_fn=tf.nn.relu, is_training=(self.mode == 'train')):
                    net = slim.conv2d(inputs, 128, [3, 3], activation_fn=tf.nn.relu, scope='conv1')  # (batch_size, 16, 16, 128)
                    net = slim.batch_norm(net, scope='bn1')
                    net = slim.conv2d(net, 256, [3, 3], scope='conv2')  # (batch_size, 8, 8, 256)
                    net = slim.batch_norm(net, scope='bn2')
                    net = slim.conv2d(net, 256, [3, 3], scope='conv3')  # (batch_size, 4, 4, 256)
                    net = slim.batch_norm(net, scope='bn3')
                    net = slim.conv2d(net, 1, [4, 4], padding='VALID', scope='conv4')  # (batch_size, 1, 1, 1)
                    net = slim.flatten(net)
        return net

    def build_model(self):
        if self.mode == 'pretrain':
            self.images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'source_images')
            self.labels = tf.placeholder(tf.int64, [None], 'source_label')
            
            # logits and accuracy
            self.logits = self.content_extractor(self.images)
            self.pred = tf.argmax(self.logits, 1)
            self.correct_pred = tf.equal(self.pred, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

            # loss and train
            self.loss = slim.losses.sparse_softmax_cross_entropy(self.logits, self.labels)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = slim.learning.create_train_op(self.loss, self.optimizer)

        elif self.mode == 'eval':
            self.src_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'source_images')
            self.y_labels = tf.placeholder(tf.float32, [None, 1, 1, self.num_class], 'g_label')
            
            self.fx = self.content_extractor(self.src_images)
            self.sampled_images = self.generator(self.fx, self.y_labels)

        elif self.mode == 'train':
            self.src_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'source_images')
            self.trg_images = tf.placeholder(tf.float32, [None, 32, 32, 1], 'target_images')
            self.y_fills = tf.placeholder(tf.float32, [None, 32, 32, self.num_class], 'd_label')
            self.y_labels = tf.placeholder(tf.float32, [None, 1, 1, self.num_class], 'g_label')

            # source domain 
            self.fx = self.content_extractor(self.src_images)
            self.fake_images = self.generator(self.fx, self.y_labels)
            self.logits_fake = self.discriminator(self.fake_images, self.y_fills)

            # target domain
            self.logits_real = self.discriminator(self.trg_images, self.y_fills, reuse=True)

            # loss
            self.d_loss_src = slim.losses.sigmoid_cross_entropy(self.logits_fake, tf.zeros_like(self.logits_fake))
            self.d_loss_trg = slim.losses.sigmoid_cross_entropy(self.logits_real, tf.ones_like(self.logits_real))
            self.d_loss = self.d_loss_src + self.d_loss_trg

            self.g_loss_fake_images = slim.losses.sigmoid_cross_entropy(self.logits_fake, tf.ones_like(self.logits_fake))
            #self.g_loss_const = tf.reduce_mean(tf.square(self.trg_images - self.fake_images)) * self.const
            self.g_loss = self.g_loss_fake_images #+ self.g_loss_const

            self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate)
            
            # trainable variable
            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'discriminator' in var.name]
            g_vars = [var for var in t_vars if 'generator' in var.name]
            
            # train op
            with tf.name_scope('train_op'):
                self.d_train_op = slim.learning.create_train_op(self.d_loss, self.d_optimizer, variables_to_train=d_vars)
                self.g_train_op = slim.learning.create_train_op(self.g_loss, self.g_optimizer, variables_to_train=g_vars)
        return 0



class Solver(object):
    def __init__(self, model, batch_size, pretrain_iter, train_iter,
                 source_dir, target_dir, sample_save_path, model_save_path, pretrained_model,
                 test_model, cls, cls_weight, config, allow_growth = True):

        self.model = model
        self.batch_size = batch_size
        self.pretrain_iter = pretrain_iter
        self.train_iter = train_iter
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.cls = cls
        self.cls_weight = cls_weight
        self.sample_save_path = sample_save_path
        self.model_save_path = model_save_path
        self.pretrained_model = pretrained_model
        self.test_model = test_model
        self.config = config 
        self.config.gpu_options.allow_growth = allow_growth

    def load_source(self, image_dir, split='train'):
        print 'loading source image dataset..'
        image_file = 'train.mat' if split == 'train' else 'train.mat'
        image_dir = os.path.join(image_dir, image_file)
        mnist = scipy.io.loadmat(image_dir)
        images = mnist['X']
        labels = mnist['y'].reshape(-1)-1
        if self.model.mode == 'eval':
            images_eval = images[np.where(labels == self.cls[0])][0:100, ]
            labels_eval = labels[np.where(labels == self.cls[0])][0:100, ]
            class_eval = np.array(self.cls)[1:]
            for i in class_eval:
                images_eval = np.concatenate((images_eval, images[np.where(labels==i)][0:100, ]))
                labels_eval = np.concatenate((labels_eval, labels[np.where(labels==i)][0:100, ]))
            images = images_eval
            labels = labels_eval
        return images, labels

    def load_target(self, image_dir, split='train'):
        print 'loading target image dataset..'
        image_file = 'train' if split == 'train' else 'test'
        image_dir = os.path.join(image_dir, image_file)
        target = scipy.io.loadmat(image_dir)
        images = target['X'] #/ 127.5 -1
        #images = np.expand_dims(images, 3)
        labels = target['y'].reshape(-1)-1
        return images, labels

    def split_class(self, source_images, source_labels, target_images, target_labels, cls):
        source_images = source_images[np.where(source_labels==cls)]
        source_labels = source_labels[np.where(source_labels==cls)]
        target_images = target_images[np.where(target_labels==cls)]
        target_labels = target_labels[np.where(target_labels==cls)]
        size = np.min((target_images.shape[0], source_images.shape[0]))
        source_images = source_images[0:size, ]
        source_labels = source_labels[0:size, ]
        target_images = target_images[0:size, ]
        return source_images, source_labels, target_images

    def load_train(self, source_dir, target_dir, split='train'):
        print 'loading train image dataset..'
        source_images, source_labels = self.load_source(source_dir, split)
        target_images, target_labels = self.load_target(target_dir, split)
        source_images_train, source_labels_train, target_images_train = self.split_class(source_images, source_labels,
                                                                        target_images, target_labels, self.cls[0])

        class_train = np.array(self.cls)[1:]
        for i in class_train:
            source_images_tmp, source_labels_tmp, target_images_tmp = self.split_class(source_images, source_labels, target_images, target_labels, i)
            source_images_train = np.concatenate((source_images_train, source_images_tmp))
            source_labels_train = np.concatenate((source_labels_train, source_labels_tmp))
            target_images_train = np.concatenate((target_images_train, target_images_tmp))

        return source_images_train, source_labels_train, target_images_train

    def merge_images(self, sources, targets, k=10):
        _, h, w, _ = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([row*h, row*w*2, 3])

        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[i*h:(i+1)*h, (j*2)*h:(j*2+1)*h, :] = s
            merged[i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h, :] = t
        return merged

    def pretrain(self):
        train_images, train_labels = self.load_source(self.source_dir, split='train')
        test_images, test_labels = self.load_source(self.source_dir, split='train')

        model = self.model
        model.build_model()

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()

            for step in range(self.pretrain_iter + 1):
                i = step % int(train_images.shape[0] / self.batch_size)
                batch_images = train_images[i * self.batch_size:(i + 1) * self.batch_size]
                batch_labels = train_labels[i * self.batch_size:(i + 1) * self.batch_size]
                feed_dict = {model.images: batch_images, model.labels: batch_labels}
                sess.run(model.train_op, feed_dict)

                if (step + 1) % 10 == 0:
                    l, acc = sess.run([model.loss, model.accuracy], feed_dict)
                    rand_idxs = np.random.permutation(test_images.shape[0])[:self.batch_size]
                    test_acc  = sess.run(fetches=model.accuracy,
                                           feed_dict={model.images: test_images[rand_idxs],
                                                      model.labels: test_labels[rand_idxs]})
                    print ('Step: [%d/%d] loss: [%.6f] train acc: [%.2f] test acc [%.2f]' % (step + 1, self.pretrain_iter, l, acc, test_acc))

                if (step + 1) % self.pretrain_iter == 0:
                    saver.save(sess, os.path.join(self.model_save_path, 'source_model'), global_step=step + 1)
                    print ('source_model-%d saved..!' % (step + 1))

    def train(self):

        source_images, source_labels, target_images = self.load_train(self.source_dir, self.target_dir, split='train')
        print np.unique(source_labels)

        onehot = np.zeros((source_labels.shape[0], 1, 1, len(self.cls)))
        for cls_idx in xrange(len(self.cls)):
            idx = np.where(source_labels == self.cls[cls_idx])
            onehot[idx, :, :, cls_idx] = 1

        if len(self.cls) > 1:
            for i, w in zip(self.cls, self.cls_weight):
                onehot[:,:,:,i][np.where(onehot[:,:,:,i]==1)] = w

        source_fills = onehot*np.ones([onehot.shape[0], 32, 32, len(self.cls)])

        if len(self.cls) == 1:
            src_labels = np.zeros((source_labels.shape[0], 1, 1, 1))
            src_fills = np.zeros((source_labels.shape[0], 32, 32, 1))

        source_images, onehot, source_fills, target_images = shuffle(source_images, onehot, source_fills, target_images)

        model = self.model
        model.build_model()

        with tf.Session(config=self.config) as sess:
            # initialize G and D
            tf.global_variables_initializer().run()
 
            # restore variables of F
            print ('loading pretrained model F..')
            variables_to_restore = slim.get_model_variables(scope='content_extractor')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_model)
            saver = tf.train.Saver()

            print ('start training..!')
            for step in range(self.train_iter + 1):
                for i in range(int(source_images.shape[0] / self.batch_size)):
                    src_images = source_images[i * self.batch_size:(i + 1) * self.batch_size]
                    src_labels = onehot[i * self.batch_size:(i + 1) * self.batch_size]
                    src_fills = source_fills[i * self.batch_size:(i + 1) * self.batch_size]
                    trg_images = target_images[i * self.batch_size:(i + 1) * self.batch_size]
 

                    feed_dict = {model.src_images: src_images, model.y_fills: src_fills,
                                 model.y_labels: src_labels, model.trg_images: trg_images}
                    sess.run(model.d_train_op, feed_dict)

                    """
		            add randomness    
		            #y_ = np.random.randint(0, 9, (self.batch_size, 1))
		            #y_label_ = onehot[y_.astype(np.int32)].reshape([self.batch_size, 1, 1, 10])
		            #y_fill_ = y_label_ * np.ones([self.batch_size, 32, 32, 10])
		            #feed_dict = {model.src_images: src_images, model.y_fills: y_fill_,
		            #model.y_labels: y_label_, model.trg_images: trg_images}
		            """

                    sess.run([model.g_train_op], feed_dict)
                    sess.run([model.g_train_op], feed_dict)
                    sess.run([model.g_train_op], feed_dict)
                    sess.run([model.g_train_op], feed_dict)
                    sess.run([model.g_train_op], feed_dict)
                    sess.run([model.g_train_op], feed_dict)

                if (step + 1) % 10 == 0:
                    dl, gl = sess.run([model.d_loss, model.g_loss], feed_dict)
                    print ('Step: [%d/%d] d_loss: [%.6f] g_loss: [%.6f]' % (step + 1, self.train_iter, dl, gl))

                if (step + 1) % self.train_iter == 0:
                    saver.save(sess, os.path.join(self.model_save_path, 'dtn'), global_step=step + 1)
                    print ('model/dtn-%d saved' % (step + 1))

    def eval(self):
        model = self.model
        model.build_model()

        source_images, source_labels = self.load_source(self.source_dir, 'train')

        with tf.Session(config=self.config) as sess:
            # load trained parameters
            print ('loading test model..')
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            print ('start sampling..!')
            for i in range(len(self.cls)):

                self.batch_size = 100
                batch_images = source_images[i * self.batch_size:(i + 1) * self.batch_size]
                src_labels = source_labels[i * self.batch_size:(i + 1) * self.batch_size]
                
                onehot = np.eye(len(self.cls))
                src_labels = onehot[src_labels].reshape((self.batch_size, 1, 1, len(self.cls)))
                
                if len(self.cls) > 1:
                    for ii, w in zip(self.cls, self.cls_weight):
                        src_labels[:, :, :, ii][np.where(src_labels[:, :, :, ii] == 1)] = w

                if len(self.cls) == 1:
                    src_labels = np.zeros((self.batch_size, 1, 1, 1))

                feed_dict = {model.src_images: batch_images, model.y_labels: src_labels }
                sampled_batch_images = sess.run(model.sampled_images, feed_dict)
                print sampled_batch_images.shape
                merged = self.merge_images(batch_images, sampled_batch_images)
                path = os.path.join(self.sample_save_path,
                                    'sample-%d-to-%d.png' % (i * self.batch_size, (i + 1) * self.batch_size))
                scipy.misc.imsave(path, merged)
                print ('saved %s' % path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="pretrain, train, or eval")
    parser.mode = 'train'
    #parser = parser.parse_args()
    model_save_path= 'model'
    sample_save_path = 'sample'

    model = CS_DTN(mode=parser.mode, learning_rate=0.0003, num_class = 2)
    solver = Solver(model, batch_size=500, pretrain_iter=200, train_iter=100,
                    source_dir='source', target_dir='target', model_save_path=model_save_path,
                    sample_save_path=sample_save_path, pretrained_model='model/source_model-200',
                    cls = [0, 1], cls_weight=[1,1], test_model='model/dtn-100', config = tf.ConfigProto())

    # create directories if not exist
    if not tf.gfile.Exists(model_save_path):
        tf.gfile.MakeDirs(model_save_path)
    if not tf.gfile.Exists(sample_save_path):
        tf.gfile.MakeDirs(sample_save_path)

    if parser.mode == 'pretrain':
        solver.pretrain()
    elif parser.mode == 'train':
        solver.train()
    else:
        solver.eval()
