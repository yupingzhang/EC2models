from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import argparse, os, sys
import numpy as np
import tensorflow as tf
import time
import upsample
import util
import preprocess
from batch_conf import Dataset 


BATCH_SIZE = 20
FOLDER_SIZE = 100


# inputs:  
# x_train, y_train
# logdir: log directory for tensorboard
# predict, bool, predict flag
def train_model(x_train, y_train, x_pred, x_coarse, rest_file, mtx, mtx_1, epochs, learning_rate, logdir, out_dir, init_w, lr_decay_rate, restore):
  """Train upsample for a number of steps."""
  tf.reset_default_graph()
  # print("x_train.shape: ", x_train.shape) # (100, 700, 3)
  # batch_size = x_train.shape[0]
  batch_size = BATCH_SIZE
  folder_size = FOLDER_SIZE

  tri_num = x_train.shape[1]
  vert_num = len(mtx)
  # print("train_model:  ", x_train.shape, y_train.shape)
  rest_pos = util.load_pos(rest_file)
  indices, faces = util.getfaces(rest_file)
  
  X = tf.placeholder(tf.float32, shape=(None, x_train.shape[1], x_train.shape[2]), name="x_train")
  Y = tf.placeholder(tf.float32, shape=(None, y_train.shape[1], y_train.shape[2]), name="y_train")
  phase = tf.placeholder(tf.bool, shape=())

  # Build a Graph that computes the output from the model
  predicts = upsample.inference(X, batch_size, tri_num, vert_num, mtx, mtx_1, phase)

  # Calculate loss
  loss_op = upsample.loss(Y, predicts)

  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)

  # training operator
  train_op = upsample.training(learning_rate, loss_op, global_step, lr_decay_rate)

  # Initialize the variables (i.e. assign their default value)
  init_op = tf.global_variables_initializer()

  # Add ops to save and restore all the variables.
  saver = tf.train.Saver()

  pred_step = 1000
  display_step = 1000

  # Build the summary Tensor based on the TF collection of Summaries.
  merged_summary = tf.summary.merge_all()

  # create batch data
  # x = tf.train.batch(tf.convert_to_tensor(x_train), batch_size);
  x = Dataset(x_train)
  y = Dataset(y_train)

  # Start training
  # with tf.Session() as sess:
  sess = tf.InteractiveSession()

  t3 = time.clock()
  summary_writer = tf.summary.FileWriter(logdir, sess.graph)

  if restore:
      saver.restore(sess, "saved_model/model.ckpt")
      print("Model restored.")
  else:
      # Run the initializer
      sess.run(init_op)

  # # load the init weights
  # if init_w:
  #   u_path = "u_weights/upsample.txt"
  #   u = util.load_weights(u_path)
  #   w0 = [v for v in tf.trainable_variables() if v.name == "fc1/W:0"][0]
  #   # w0 = tf.get_variable('W', initializer=u)
  #   # w0.initializer.run()
  #   w0.load(u, session=sess)
  #   sess.run(w0)

  # Fit all training data
  for epoch in range(epochs):
      # sess.run(train_op, feed_dict={X: x_train, Y: y_train})
      [t, lr] = sess.run(train_op, feed_dict={X: x.next_batch(batch_size), Y: y.next_batch(batch_size), phase: True})
      # print("Epoch:", '%04d' % (epoch+1), "g = ", g)

      # Display logs per epoch step
      if (epoch+1) % display_step == 0:
          # [c, s] = sess.run([loss, merged_summary], feed_dict={X: x_train, Y:y_train})
          [c, s] = sess.run([loss_op, merged_summary], feed_dict={X: x.next_batch(batch_size), Y: y.next_batch(batch_size), phase: False})
          summary_writer.add_summary(s, epoch)
          print("Epoch:", '%04d' % (epoch+1), "learning_rate: ", lr, "loss = ", "{:.9f}".format(c))
 
      if (epoch+1) % pred_step == 0:
          t = int((epoch+1) / pred_step)
          pred(x_pred, x_coarse, vert_num, out_dir, indices, faces, t, predicts, X, phase, sess)
          print("Epoch:", '%04d' % (epoch+1), "predicting: ", '%03d' % t)
          
  summary_writer.close()
  print (time.clock() - t3, "seconds for training.")

  # Save the variables to disk.
  save_path = saver.save(sess, "saved_model/model.ckpt")
  print("Model saved in file: %s" % save_path)
  


# predict
# t: index (pred every x epoch and create a new folder for new prediction)
# x_coarse is the original vertex position, has to be the same dimension as y_mesh
def pred(x_pred, x_coarse, vert_num, out_dir, indices, faces, t, predicts, X, phase, sess):
  pred_input = Dataset(x_pred)
  folder_size = x_pred.shape[0]
  n = x_pred.shape[1]
  
  pred_output = sess.run(predicts, feed_dict={ X:pred_input.next_folder(folder_size), phase: False})
  # print(pred_output.shape)
  
  # save to new dir
  sdir = out_dir + '{0:03d}'.format(t) + '/'
  if not os.path.exists(sdir):
    os.makedirs(sdir)
  print("Save to >>> ", sdir)

  for i in range(0, folder_size):
      # print(tf.slice(pred_output, [i, 0, 0], [1, n, 9])).  #(1, 1292, 9)
      y_mesh = tf.squeeze(tf.slice(pred_output, [i, 0, 0], [1, n, 9])).eval()
      y_mesh += x_coarse[i]

      obj_out = sdir + '{0:05d}'.format(i) + '.obj'
      util.tri2obj(y_mesh, vert_num, indices, faces, obj_out) 



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-train', action='store_true', default=True, help="train flag")
  # parser.add_argument('-eval', action='store_true', default=False, help="evaluate flag")
  # parser.add_argument('-pred', action='store_true', default=True, help="predict flag")
  # parser.add_argument('-w', action='store_true', default=False, help="load weights flag")
  parser.add_argument('-c', help="training: coarse dir")
  parser.add_argument('-f', help="training: fine scale with track dir")
  parser.add_argument('-logdir', help="logdir")
  # parser.add_argument('-tc', help="test dataset: coarse dir")
  # parser.add_argument('-tf', help="test dataset: fine scale with track dir")
  parser.add_argument('-x', help="predict input dataset dir") 
  parser.add_argument('-o', help="predict output dir") 
  parser.add_argument('-l', help="learning rate") 
  parser.add_argument('-e', help="epochs") 
  # parser.add_argument('-p', help="png file name")
  # parser.add_argument("-resume", help="bool flag, False by default")
  # parser.add_argument("-modelh5", help="load exist model")
  # parser.add_argument("-modelweighth5", help="load model weights")
  # parser.add_argument('-m', help="M")
  # parser.add_argument('-n', help="N")
  parser.add_argument('-restore', action='store_true', default=False, help="restore trained model")
  parser.add_argument('-init_w', action='store_true', default=False, help="init the weight from upsample.txt")
  parser.add_argument('-lr_decay', help="learning rate decay rate")

  # FLAGS = parser.parse_args()
  # args = parser.parse_args()
  args, unknown = parser.parse_known_args()
  if len(sys.argv) < 3:
    print("Usage: python upsample_train.py -c -f -logdir -x -o -l -e -restore -init_w -lr_decay")
    return
  # if args.m and args.n is not None:
  #     m = int(args.m)
  #     n = int(args.n)
  #     print("m and n for prediction: ", m, n)
  # else:
  #     m = 700
  #     n = 700
  #     print("No parameters m and n for prediction, use: ", m, n)

  restore = False
  init_w = False
  lr_decay_rate = 0
  if args.restore:
    restore = True
  if args.init_w:
    init_w = True
  if args.lr_decay:
    lr_decay_rate = args.lr_decay

  if args.train:    
      x_train = np.empty(0)
      y_train = np.empty(0)
      x_test = np.empty(0)
      y_test = np.empty(0)

      learning_rate = float(args.l)
      epochs = int(args.e)
      coarseDir = args.c
      fineDir = args.f
      logdir = args.logdir

      if not os.path.exists(logdir):
        os.makedirs(logdir)

      sdir = coarseDir
      rest_file = sdir + [f for f in os.listdir(sdir) if not f.startswith('.')][0] + "/00001_00.obj"
      dim, mtx, mtx_1 = preprocess.meshmtx_wnb(rest_file)
      rest_pos = util.load_pos(rest_file)

      print("training dataset: ")
      print(">>>  " + str(coarseDir) + "  >>>  " + str(fineDir))
      t0 = time.clock()
      print(">>>>>>> loading data for training  >>>>>>> ")
      for dirName, subdirList, fileList in os.walk(coarseDir):
          total = len(subdirList)
          count = 0
          for subdir in subdirList:
              # print('Found directory: %s' % subdir)
              if count%40 == 0:
                  print(str(float(count)/total*100) + '%')
              count = count + 1
              x, y = util.load_data(coarseDir + subdir, fineDir + subdir, rest_pos)

              if x_train.size == 0:
                  x_train = x
                  y_train = y
              else: 
                  x_train = np.vstack((x_train, x))
                  y_train = np.vstack((y_train, y))  

      print(time.clock() - t0, "seconds loading training data.")
      if x_train.size == 0:
          print("Error: no input training data.")
          return 0
      
      # load data
      x_pred = np.empty(0)
      x_coarse = np.empty(0)
      outDir = "pred/"
  # if args.pred:
      inDir = args.x
      outDir = args.o
      print(">>>>>>> loading data for prediction >>>>>>>> ")
      t1 = time.clock()
      for dirName, subdirList, fileList in os.walk(inDir):
          total = len(subdirList)
          for subdir in subdirList:
              # print('Found directory: %s' % subdir)
              x_p, x_c = util.load_input_only(inDir + subdir, rest_file)
              if x_pred.size == 0:
                  x_pred = x_p
              else: 
                  x_pred = np.vstack((x_pred, x_p))

              if x_coarse.size == 0:
                  x_coarse = x_c
              else: 
                  x_coarse = np.vstack((x_coarse, x_c))

      print (time.clock() - t1, "seconds loading test data.")

  # batch_size = x_pred.shape[0]

  # for learning_rate in [1E-1, 1E-2]:
  #   print('Starting run for learning_rate %f' % learning_rate)

  # train_model(x_train, y_train, dim, mtx, mtx_1, epochs, learning_rate, logdir)
  train_model(x_train, y_train, x_pred, x_coarse, rest_file, mtx, mtx_1, epochs, learning_rate, logdir, outDir, init_w, lr_decay_rate, restore)
  

if __name__ == "__main__":
  # FLAGS = parser.parse_args()
  # tf.app.run()
  main()


