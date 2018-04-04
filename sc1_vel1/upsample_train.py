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

##
# inputs: 
#
# @param      x_train        coarse mesh - rest pos
# @param      y_train        fine mesh - coarse mesh
# @param      x_pred         prediction input, coarse mesh - rest pos
# @param      x_coarse       prediction coarse mesh
# @param      rest_file      The rest pos
# @param      mtx            mtx: 1292*3, 700
# @param      mtx_1          mtx_1: 
# @param      epochs         The epochs
# @param      learning_rate  The learning rate
# @param      logdir         log directory for tensorboard
# @param      out_dir        The out dir
# @param      init_w         The initialize weight flag
# @param      lr_decay_rate  The lr decay rate
# @param      restore        The restore flag
#
# @return     { description_of_the_return_value }
# 
# 
def train_model(data, x_pred, rest_pos, rest_file, mask, epochs, learning_rate, logdir, out_dir, init_w, lr_decay_rate, restore, alpha, beta):
  """Train upsample for a number of steps."""
  tf.reset_default_graph()
  # print("x_train.shape: ", x_train.shape) # (100, 700, 3)
  # batch_size = x_train.shape[0]
  batch_size = BATCH_SIZE
  folder_size = FOLDER_SIZE

  vert_num = data.shape[1]
  # print("train_model:  ", x_train.shape, y_train.shape)
  out_dir += str(alpha)[0:5] +'_'+ str(beta)[0:5] + '/'
  if not os.path.exists(out_dir):
      os.makedirs(out_dir)

  indices, faces = util.getfaces(rest_file)
  
  X = tf.placeholder(tf.float32, shape=(None, vert_num, 6), name="x_train")
  Y = tf.placeholder(tf.float32, shape=(None, vert_num, 3), name="y_train")

  phase = tf.placeholder(tf.bool)
  keep_prob = tf.placeholder(tf.float32)

  # Build a Graph that computes the output from the model
  predicts = upsample.inference(X, batch_size, vert_num, mask, phase, keep_prob) #mtx, mtx_1,

  # Calculate loss
  loss_op = upsample.loss(Y, predicts, mask, alpha, beta)

  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)

  # training operator
  train_op = upsample.training(learning_rate, loss_op, global_step, lr_decay_rate)

  # Initialize the variables (i.e. assign their default value)
  init_op = tf.global_variables_initializer()

  # Add ops to save and restore all the variables.
  saver = tf.train.Saver()

  pred_step = 1000
  display_step = 250

  # Build the summary Tensor based on the TF collection of Summaries.
  merged_summary = tf.summary.merge_all()

  # create batch data
  training_data = Dataset(data)

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
  
  sc1_w0 = [v for v in tf.trainable_variables() if v.name == "sc1/weights:0"][0]
  sc1_w0.load(mask, session=sess)
  sess.run(sc1_w0)

  # check batch data
  # batch = training_data.next_batch(batch_size)
  # print("batch.shape:  ", batch.shape)
  # res_pos = util.load_pos(rest_file)
  # for i in range(1,10):
  #     x = batch[i,:,0:3]
  #     y = batch[i,:,6:9]
  #     x_coarse = np.squeeze(x) + res_pos
  #     y_fine = np.squeeze(y) + x_coarse
      
  #     util.write_obj(res_pos, vert_num, faces, "rest.obj")
  #     util.write_obj(x_coarse, vert_num, faces, "x_coarse_"+str(i)+".obj")
  #     util.write_obj(y_fine, vert_num, faces, "y_fine_"+str(i)+".obj")



  # Fit all training data
  for epoch in range(epochs):
      batch = training_data.next_batch(batch_size)

      [t, lr] = sess.run(train_op, feed_dict={X: batch[:,:,0:6], Y: batch[:,:,6:9], phase: True, keep_prob: 0.6})
      # print("Epoch:", '%04d' % (epoch+1), "g = ", g)

      # Display logs per epoch step
      if (epoch+1) % display_step == 0:
          # [c, s] = sess.run([loss, merged_summary], feed_dict={X: x_train, Y:y_train})
          [c, s] = sess.run([loss_op, merged_summary], feed_dict={X: batch[:,:,0:6], Y: batch[:,:,6:9], phase: False, keep_prob: 1.0})
          summary_writer.add_summary(s, epoch)
          print("Epoch:", '%04d' % (epoch+1), "learning_rate: ", lr, "loss = ", "{:.9f}".format(c))
          # w = [v for v in tf.trainable_variables() if v.name == "sc1/weights/Variable:0"][0]
          # print(sess.run(w))
 
      if (epoch+1) % pred_step == 0:
          t = int((epoch+1) / pred_step)
          # pred(x_pred, x_coarse, vert_num, out_dir, indices, faces, t, predicts, X, phase, keep_prob, sess)
          pred_data = Dataset(x_pred)
          pred_output = sess.run(predicts, feed_dict={ X:pred_data.next_folder(folder_size), phase: False, keep_prob: 1.0})
          save_obj(pred_output, rest_pos, vert_num, out_dir, faces, t)
 
          print("Epoch:", '%04d' % (epoch+1), "predicting: ", '%03d' % t)
          print('Global_step: %s' % tf.train.global_step(sess, global_step))
          
  summary_writer.close()
  print (time.clock() - t3, "seconds for training.")

  # Save the variables to disk.
  save_path = saver.save(sess, "saved_model/model.ckpt")
  print("Model saved in file: %s" % save_path)
  

##
# predict t: index (pred every x epoch and create a new folder for new prediction)
# x_coarse is the original vertex position, has to be the same dimension as y_mesh
#
# @param      x_pred    The x predicate
# @param      x_coarse  The x coarse
# @param      vert_num  The vertical number
# @param      out_dir   The out dir
# @param      indices   The indices
# @param      faces     The faces
# @param      t         { parameter_description }
# @param      predicts  The predicts
# @param      X         { parameter_description }
# @param      phase     The phase (training or learning)
# @param      sess      The sess
#
# @return     { description_of_the_return_value }
#
# def pred(x_pred, x_coarse, vert_num, out_dir, indices, faces, t, predicts, X, phase, keep_prob, sess):
  # pred_input = Dataset(x_pred)
  # folder_size = x_pred.shape[0]
  # n = x_pred.shape[1]
  
  # pred_output = sess.run(predicts, feed_dict={ X:pred_input.next_folder(folder_size), phase: False, keep_prob: 1.0})
  


def save_obj(output, rest_pos, vert_num, out_dir, faces, t):
    # save to new dir
    sdir = out_dir + '{0:03d}'.format(t) + '/'
    if not os.path.exists(sdir):
      os.makedirs(sdir)
    print("Save to >>> ", sdir)

    for i in range(0, FOLDER_SIZE):
        # print(tf.slice(pred_output, [i, 0, 0], [1, n, 9])).  #(1, 1292, 9)
        # y_mesh = tf.squeeze(tf.slice(pred_output, [i, 0, 0], [1, n, 9])).eval()
        y_mesh = tf.squeeze(tf.slice(output, [i, 0, 0], [1, vert_num, 3])).eval()
        y_mesh += rest_pos

        obj_out = sdir + '{0:05d}'.format(i + 1) + '.obj'
        util.write_obj(y_mesh, vert_num, faces, obj_out)


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

  # parser.add_argument('-alpha', help="alpha: input noise weight")
  # parser.add_argument('-beta', help="beta: l2 weight for loss")

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
    lr_decay_rate = float(args.lr_decay)

  if args.train:    
      x_train = np.empty(0)
      x_train_vel = np.empty(0)
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
      
      rest_file = fineDir + [f for f in os.listdir(fineDir) if not f.startswith('.')][0] + "/00000_00.obj"
      print("rest position: ", rest_file)
      # dim, mtx, mtx_1 = preprocess.meshmtx_wnb(rest_file)

      # mask matrix based geodesic distance
      rest_vert, edges, faces = preprocess.obj_loader(rest_file)
      num_v = len(rest_vert)
      rest_pos = preprocess.get_pos(rest_vert)

      geo_dist = preprocess.compute_geodesic_distance(num_v, edges)
      mask = preprocess.compute_mask(num_v, geo_dist)

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

      data = np.concatenate((x_train, y_train), axis=2)
      print(time.clock() - t0, "seconds loading training data.")
      
      # load data
      x_pred = np.empty(0)
      # x_coarse = np.empty(0)
      
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
              x_p = util.load_input_only(inDir + subdir, rest_pos)
              if x_pred.size == 0:
                  x_pred = x_p
              else: 
                  x_pred = np.vstack((x_pred, x_p))

              # if x_coarse.size == 0:
              #     x_coarse = x_c
              # else: 
              #     x_coarse = np.vstack((x_coarse, x_c))

      print (time.clock() - t1, "seconds loading test data.")
  
  max_count = 40
  for learning_rate in [1E-2, 1E-3]:
      print('Starting run for learning_rate %f' % learning_rate)
      # alpha = beta = 0.0001
      for count in range(max_count):
          # random sample
          alpha = 10.0 ** np.random.uniform(-3, -5)
          beta = 10.0 ** np.random.uniform(-1, -4)
          train_model(data, x_pred, rest_pos, rest_file, mask, epochs, learning_rate, logdir, outDir, init_w, lr_decay_rate, restore, alpha, beta)
          ###mtx, mtx_1,# x_coarse,

if __name__ == "__main__":
  # FLAGS = parser.parse_args()
  # tf.app.run()
  main()


