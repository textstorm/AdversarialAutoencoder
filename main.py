
import tensorflow as tf
import numpy as np
import utils
import config
import time
import os

from model import AVB
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(args):
  #
  save_dir = os.path.join(args.save_dir, args.model_type)
  img_dir = os.path.join(args.img_dir, args.model_type)
  log_dir = os.path.join(args.log_dir, args.model_type)
  train_dir = args.train_dir

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  if not os.path.exists(img_dir):
    os.makedirs(img_dir)

  mnist = utils.read_data_sets(args.train_dir)
  # summary_writer = tf.summary.FileWriter(log_dir)
  config_proto = utils.get_config_proto()

  sess = tf.Session(config=config_proto)
  model = AVB(args, sess, name="dcvae")

  total_batch = mnist.train.num_examples // args.batch_size

  for epoch in range(1, args.nb_epochs + 1):
    print "Epoch %d start with learning rate %f" % (epoch, model.learning_rate.eval(sess))
    print "- " * 50
    epoch_start_time = time.time()
    step_start_time = epoch_start_time
    for i in range(1, total_batch + 1):
      global_step = sess.run(model.global_step)
      x_batch, y_batch = mnist.train.next_batch(args.batch_size)

      _, vae_loss = model.train_vae(x_batch, noise)
      _, adv_loss = model.train_adv(noise)

      if i % args.print_step == 0:
        print "epoch %d, batch %d, vae_loss %d, adv_loss %d" % (epoch, i, vae_loss, adv_loss)

    if epoch % 50 == 0:
      print "- " * 5

    if args.anneal and epoch >= args.anneal_start:
      sess.run(model.learning_rate_decay_op)

    if epoch % args.save_epoch == 0:
      # z = np.random.normal(size=[100, args.latent_dim])
      gen_images = np.reshape(model.generate(z, 100), (100, 28, 28, 1))
      utils.save_images(gen_images, [10, 10], os.path.join(img_dir, "sample%s.jpg" % epoch))

  model.saver.save(sess, os.path.join(save_dir, "model.ckpt"))
  print "Model stored...."

if __name__ == "__main__":
  args = config.get_args()
  main(args)