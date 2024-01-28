import csv
#import cloud
import os
import random
import tensorflow as tf
import numpy as np
import time
import copy
import logging

from .hparams.registry import get_hparams
from .models.registry import get_model
from .data.registry import get_input_fns
from .training.lr_schemes import get_lr
from .training.envs import get_env
from .training import flags
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
import glob
import shutil

def files(curr_dir = '.', ext = '*.exe'):
    """当前目录下的文件"""
    for i in glob.glob(os.path.join(curr_dir, ext)):
        yield i

def filecopy(src,des,src_file_name=None,des_file_name=None):
  dest_dir = des
  src=src+".*"
  new_str=""
  if des_file_name:
      mat = des_file_name + ".*"
      for i in files(dest_dir, mat):
          os.remove(i)
  for file in glob.glob(src):
     print(file)
     #if(not file.endswith(".meta")):
     if (not des_file_name):
        shutil.copy(file, dest_dir)
        file_base = os.path.basename(src)
        new_str = os.path.join(dest_dir,file_base)
     else:
        file_base=os.path.basename(file)
        new_str1 = file_base.replace(src_file_name,des_file_name)
        new_str1 = os.path.join(des,new_str1)
        str = shutil.copy(file, new_str1)
        new_str = os.path.join(des,des_file_name)
  return new_str

def copy_meta(meta_name,src):
    src_meta_name = src+".meta"
    if(meta_name!=src_meta_name):
        shutil.copy(meta_name, src_meta_name)


def init_flags(env="local",eval_every=1000,train_epochs=256):
  tf.flags.DEFINE_string("env", env, "Which environment to use.")  # required
  tf.flags.DEFINE_string("hparams", None, "Which hparams to use.")  # required
  # Utility flags
  tf.flags.DEFINE_string("hparam_override", "",
                         "Run-specific hparam settings to use.")
  tf.flags.DEFINE_boolean("fresh", False, "Remove output_dir before running.")
  tf.flags.DEFINE_integer("seed", None, "Random seed.")
  tf.flags.DEFINE_integer("train_epochs", train_epochs,
                          "Number of training epochs to perform.")
  tf.flags.DEFINE_integer("eval_steps", None,
                          "Number of evaluation steps to perform.")
  # TPU flags
  tf.flags.DEFINE_string("tpu_name", "", "Name of TPU(s)")
  tf.flags.DEFINE_integer(
      "tpu_iterations_per_loop", 1000,
      "The number of training steps to run on TPU before"
      "returning control to CPU.")
  tf.flags.DEFINE_integer(
      "tpu_shards", 8, "The number of TPU shards in the system "
      "(a single Cloud TPU has 8 shards.")
  tf.flags.DEFINE_boolean(
      "tpu_summarize", False, "Save summaries for TensorBoard. "
      "Warning: this will slow down execution.")
  tf.flags.DEFINE_boolean("tpu_dedicated", False,
                          "Do not use preemptible TPUs.")
  tf.flags.DEFINE_string("data_dir", None, "The data directory.")
  tf.flags.DEFINE_string("output_dir", None, "The output directory.")
  tf.flags.DEFINE_integer("eval_every", eval_every,
                          "Number of steps between evaluations.")


tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = None


def init_random_seeds():
  tf.set_random_seed(FLAGS.seed)
  random.seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)


def init_model(hparams_name):
  flags.validate_flags(FLAGS)

  tf.reset_default_graph()

  hparams = get_hparams(hparams_name)
  hparams = hparams.parse(FLAGS.hparam_override)
  hparams = flags.update_hparams(FLAGS, hparams, hparams_name)

  # set larger eval_every for TPUs to improve utilization
  if FLAGS.env == "tpu":
    FLAGS.eval_every = max(FLAGS.eval_every, 5000)
    hparams.tpu_summarize = FLAGS.tpu_summarize

  tf.logging.warn("\n-----------------------------------------\n"
                  "BEGINNING RUN:\n"
                  "\t hparams: %s\n"
                  "\t output_dir: %s\n"
                  "\t data_dir: %s\n"
                  "-----------------------------------------\n" %
                  (hparams_name, hparams.output_dir, hparams.data_dir))

  return hparams


def construct_estimator(model_fn, hparams, tpu=None):
  if hparams.use_tpu:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        tpu=tpu.name)
    master = tpu_cluster_resolver.get_master()
    config = tpu_config.RunConfig(
        master=master,
        evaluation_master=master,
        model_dir=hparams.output_dir,
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True),
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=FLAGS.tpu_iterations_per_loop,
            num_shards=FLAGS.tpu_shards),
        save_checkpoints_steps=FLAGS.eval_every)
    estimator = tpu_estimator.TPUEstimator(
        use_tpu=hparams.use_tpu,
        model_fn=model_fn,
        model_dir=hparams.output_dir,
        config=config,
        train_batch_size=hparams.batch_size,
        eval_batch_size=hparams.batch_size)
  else:
    gpu_config = tf.ConfigProto(allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        keep_checkpoint_max=1000,save_checkpoints_steps=FLAGS.eval_every, session_config=gpu_config)

    estimator = tf.estimator.Estimator(
        model_fn=tf.contrib.estimator.replicate_model_fn(model_fn),
        model_dir=hparams.output_dir,
        config=run_config)

  return estimator


def _run(hparams_name):
  """Run training, evaluation and inference."""
  hparams = init_model(hparams_name)
  original_batch_size = hparams.batch_size
  if tf.gfile.Exists(hparams.output_dir) and FLAGS.fresh:
    tf.gfile.DeleteRecursively(hparams.output_dir)

  if not tf.gfile.Exists(hparams.output_dir):
    tf.gfile.MakeDirs(hparams.output_dir)
  model_fn = get_model(hparams)
  train_input_fn, eval_input_fn, test_input_fn = get_input_fns(hparams)

  tpu = None

  estimator = construct_estimator(model_fn, hparams, tpu)

  if not hparams.use_tpu:
    features, labels = train_input_fn()
    sess = tf.Session()
    tf.train.get_or_create_global_step()

    model_fn(features, labels, tf.estimator.ModeKeys.TRAIN)
    sess.run(tf.global_variables_initializer())

  # output metadata about the run
  with tf.gfile.GFile(os.path.join(hparams.output_dir, 'hparams.txt'),
                      'w') as hparams_file:
    hparams_file.write("{}\n".format(time.time()))
    hparams_file.write("{}\n".format(str(hparams)))

  def loop(steps=FLAGS.eval_every):
    ret = None
    estimator.train(train_input_fn, steps=steps)
    if eval_input_fn:
      ret = estimator.evaluate(eval_input_fn, steps=hparams.eval_steps, name="eval")
    if test_input_fn:
      ret = estimator.evaluate(test_input_fn, steps=hparams.eval_steps, name="test")
    return ret

  loop(1)

  steps = estimator.get_variable_value("global_step")
  k = steps * original_batch_size / float(hparams.epoch_size)
  while k <= hparams.train_epochs:
    tf.logging.info("Beginning epoch %f / %d" % (k, hparams.train_epochs))

    ret = loop()
    print(ret)
    steps = estimator.get_variable_value("global_step")
    k = steps * original_batch_size / float(hparams.epoch_size)

def _run_new(hparams_name,best_go_max=50):
  """Run training, evaluation and inference."""
  d_filename = "best_train_model"
  hparams = init_model(hparams_name)
  original_batch_size = hparams.batch_size
  if tf.gfile.Exists(hparams.output_dir) and FLAGS.fresh:
    tf.gfile.DeleteRecursively(hparams.output_dir)

  if not tf.gfile.Exists(hparams.output_dir):
    tf.gfile.MakeDirs(hparams.output_dir)
  model_fn = get_model(hparams)
  train_input_fn, eval_input_fn, test_input_fn = get_input_fns(hparams)

  tpu = None
  estimator = construct_estimator(model_fn, hparams, tpu)

  if not hparams.use_tpu:
    features, labels = train_input_fn()
    sess = tf.Session()
    tf.train.get_or_create_global_step()

    model_fn(features, labels, tf.estimator.ModeKeys.TRAIN)
    sess.run(tf.global_variables_initializer())

  # output metadata about the run
  with tf.gfile.GFile(os.path.join(hparams.output_dir, 'hparams.txt'),
                      'w') as hparams_file:
    hparams_file.write("{}\n".format(time.time()))
    hparams_file.write("{}\n".format(str(hparams)))

  def loop(steps=FLAGS.eval_every):
    ret = {}
    estimator.train(train_input_fn, steps=steps)
    if eval_input_fn:
      ret["eval"] = estimator.evaluate(eval_input_fn, steps=hparams.eval_steps, name="eval")
    if test_input_fn:
      ret["test"] = estimator.evaluate(test_input_fn, steps=hparams.eval_steps, name="test")
    return ret

  loop(1)
  rets = []
  best_eval_acc = 0.0
  steps = estimator.get_variable_value("global_step")
  k = steps * original_batch_size / float(hparams.epoch_size)
  print(hparams.learning_rate)
  while k <= hparams.train_epochs:
    tf.logging.info("Beginning epoch %f / %d" % (k, hparams.train_epochs))

    ret = loop()
    print(ret)
    steps = estimator.get_variable_value("global_step")
    k = steps * original_batch_size / float(hparams.epoch_size)


    if (best_eval_acc < ret["eval"]["acc"]):
        best_eval_acc = ret["eval"]["acc"]
        now = int(round(time.time() * 1000))
        now_str=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now / 1000))
        old_file = tf.train.latest_checkpoint(hparams.output_dir)
        fp = open("best_models_list.csv", "a")
        content = "\n"+now_str+","+str(ret["eval"]["acc"])+","+str(ret["test"]["acc"])+","+old_file
        fp.write(content)
        fp.close()

        best_go_cnt = 0
    else:
        best_go_cnt += 1
        if best_go_max > 0 and best_go_max < best_go_cnt:
            break


def new_construct_estimator(FLAGS,model_fn, hparams, tpu=None,model_dir=None):
  if not model_dir:
    model_dir=hparams.output_dir
  if hparams.use_tpu:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        tpu=tpu.name)
    master = tpu_cluster_resolver.get_master()
    config = tpu_config.RunConfig(
        master=master,
        evaluation_master=master,
        model_dir=model_dir,
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True),
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=FLAGS.tpu_iterations_per_loop,
            num_shards=FLAGS.tpu_shards),
        save_checkpoints_steps=FLAGS.eval_every)
    estimator = tpu_estimator.TPUEstimator(
        use_tpu=hparams.use_tpu,
        model_fn=model_fn,
        model_dir=model_dir,
        config=config,
        train_batch_size=hparams.batch_size,
        eval_batch_size=hparams.batch_size)
  else:
    gpu_config = tf.ConfigProto(allow_soft_placement=True)
    gpu_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=FLAGS.eval_every, session_config=gpu_config)

    estimator = tf.estimator.Estimator(
        model_fn=tf.contrib.estimator.replicate_model_fn(model_fn),
        model_dir=model_dir,
        config=run_config)

  return estimator

def writer_to_csv(filename,content):
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerows(content)


def new_run(FLAGS, hparams, output_dir, train_epochs, model_fn, train_input_fn, eval_input_fn, test_input_fn, sess,
            saver, restore_saver_file, save_saver_file,result_file_name="trainstage_acc.csv", gens=[0]):
    """Run training, evaluation and inference."""
    original_batch_size = hparams.batch_size
    if tf.gfile.Exists(hparams.output_dir) and FLAGS.fresh:
        tf.gfile.DeleteRecursively(hparams.output_dir)

    if not tf.gfile.Exists(hparams.output_dir):
        tf.gfile.MakeDirs(hparams.output_dir)

    tpu = None
    estimator = new_construct_estimator(FLAGS, model_fn, hparams, tpu)

    saver.restore(sess, restore_saver_file)
    tf.train.get_or_create_global_step()

    # output metadata about the run
    with tf.gfile.GFile(os.path.join(hparams.output_dir, 'hparams.txt'),
                        'w') as hparams_file:
        hparams_file.write("{}\n".format(time.time()))
        hparams_file.write("{}\n".format(str(hparams)))

    def loop(steps=FLAGS.eval_every):
        ret = {}
        estimator.train(train_input_fn, steps=steps)
        if eval_input_fn:
            ret1 = estimator.evaluate(eval_input_fn, steps=hparams.eval_steps, name="eval")
            ret["eval"] = ret1
        if test_input_fn:
            ret2 = estimator.evaluate(test_input_fn, steps=hparams.eval_steps, name="test")
            ret["test"] = ret2
        return ret

    loop(1)

    steps = 0
    k = steps * original_batch_size / float(hparams.epoch_size)
    rets = []
    best_eval_acc = 0.
    best_test_acc = 0.
    while k <= train_epochs:
        tf.logging.info("Beginning epoch %f / %d" % (k, train_epochs))
        ret = loop(steps=FLAGS.eval_every)
        now = int(round(time.time() * 1000))
        str_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now / 1000))
        writer_to_csv(result_file_name, [gens + [k, ret["eval"]["acc"], ret["test"]["acc"], str_now]])
        print(ret)
        rets.append(ret)
        steps += FLAGS.eval_every
        k = steps * original_batch_size / float(hparams.epoch_size)

        if (best_eval_acc < ret["eval"]["acc"]):
            best_eval_acc = ret["eval"]["acc"]
            best_test_acc = ret["test"]["acc"]
            saver.save(sess, save_saver_file)
    return best_eval_acc, best_test_acc

def write_to_check_file(model_dir,content):
  cfile = os.path.join(model_dir, "checkpoint")
  fp=open(cfile,"w")
  fp.write('model_checkpoint_path: "'+os.path.basename(content)+'"')
  fp.close()

def test1(FLAGS, hparams, train_epochs, model_fn, train_input_fn, eval_input_fn,
            restore_file,save_file, result_file_name="trainstage_acc.csv", gens=[0],best_go_max=0):
    """Run training, evaluation and inference."""
    hparams_list = FLAGS.hparams.split(",")
    hparam_name = hparams_list[0]


    original_batch_size = hparams.batch_size
    if tf.gfile.Exists(hparams.output_dir) and FLAGS.fresh:
        tf.gfile.DeleteRecursively(hparams.output_dir)

    if not tf.gfile.Exists(hparams.output_dir):
        tf.gfile.MakeDirs(hparams.output_dir)

    #saver = tf.train.Saver(max_to_keep=10)

    tpu = None
    model_dir = os.path.join(hparams.output_dir,"tmp")
    try:
        shutil.rmtree(model_dir)
    except:
        print("不能删除文件夹："+model_dir)
    os.mkdir(model_dir)
    estimator = new_construct_estimator(FLAGS, model_fn, hparams, tpu,model_dir=model_dir)

    #saver.restore(sess, saver_file)
    #tf.train.get_or_create_global_step()
    filecopy(restore_file,model_dir)
    write_to_check_file(model_dir, restore_file)
    # output metadata about the run
    with tf.gfile.GFile(os.path.join(hparams.output_dir, 'hparams.txt'),
                        'w') as hparams_file:
        hparams_file.write("{}\n".format(time.time()))
        hparams_file.write("{}\n".format(str(hparams)))

    output_dir111 = tf.train.latest_checkpoint(model_dir)

    def loop(steps=FLAGS.eval_every):
        ret = {}
        estimator.train(train_input_fn, steps=steps)
        if eval_input_fn:
            ret1 = estimator.evaluate(eval_input_fn, steps=hparams.eval_steps, name="eval")
            ret["eval"] = ret1

        return ret

    loop(1)


    steps = 0
    k = steps * original_batch_size / float(hparams.epoch_size)
    rets = []
    best_eval_acc = 0.
    best_go_cnt=0
    while k <= train_epochs:
        tf.logging.info("Beginning epoch %f / %d" % (k, train_epochs))
        ret = loop(steps=FLAGS.eval_every)
        now = int(round(time.time() * 1000))
        str_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now / 1000))
        writer_to_csv(result_file_name, [gens + [k, ret["eval"]["acc"], str_now]])
        print(ret)
        rets.append(ret)
        steps += FLAGS.eval_every
        k = steps * original_batch_size / float(hparams.epoch_size)

        if (best_eval_acc < ret["eval"]["acc"]):
            best_eval_acc = ret["eval"]["acc"]
            old_file = tf.train.latest_checkpoint(model_dir)
            s_filename = os.path.basename(old_file)
            d_filename = os.path.basename(save_file)
            filecopy(old_file, hparams.output_dir,s_filename,d_filename)
            best_go_cnt=0
        else:
            best_go_cnt+=1
            if best_go_max>0 and best_go_max < best_go_cnt:
                break
            #write_to_check_file(content=save_file, model_dir=hparams.output_dir)
            #acc1 = test_new_eval_test(FLAGS, hparams, model_fn, eval_input_fn, test_input_fn, sess=sess)
            #print("best_eval_acc="+str(best_eval_acc)+",  best_test_acc="+str(best_test_acc))
            #print(acc1)
    return best_eval_acc


def new_eval_test(FLAGS, hparams, model_fn, eval_input_fn, test_input_fn,result_file_name="trainstage_acc.csv"):
    """Run training, evaluation and inference."""
    original_batch_size = hparams.batch_size

    if not tf.gfile.Exists(hparams.output_dir):
        tf.gfile.MakeDirs(hparams.output_dir)

    tpu = None

    estimator = new_construct_estimator(FLAGS, model_fn, hparams, tpu)
    #steps = estimator.get_variable_value("global_step")
    #tf.train.get_or_create_global_step()

    # output metadata about the run
    def loop(steps=FLAGS.eval_every):
        ret = {}
        #estimator.train(train_input_fn, steps=steps)
        if eval_input_fn:
            ret1 = estimator.evaluate(eval_input_fn, steps=hparams.eval_steps, name="eval")
            ret["eval"] = ret1
        if test_input_fn:
            ret2 = estimator.evaluate(test_input_fn, steps=hparams.eval_steps, name="test")
            ret["test"] = ret2
        return ret

    best_eval_acc = loop(1)
    writer_to_csv(result_file_name, [[-1 , -1, -1, best_eval_acc["eval"]["acc"], best_eval_acc["test"]["acc"]]])

    return best_eval_acc


def test_new_eval_test(FLAGS, hparams, model_fn, eval_input_fn, test_input_fn, sess,checkpoint_path=None,result_file_name="trainstage_acc.csv",steps=None):
    """Run training, evaluation and inference."""
    original_batch_size = hparams.batch_size
    if steps == None:
        steps = hparams.eval_steps
    if not tf.gfile.Exists(hparams.output_dir):
        tf.gfile.MakeDirs(hparams.output_dir)

    tpu = None

    estimator = new_construct_estimator(FLAGS, model_fn, hparams, tpu,model_dir=hparams.output_dir)
    # steps = estimator.get_variable_value("global_step")
    # tf.train.get_or_create_global_step()

    # output metadata about the run
    def loop(steps=steps):
        ret = {}
        # estimator.train(train_input_fn, steps=steps)
        if eval_input_fn:
            ret1 = estimator.evaluate(eval_input_fn, steps=steps, name="eval",checkpoint_path=checkpoint_path)
            ret["eval"] = ret1
        if test_input_fn:
            ret2 = estimator.evaluate(test_input_fn, steps=steps, name="test",checkpoint_path=checkpoint_path)
            ret["test"] = ret2
        return ret

    best_eval_acc = loop(steps=steps)
    writer_to_csv(result_file_name, [[-1, -1, -1, best_eval_acc["eval"]["acc"], best_eval_acc["test"]["acc"]]])

    return best_eval_acc


def main(_):
  global FLAGS
  FLAGS = tf.app.flags.FLAGS

  init_random_seeds()
  #if FLAGS.env != "local":
  #  cloud.connect()
  for hparams_name in FLAGS.hparams.split(","):
    _run_new(hparams_name,best_go_max=500)

if __name__ == "__main__":
  init_flags()
  tf.app.run()
