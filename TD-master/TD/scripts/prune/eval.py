import tensorflow as tf
import os
import numpy as np
import time
import shutil

from ...hparams.registry import get_hparams
from ...models.registry import get_model
from ...data.registry import get_input_fns
from ...training import flags
from .prune import get_prune_fn, get_current_weights, get_louizos_masks, get_smallify_masks, prune_weights, new_prune_weights,is_prunable_weight



def check_make_path(p):
  if not os.path.exists(p):  # 判断文件夹是否存在
    os.mkdir(p)

def new_init_flags(train_steps=10000,eval_every=1000,env="local"):
  tf.flags.DEFINE_string("model", None, "Which model to use.")
  tf.flags.DEFINE_string("data", None, "Which data to use.")
  tf.flags.DEFINE_string("env", env, "Which environment to use.")
  tf.flags.DEFINE_string("hparams", None, "Which hparams to use.")
  tf.flags.DEFINE_string("hparam_override", "",
                         "Run-specific hparam settings to use.")
  tf.flags.DEFINE_string("output_dir", None, "The output directory.")
  tf.flags.DEFINE_string("data_dir", None, "The data directory.")
  tf.flags.DEFINE_integer("train_steps", train_steps,
                          "Number of training steps to perform.")
  tf.flags.DEFINE_integer("eval_every", eval_every,
                          "Number of steps between evaluations.")
  tf.flags.DEFINE_string(
      "post_weights_dir", "",
      "folder of the weights, if not set defaults to output_dir")
  tf.flags.DEFINE_string("prune_percent", "0.5",
                         "percent of weights to prune, comma separated")
  tf.flags.DEFINE_string("prune", "prune_mask", "one_shot or fisher")#tf.flags.DEFINE_string("prune", "weight", "one_shot or fisher")#
  tf.flags.DEFINE_boolean("variational", False, "use evaluate")
  tf.flags.DEFINE_string("eval_file", "eval_prune_results",
                         "file to put results")
  tf.flags.DEFINE_integer("train_epochs", None,
                          "Number of training epochs to perform.")
  tf.flags.DEFINE_integer("eval_steps", None,
                          "Number of evaluation steps to perform.")
  tf.flags.DEFINE_boolean("fresh", False, "Remove output_dir before running.")


def init_flags():
  tf.flags.DEFINE_string("model", None, "Which model to use.")
  tf.flags.DEFINE_string("data", None, "Which data to use.")
  tf.flags.DEFINE_string("env", "local_lewis", "Which environment to use.")
  tf.flags.DEFINE_string("hparams", None, "Which hparams to use.")
  tf.flags.DEFINE_string("hparam_override", "",
                         "Run-specific hparam settings to use.")
  tf.flags.DEFINE_string("output_dir", None, "The output directory.")
  tf.flags.DEFINE_string("data_dir", None, "The data directory.")
  tf.flags.DEFINE_integer("train_steps", 10000,
                          "Number of training steps to perform.")
  tf.flags.DEFINE_integer("eval_every", 1000,
                          "Number of steps between evaluations.")
  tf.flags.DEFINE_string(
      "post_weights_dir", "",
      "folder of the weights, if not set defaults to output_dir")
  tf.flags.DEFINE_string("prune_percent", "0.5",
                         "percent of weights to prune, comma separated")
  tf.flags.DEFINE_string("prune", "prune_mask", "one_shot or fisher")#tf.flags.DEFINE_string("prune", "weight", "one_shot or fisher")#
  tf.flags.DEFINE_boolean("variational", False, "use evaluate")
  tf.flags.DEFINE_string("eval_file", "eval_prune_results",
                         "file to put results")
  tf.flags.DEFINE_integer("train_epochs", None,
                          "Number of training epochs to perform.")
  tf.flags.DEFINE_integer("eval_steps", None,
                          "Number of evaluation steps to perform.")
  tf.flags.DEFINE_boolean("fresh", False, "Remove output_dir before running.")

def new_get_orig_weights(hparams,sess,model_fn,eval_input_fn):
  features, labels = eval_input_fn()
  #sess = tf.Session()

  tf.train.create_global_step()
  model_fn(features, labels, tf.estimator.ModeKeys.TRAIN)
  saver = tf.train.Saver()
  ckpt_dir = tf.train.latest_checkpoint(hparams.output_dir)
  print("Loading model from...", ckpt_dir)
  saver.restore(sess, ckpt_dir)

  fits = []

  mode = "standard"
  orig_weights = get_current_weights(sess)
  return orig_weights,saver

def get_orig_weights(hparams,sess,model_fn,eval_input_fn):
  features, labels = eval_input_fn()
  #sess = tf.Session()

  #tf.train.create_global_step()
  model_fn(features, labels, tf.estimator.ModeKeys.TRAIN)
  saver = tf.train.Saver()
  ckpt_dir = tf.train.latest_checkpoint(hparams.output_dir)
  print("Loading model from...", ckpt_dir)
  saver.restore(sess, ckpt_dir)

  fits = []

  mode = "standard"
  orig_weights = get_current_weights(sess)
  return orig_weights,saver

def weight_to_saver(FLAGS, hparam_name,file_name_frefix,cnt,weight_name,sess,bests,orig_weights,saver,old_ckpt):
  louizos_masks, smallify_masks = None, None
  if "louizos" in hparam_name:
    louizos_masks = get_louizos_masks(sess, orig_weights)
    mode = "louizos"
  elif "smallify" in hparam_name:
    smallify_masks = get_smallify_masks(sess, orig_weights)
  elif "variational" in hparam_name:
    mode = "variational"

  models=[]
  for i,p in enumerate(bests):
    ckpt_dir = old_ckpt
    saver = tf.train.Saver(max_to_keep=1)
    print("weight_to_saver Loading model from...", ckpt_dir)
    saver.restore(sess, ckpt_dir)
    prune_fn = get_prune_fn(FLAGS.prune)()
    w_copy = dict(orig_weights)
    sm_copy = dict(smallify_masks) if smallify_masks is not None else None
    lm_copy = dict(louizos_masks) if louizos_masks is not None else None
    post_weights_pruned, weight_counts = new_prune_weights(
      prune_fn,
      w_copy,
      weight_name,
      p[1])
    for v in tf.trainable_variables():
      if is_prunable_weight(v) and v.name.strip(":0")==weight_name:
        assign_op = v.assign(
          np.reshape(post_weights_pruned[v.name.strip(":0")], v.shape))
        sess.run(assign_op)
    print("population:"+str(i))
    filename=file_name_frefix+str(cnt)+"_"+str(i)
    #print(filename)
    #tf.train.get_or_create_global_step()
    saver.save(sess, filename)
    models.append([p[0],filename,p[1]])
  return models
  

def ga_eval_model(FLAGS, hparam_name,hparams,weight_name,sess,model_fn,eval_input_fn,population,orig_weights,meta_name,eval_steps,old_ckpt):
  fits=[]
  stamps=[]
  save_model = os.path.join(hparams.output_dir, "tmp", "model")
  def copy_meta(meta_name, src):
    src_meta_name = src + ".meta"
    if (meta_name != src_meta_name):
      shutil.copy(meta_name, src_meta_name)

  #all_var_list = [var for var in tf.trainable_variables()]
  copy_meta(meta_name, save_model)

  def saver_restore_part(sess, ckpt_dir):
    all_var_list = [var for var in tf.trainable_variables()]
    saver = tf.train.Saver(var_list=all_var_list)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    saver.restore(sess, ckpt_dir)

  def saver_save(sess, output_dir):
    saver = tf.train.Saver()
    saver.save(sess, output_dir)
  for i,p in enumerate(population):
    ckpt_dir = old_ckpt
    print("Loading model from...", ckpt_dir)
    saver_restore_part(sess, ckpt_dir)

    prune_fn = get_prune_fn(FLAGS.prune)()
    w_copy = dict(orig_weights)
    post_weights_pruned, weight_counts = new_prune_weights(
      prune_fn,
      w_copy,
      weight_name,
      p)
    print(weight_counts)

    print("there are ", len(tf.trainable_variables()), " weights")
    for v in tf.trainable_variables():
      if is_prunable_weight(v) and v.name.strip(":0")==weight_name:
        assign_op = v.assign(
          np.reshape(post_weights_pruned[v.name.strip(":0")], v.shape))
        sess.run(assign_op)
    print("population:"+str(i))
    #path = os.path.join(hparams.output_dir, "tmp")
    #check_make_path(path)
    gs = tf.train.get_global_step()
    save_model = os.path.join(hparams.output_dir, "tmp", "model")
    saver_save(sess,save_model)#,write_meta_graph=False

    estimator = tf.estimator.Estimator(
      model_fn=tf.contrib.estimator.replicate_model_fn(model_fn),
      model_dir=os.path.join(hparams.output_dir, "tmp"))
    print("Processing pruning {prune_percent} of weights for {hparams.eval_steps} steps")
    output_dir = os.path.join(hparams.output_dir, "tmp")
    output_dir = tf.train.latest_checkpoint(output_dir)
    print("aaaaaaaaaaaaaaaaa   "+output_dir)
    est = estimator.evaluate(eval_input_fn, steps=eval_steps,name="eval"+str(i))
    acc = est['acc']
    print("Accuracy @ prune {100*prune_percent}% is {acc}")
    fits.append(acc)
    now = int(round(time.time() * 1000))
    stamps.append(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000)))
  return fits,stamps

def eval_model(FLAGS, hparam_name):
  hparams = get_hparams(hparam_name)
  hparams = hparams.parse(FLAGS.hparam_override)
  hparams = flags.update_hparams(FLAGS, hparams,hparam_name)

  model_fn = get_model(hparams)
  _, _, test_input_fn = get_input_fns(hparams, generate=False)

  features, labels = test_input_fn()
  sess = tf.Session()



  tf.train.create_global_step()
  model_fn(features, labels, tf.estimator.ModeKeys.TRAIN)
  saver = tf.train.Saver()
  ckpt_dir = tf.train.latest_checkpoint(hparams.output_dir)
  print("Loading model from...", ckpt_dir)
  saver.restore(sess, ckpt_dir)

  evals = []
  prune_percents = [float(i) for i in FLAGS.prune_percent.split(",")]

  mode = "standard"
  orig_weights = get_current_weights(sess)
  louizos_masks, smallify_masks = None, None
  if "louizos" in hparam_name:
    louizos_masks = get_louizos_masks(sess, orig_weights)
    mode = "louizos"
  elif "smallify" in hparam_name:
    smallify_masks = get_smallify_masks(sess, orig_weights)
  elif "variational" in hparam_name:
    mode = "variational"

  for prune_percent in prune_percents:
    if prune_percent > 0.0:
      prune_fn = get_prune_fn(FLAGS.prune)(mode, k=prune_percent)
      w_copy = dict(orig_weights)
      sm_copy = dict(smallify_masks) if smallify_masks is not None else None
      lm_copy = dict(louizos_masks) if louizos_masks is not None else None
      post_weights_pruned, weight_counts = prune_weights(
          prune_fn,
          w_copy,
          louizos_masks=lm_copy,
          smallify_masks=sm_copy,
          hparams=hparams)
      print("current weight counts at {}: {}".format(prune_percent,
                                                     weight_counts))

      print("there are ", len(tf.trainable_variables()), " weights")
      for v in tf.trainable_variables():
        if is_prunable_weight(v):
          assign_op = v.assign(
              np.reshape(post_weights_pruned[v.name.strip(":0")], v.shape))
          sess.run(assign_op)

    saver.save(sess, os.path.join(hparams.output_dir, "tmp", "model"))
    estimator = tf.estimator.Estimator(
        model_fn=tf.contrib.estimator.replicate_model_fn(model_fn),
        model_dir=os.path.join(hparams.output_dir, "tmp"))
    print(
        "Processing pruning {prune_percent} of weights for {hparams.eval_steps} steps"
    )  
    acc = estimator.evaluate(test_input_fn, hparams.eval_steps)['acc']
    print("Accuracy @ prune {100*prune_percent}% is {acc}")
    evals.append(acc)
  return evals



def _run(FLAGS):
  eval_file = open(FLAGS.eval_file, "w")

  hparams_list = FLAGS.hparams.split(",")
  total_evals = {}
  for hparam_name in hparams_list:
    evals = eval_model(FLAGS, hparam_name)

    print(hparam_name, ":", evals)
    eval_file.writelines("{}:{}\n".format(hparam_name, evals))
    total_evals[hparam_name] = evals
    tf.reset_default_graph()

  print("processed results:", total_evals)
  eval_file.close()

def main():
    init_flags()
    FLAGS = tf.app.flags.FLAGS
    _run(FLAGS)

if __name__ == "__main__":
  init_flags()
  FLAGS = tf.app.flags.FLAGS
  _run(FLAGS)
