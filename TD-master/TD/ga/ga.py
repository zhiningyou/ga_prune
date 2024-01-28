import copy
import csv
import numpy as np
import heapq
import time
import random
import tensorflow as tf
from itertools import accumulate as _accumulate, repeat as _repeat
from bisect import bisect as _bisect
from ..scripts.prune.eval import ga_eval_model
from ..scripts.prune.prune import get_prune_fn,new_prune_weights,is_prunable_weight

def takeFirst(elem):
    return elem[0]
def takeSecond(elem):
    return elem[1]

def sample_back(samples, m, weights=None,cum_weights=None):
    """Return a k sized list of population elements chosen with replacement.
    If the relative weights or cumulative weights are not specified,
    the selections are made with equal probability.
    """
    n = len(samples)
    if cum_weights is None:
        if weights is None:
            _int = int
            n += 0.0  # convert to float for a small speed improvement
            return [samples[_int(random.random() * n)] for i in _repeat(None, m)]
        cum_weights = list(_accumulate(weights))
    elif weights is not None:
        raise TypeError('Cannot specify both weights and cumulative weights')
    if len(cum_weights) != n:
        raise ValueError('The number of weights does not match the population')
    bisect = _bisect
    try:
        total = cum_weights[-1] + 0.0  # convert to float
        hi = n - 1
        ret = [samples[bisect(cum_weights, random.random() * total, 0, hi)]
            for i in _repeat(None, m)]
    except:
        print("An exception occurred")
    return ret

def sample_no_back(samples, m, weights=None):
    """
    :samples: [(item, weight), ...]
    :k: number of selected items
    :returns: [(item, weight), ...]
    """
    n = len(samples)
    if weights is None:
        weights = [1 for i in range(n)]
    new_samples = zip(samples,weights)
    heap = [] # [(new_weight, item), ...]
    for sample in new_samples:
        wi = sample[1]
        ui = random.uniform(0, 1)
        ki = ui ** (1/wi)

        if len(heap) < m:
            heapq.heappush(heap, (ki, sample))
        elif ki > heap[0][0]:
            heapq.heappush(heap, (ki, sample))

            if len(heap) > m:
                heapq.heappop(heap)

    return [item[1][0] for item in heap]

def get_nonzero_pos(mask):
    new_mask = np.array(mask)
    nonzero = np.nonzero(new_mask)
    n = len(nonzero)
    l = [nonzero[i].tolist() for i in range(n)]
    z = list(zip(l[0], l[1]))
    return z

def get_zero_pos(mask):
    new_mask = np.array(~mask)
    nonzero = np.nonzero(new_mask)
    n = len(nonzero)
    l = [nonzero[i].tolist() for i in range(n)]
    z = list(zip(l[0], l[1]))
    return z

def weights_to_masks(orig_weights):
    masks = {}
    for key,value in orig_weights.items():
        mask=value!=0
        w = copy.deepcopy(mask)
        mask=mask.reshape([-1,w.shape[-1]])
        #prev_dim = np.product(w.shape)
        #mask = mask.reshape([prev_dim])
        masks[key]=mask
    return masks    

def select_bests(population,fits,bests,max_num):
    changed = False
    size = len(fits)
    for i in range(size):
        best_num = len(bests)
        if(best_num<max_num):
            bests.append([fits[i],population[i]])
            bests.sort(key=takeFirst)
        else:    
            for j in range(best_num):
                if(bests[j][0]<fits[i]):
                    bests[j] = [fits[i],population[i]]
                    #print(bests)
                    print("*********************************")
                    bests.sort(key=takeFirst)
                    changed = True
                    break
    return bests,changed

def writer_to_csv(filename,content,overwrite=False):
    dotype = 'a'
    if overwrite:
        dotype = 'w'
    with open(filename, dotype) as f:
        writer = csv.writer(f)
        writer.writerows(content)

class Ga_layer(object):
    def __init__(self,FLAGS, hparam_name,hparams,layer_name,sess,model_fn,eval_input_fn,orig_weights,model_p,eval_steps=5,v_rate=0.2,c_rate=0.8,best_num=3,prune_rate=0.5,ep=2,population_num=5,fit_file_name="fits.csv",best_file_name="best_fits.csv",out_gen=[0],meta_name=None):
        self.FLAGS=FLAGS
        self.hparam_name=hparam_name
        self.hparams=hparams
        self.layer_name=layer_name
        self.orig_weights=orig_weights
        self.total_old_masks=weights_to_masks(orig_weights)
        self.layer_old_mask=self.total_old_masks[layer_name]
        self.nonzero_pos = get_nonzero_pos(self.layer_old_mask)
        self.sess=sess
        self.model_fn=model_fn
        self.eval_input_fn=eval_input_fn
        self.v_rate=v_rate
        self.c_rate=c_rate
        self.best_num=best_num
        self.ep=ep
        self.population_num=population_num
        self.prune_rate=prune_rate
        self.gen_num=len(self.nonzero_pos)
        self.max_nonzero_num=self.gen_num*(1-prune_rate)
        self.bests = []
        self.fit_file_name=fit_file_name
        self.best_file_name=best_file_name
        self.out_gen=out_gen
        self.eval_steps=eval_steps
        self.meta_name = meta_name
        self.model_p = model_p

    def prune(self,old_mask,nonzero_pos,prune_num):#
        if(prune_num<=0):
            return old_mask
        mask=np.copy(old_mask)
        sample_pos=sample_no_back(nonzero_pos, prune_num)
        for pos in range(len(sample_pos)):
            mask[sample_pos[pos]] = 0.
        return mask

    def set_activate(self,old_mask,zero_pos,activate_num):#
        if(activate_num<=0):
            return old_mask
        mask=np.copy(old_mask)
        sample_pos=sample_no_back(zero_pos, activate_num)
        for pos in range(len(sample_pos)):
            mask[sample_pos[pos]] = 1.
        return mask

    def variation(self,mask):#要重写，交换mask的0，1位置
        cnt = int(self.gen_num * self.prune_rate)
        if(cnt<=0):
            cnt=1
        return self.prune(mask,self.nonzero_pos,cnt)

    def cross(self,mask1,mask2):
        #change_num=random.randint(1,self.gen_num)
        mask_and = mask1!=mask2
        nonzero_pos = get_nonzero_pos(mask_and)
        old_nonzero_num1 = np.sum(mask1)
        old_nonzero_num2 = np.sum(mask2)
        set_nonzero_num = max(old_nonzero_num1,old_nonzero_num2)
        num = len(nonzero_pos)-1
        if(num<=1):
            return None,None
        change_num = random.randint(1,num)
        cross_pos=sample_no_back(nonzero_pos,change_num)
        if len(cross_pos)<=1:
            return None,None

        for pos in cross_pos:
            temp = mask1[pos]
            mask1[pos] = mask2[pos]
            mask2[pos] = temp

        nonzero_num1 = np.sum(mask1)
        will_prune_num1 = nonzero_num1 - self.max_nonzero_num
        if(will_prune_num1>0):
            nonzero_pos1 = get_nonzero_pos(mask1)
            mask1 = self.prune(mask1,nonzero_pos1,will_prune_num1)
        elif(nonzero_num1<set_nonzero_num):
            zero_pos1 = get_zero_pos(mask1)
            activate_num = set_nonzero_num - nonzero_num1
            mask1 = self.set_activate(mask1,zero_pos1,activate_num)

        nonzero_num2 = np.sum(mask2)
        will_prune_num2 = nonzero_num2 - self.max_nonzero_num
        if(will_prune_num2>0):
            nonzero_pos2 = get_nonzero_pos(mask2)
            mask2 = self.prune(mask2,nonzero_pos2,will_prune_num1)
        elif(nonzero_num2<set_nonzero_num):
            zero_pos2 = get_zero_pos(mask2)
            activate_num = set_nonzero_num - nonzero_num2
            mask2 = self.set_activate(mask2,zero_pos2,activate_num)

        return mask1,mask2

    def first_population(self,old_mask):
        population=[]
        cnt = self.gen_num * self.prune_rate
        for i in range(self.population_num):
            v1=self.prune(old_mask,self.nonzero_pos,cnt)
            population.append(v1)
        return population


    def population_fitness(self,population,weight_name,old_ckpt):
        fits=[]
        stamps = []
        prune_fn = get_prune_fn(self.FLAGS.prune)()
        w_copy = copy.deepcopy(dict(self.orig_weights))
        old_w = copy.deepcopy(w_copy[weight_name])
        for i, p in enumerate(population):
            self.sess.run(tf.local_variables_initializer())
            w_copy[weight_name] = copy.deepcopy(old_w)
            post_weights_pruned, weight_counts = new_prune_weights(
                prune_fn,
                w_copy,
                weight_name,
                p)
            print(weight_counts)

            print("there are ", len(tf.trainable_variables()), " weights")
            for v in tf.trainable_variables():
                if is_prunable_weight(v) and v.name.strip(":0") == weight_name:
                    assign_op = v.assign(
                        np.reshape(post_weights_pruned[v.name.strip(":0")], v.shape))
                    self.sess.run(assign_op)
                    break
            for i in range(self.eval_steps):
                acc = self.sess.run(self.model_p.eval_metric_ops['acc'])
            print(acc[0])
            fits.append(acc[0])
            now = int(round(time.time() * 1000))
            stamps.append(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now / 1000)))
        #fits,stamps=ga_eval_model(FLAGS=self.FLAGS, hparam_name=self.hparam_name, hparams=self.hparams,weight_name=weight_name, sess=self.sess, model_fn=self.model_fn, eval_input_fn=self.eval_input_fn, population=population, orig_weights=self.orig_weights, meta_name=self.meta_name,eval_steps=self.eval_steps,old_ckpt=old_ckpt)

        return fits,stamps

    def evo(self,weight_name,old_ckpt,eval_input_fn,model_fn):
        gens = copy.deepcopy(self.out_gen)
        size = np.size(self.layer_old_mask)
        population=self.first_population(self.layer_old_mask)
        fits,stamps=self.population_fitness(population=population,weight_name=weight_name,old_ckpt=old_ckpt)
        v_rate = self.v_rate
        c_rate = self.c_rate

        for e in range(self.ep):
            tf.reset_default_graph()
            features, labels = eval_input_fn()
            gs = tf.train.get_or_create_global_step()
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            self.model_p = model_fn(features, labels, tf.estimator.ModeKeys.TRAIN)
            saver = tf.train.Saver()

            saver.restore(self.sess, old_ckpt)
            pickups_size = self.population_num+2
            npfits = np.array(fits)
            weights = np.multiply(npfits,npfits).tolist()
            pickups=sample_back(population, pickups_size, weights=weights)
            new_population=[]
            v_times=int(self.population_num*v_rate)
            for v in range(v_times):
                v1 = copy.deepcopy(pickups[v])
                v1= self.variation(v1)
                new_population.append(v1)
            c_times = int(self.population_num*c_rate/2.0)
            pickups_pos = v_times
            cnt = 0
            while(pickups_pos+1<pickups_size and cnt<c_times):
                c1 = copy.deepcopy(pickups[pickups_pos])
                c2 = copy.deepcopy(pickups[pickups_pos+1])
                c1,c2= self.cross(c1,c2)
                pickups_pos = pickups_pos + 2
                if(not c1 is None):
                    new_population.append(c1)
                    new_population.append(c2)
                    cnt = cnt + 1
            s_times=int(self.population_num*(1.0-v_rate-c_rate))
            cnt=0
            while (pickups_pos < pickups_size and cnt < s_times):
                s1=copy.deepcopy(pickups[pickups_pos])
                new_population.append(s1)
                pickups_pos = pickups_pos + 1

            fits,stamps=self.population_fitness(new_population,weight_name,old_ckpt=old_ckpt)
            w_fits=[]
            for k,f in enumerate(fits):
                zero_num = size - np.count_nonzero(new_population[k])
                w_fits.append(gens+[e,f,zero_num,size,stamps[k]])
            writer_to_csv(self.fit_file_name,w_fits)
            population=new_population
            self.bests,changed = select_bests(population,fits,self.bests,self.best_num)
            w_fits=[]
            for k in range(len(self.bests)):
                zero_num = size - np.count_nonzero(self.bests[k][1])
                w_fits.append(gens+[e,self.bests[k][0],zero_num,size])
            writer_to_csv(self.best_file_name, w_fits)
            #if(not changed):
            #    v_rate = v_rate * 1.2
            #    c_rate = min(c_rate,(1-v_rate))
        return self.bests,self.sess
    


    