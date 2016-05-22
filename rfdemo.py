import os,sys,pdb,cPickle
import numpy as np
import pandas as pd
import multiprocessing as mp
import cv2
import argparse
from config import g_config
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import datetime
import gc
def scan_for_train():
    label_list = 'c0,c1,c2,c3,c4,c5,c6,c7,c8,c9'.split(',')
    labels = []
    paths = []
    for label in label_list:
        root = os.path.join(g_config._train_root, label)
        objs = os.listdir( root )
        num = 0
        for obj in objs:
            fname = os.path.join( root, obj )
            if os.path.isfile(fname) and 0 == cmp(os.path.splitext(fname)[1],'.jpg'):
                paths.append(fname)
                labels.append(label)
                num += 1
    df = pd.DataFrame( {'path':paths, 'label':labels} )
    for label in label_list:
        df[label] = df.label == label
    df = df.drop('label', axis=1)
    return df

def scan_for_test():
    label_list = 'c0,c1,c2,c3,c4,c5,c6,c7,c8,c9'.split(',')
    objs = os.listdir( g_config._test_dir )
    paths = []
    num = 0
    for obj in objs:
        fname = os.path.join(g_config._test_dir, obj)
        if os.path.isfile(fname) and 0 == cmp(os.path.splitext(fname)[1],'.jpg'):
            paths.append(fname)
            num += 1
            if num > 100000:
                break
    print ''
    df = pd.DataFrame( {'path':paths} )
    for label in label_list:
        df[label] = 0
    return df

def gen_image_feature(path):
    img = cv2.imread(path,0)
    img = cv2.resize(img, (64,64) ).flatten().tolist()
    return img

def _gen_feature(df):
    results = df.apply(lambda X: gen_image_feature(X['path']), axis = 1)
    return list(results) 

def gen_feature(df):
    workers = np.minimum(mp.cpu_count(), df.shape[0])
    pool = mp.Pool(workers)
    results = pool.map(_gen_feature, np.array_split(df,workers))
    pool.close()
    feats = []
    for res in results:
        feats.extend(res)
    return feats
    

def train():
    label_list = 'c0,c1,c2,c3,c4,c5,c6,c7,c8,c9'.split(',')
    print "load train set"
    df = scan_for_train()

    print "gen feature for train set"
    X = gen_feature(df)
    clf_list = []

    for label in label_list:
        print "train for %s"%label
        clf = RandomForestClassifier(n_jobs=mp.cpu_count(), n_estimators = 100)
        clf.fit(X, df[label])
        clf_list.append(clf)

    with open(os.path.join(g_config._tmpdir, 'rf.pkl'), 'wb') as f:
        cPickle.dump(clf_list, f)

def _test(args):
    label_list = 'c0,c1,c2,c3,c4,c5,c6,c7,c8,c9'.split(',')
    clfpath, X, name_list = args
    with open(os.path.join(clfpath), 'rb') as f:
        clf_list = cPickle.load(f)
    results = []
    for label, clf in zip(label_list,clf_list):
        results.append(clf.predict_proba(X)[:,1])
    return (results,name_list)

def test():
    label_list = 'c0,c1,c2,c3,c4,c5,c6,c7,c8,c9'.split(',')
    print 'load test set'
    df = scan_for_test()
    print 'gen feature for test set'

    df_list = np.array_split(df,10)
    del df
    gc.collect()

    img_list = []
    results = [[] for k in range(len(label_list))]
    df_offset = 0
    for groupidx,df in enumerate(df_list):
        X = gen_feature(df)
        print 'run clf on group %d'%groupidx
        workers = np.minimum(mp.cpu_count(), len(X))
        X_list = [[] for k in range(workers)]
        name_list = [[] for k in range(workers)]
        idx = 0
        for k in range(len(X)):
            X_list[idx].append( X[k] )
            name_list[idx].append( df.path[k + df_offset].split('\\')[-1] )
            idx  += 1
            if idx >= workers:
                idx = 0
        df_offset += df.shape[0]

        clf_path = os.path.join(g_config._tmpdir, 'rf.pkl')
        pool = mp.Pool(workers)
        ret_list = pool.map( _test, [(clf_path, Xs, names) for Xs, names in zip(X_list, name_list)])
        pool.close()

        for ret in ret_list:
            for k in range(len(label_list)):
                results[k].extend(ret[0][k])
            img_list.extend(ret[1])

    c0,c1,c2,c3,c4,c5,c6,c7,c8,c9 = results
    sub = pd.DataFrame({'img':img_list,
    'c0':c0, 'c1':c1, 'c2':c2, 'c3':c3, 'c4':c4, 'c5':c5, 'c6':c6,
    'c7':c7, 'c8':c8, 'c9':c9})
    sub = sub.sort('img')
    resultfile =  'submission_%s.csv'%datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    resultfile = os.path.join( g_config._submission_dir, resultfile)
    sub['img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9'.split(',')].to_csv(resultfile, index=False)
    print 'submisson.csv done!'

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('mode',help='train/test')
    args = ap.parse_args()
    if 0 == cmp(args.mode,'train'):
        train()
    elif 0 == cmp(args.mode, 'test'):
        test()
    else:
        print 'error option'




