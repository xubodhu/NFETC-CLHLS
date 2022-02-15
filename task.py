from utils import data_utils, embedding_utils, pkl_utils
from utils.eval_utils import strict, loose_macro, loose_micro, label_path
import numpy as np
from sklearn.model_selection import ShuffleSplit
import os
import config
import pickle
import tensorflow as tf
from nfetc_ls import NFETC_LS

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Task:
    def __init__(self, model_name, data_name, cv_runs, params_dict, logger, portion=100, save_name=''):
        print('Loading data...')
        if portion <= 100:  # all the data, portion% clean + all noisy
            self.portion = '-' + str(portion) if portion != 100 else ''

        print('run task on: ', self.portion, ' dataset: ', data_name)
        self.data_name = data_name
        if data_name == 'ontonotes':
            words_train, mentions_train, positions_train, labels_train,one_label_ids,multi_label_ids = data_utils.load(
                config.ONTONOTES_TRAIN_CLEAN + self.portion,params_dict['filter'])
            words, mentions, positions, labels = data_utils.load(config.ONTONOTES_TEST_CLEAN)
            type2id, typeDict = pkl_utils.load(config.ONTONOTES_TYPE)
            num_types = len(type2id)
            type_info = config.ONTONOTES_TYPE
        elif data_name == 'bbn':
            words_train, mentions_train, positions_train, labels_train,one_label_ids,multi_label_ids = data_utils.load(
                config.BBN_TRAIN_CLEAN + self.portion,params_dict['filter'])
            words, mentions, positions, labels = data_utils.load(config.BBN_TEST_CLEAN)
            type2id, typeDict = pkl_utils.load(config.BBN_TYPE)
            num_types = len(type2id)
            type_info = config.BBN_TYPE
        elif data_name == 'wikim':
            words_train, mentions_train, positions_train, labels_train,one_label_ids,multi_label_ids = data_utils.load(
                config.WIKIM_TRAIN_CLEAN+ self.portion,params_dict['filter'])
            words, mentions, positions, labels = data_utils.load(config.WIKIM_TEST_CLEAN)
            type2id, typeDict = pkl_utils.load(config.WIKIM_TYPE)
            num_types = len(type2id)
            type_info = config.WIKIM_TYPE
        else:
            assert False, 'you have to specify the name of dataset with -d (ie. bbn/....)'
        self.model_name = model_name
        self.savename = save_name
        self.data_name = data_name
        self.cv_runs = cv_runs
        self.num_classes = len(type2id)
        self.params_dict = params_dict
        self.hparams = AttrDict(params_dict)
        self.logger = logger
        self.batch_size = self.hparams.batch_size
        self.num_epochs = self.hparams.num_epochs

        self.id2type = {type2id[x]: x for x in type2id.keys()}

        def type2vec(types):  # only terminal will be labeled
            tmp = np.zeros(num_types)
            for t in str(types).split():
                if t in type2id.keys():
                    tmp[type2id[t]] = 1.0
            return tmp


        labels_train = np.array([type2vec(t) for t in labels_train])
        labels = np.array([type2vec(t) for t in labels])
        train_id = np.array(range(len(labels_train))).astype(np.int32)

        tempname = self.data_name + config.testemb
        tempname = os.path.join(config.PKL_DIR, tempname)
        print('Temp name: ',tempname)
        if os.path.exists(tempname):
            self.embedding = pickle.load(open(tempname, 'rb'))
            print('embedding load over')
        else:
            self.embedding = embedding_utils. \
                Embedding.fromCorpus(config.EMBEDDING_DATA, list(words_train) + list(words),
                                     config.MAX_DOCUMENT_LENGTH, config.MENTION_SIZE)
            pickle.dump(self.embedding, open(tempname, 'wb'))
            print('embedding dump over')
        self.embedding.max_document_length = config.MAX_DOCUMENT_LENGTH

        print('Preprocessing data...')
        #
        self.one_label_ids = one_label_ids
        self.multi_label_ids = multi_label_ids
        self.warm_epochs = self.hparams.warm_epochs
        self.e_1 = self.hparams.e_1
        self.e_2 = self.hparams.e_2
        print('one label rate: ', len(self.one_label_ids) / labels_train.shape[0])
        print('multi label rate: ', len(self.multi_label_ids) / labels_train.shape[0])
        #
        if True:
            textlen_train = np.array(
                [self.embedding.len_transform1(x) for x in words_train])  # with cut down len sequence
            words_train = np.array([self.embedding.text_transform1(x) for x in
                                    words_train])  # with cut down word id sequence and mask with zero <PAD>
            mentionlen_train = np.array([self.embedding.len_transform2(x) for x in mentions_train])  # mention len
            mentions_train = np.array(
                [self.embedding.text_transform2(x) for x in mentions_train])  # mention text indexer
            positions_train = np.array(
                [self.embedding.position_transform(x) for x in positions_train])  # start ,end position
            print('get train data')

            textlen = np.array([self.embedding.len_transform1(x) for x in words])
            words = np.array([self.embedding.text_transform1(x) for x in words])  # padding and cut down
            mentionlen = np.array([self.embedding.len_transform2(x) for x in mentions])
            mentions = np.array([self.embedding.text_transform2(x) for x in mentions])
            positions = np.array([self.embedding.position_transform(x) for x in positions])
            print('get test data')

        ss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=config.RANDOM_SEED)
        for test_index, valid_index in ss.split(np.zeros(len(labels)), labels):
            textlen_test, textlen_valid = textlen[test_index], textlen[valid_index]
            words_test, words_valid = words[test_index], words[valid_index]
            mentionlen_test, mentionlen_valid = mentionlen[test_index], mentionlen[valid_index]
            mentions_test, mentions_valid = mentions[test_index], mentions[valid_index]
            positions_test, positions_valid = positions[test_index], positions[valid_index]
            labels_test, labels_valid = labels[test_index], labels[valid_index]

        self.train_set = list(
            zip(words_train, textlen_train, mentions_train, mentionlen_train, positions_train, labels_train, train_id))
        self.valid_set = list(
            zip(words_valid, textlen_valid, mentions_valid, mentionlen_valid, positions_valid, labels_valid, ))
        self.test_set = list(
            zip(words_test, textlen_test, mentions_test, mentionlen_test, positions_test, labels_test, ))

        self.full_test_set = list(zip(words, textlen, mentions, mentionlen, positions, labels, ))

        self.labels_train = labels_train
        self.labels_test = labels_test
        self.labels = labels
        #self.labels_train_s = labels_train_s
        self.labels_valid = labels_valid

        self.num_types = num_types
        self.num_train = len(labels_train)
        self.type_info = type_info
        self.logger.info('train set size:%d, test set size: %d' % (len(self.train_set), len(self.full_test_set)))

        self.model = self._get_model()
        self.saver = tf.train.Saver(tf.global_variables())
        checkpoint_dir = os.path.abspath(config.CHECKPOINT_DIR)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.checkpoint_prefix = os.path.join(checkpoint_dir, self.__str__())

    def __str__(self):
        return self.model_name + self.savename

    def _get_model(self):
        np.random.seed(config.RANDOM_SEED)
        kwargs = {
            'sequence_length': config.MAX_DOCUMENT_LENGTH,
            'mention_length': config.MENTION_SIZE,
            'num_classes': self.num_types,
            'vocab_size': self.embedding.vocab_size,
            'embedding_size': self.embedding.embedding_dim,
            'position_size': self.embedding.position_size,
            'pretrained_embedding': self.embedding.embedding,
            'wpe': np.random.random_sample((self.embedding.position_size, self.hparams.wpe_dim)),
            'type_info': self.type_info,
            'num_train': self.num_train,
            'hparams': self.hparams,
            'dataset': self.data_name
        }
        return NFETC_LS(**kwargs)

    def _print_param_dict(self, d, prefix='      ', incr_prefix='      '):
        for k, v in sorted(d.items()):
            if isinstance(v, dict):
                self.logger.info('%s%s:' % (prefix, k))
                self.print_param_dict(v, prefix + incr_prefix, incr_prefix)
            else:
                self.logger.info('%s%s: %s' % (prefix, k, v))

    def create_session(self):
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=8,
            allow_soft_placement=True,
            log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        return tf.Session(config=session_conf)

    def get_scores(self, preds, target='test'):
        preds = [label_path(self.id2type[x]) for x in preds]

        # print(self.test_set[0])
        def vec2type(v):
            s = []
            for i in range(len(v)):
                if v[i]:
                    s.extend(label_path(self.id2type[i]))
            return set(s)

        print('eval on ', target)
        if target == 'test':
            labels_test = [vec2type(x) for x in self.labels_test]  # path will caculate the father node for strict acc
        else:
            labels_test = [vec2type(x) for x in self.labels_valid]
        words = [self.embedding.i2w(k[0]) for k in self.full_test_set]
        mentions = [self.embedding.i2w(k[2]) for k in self.full_test_set]
        acc = strict(labels_test, preds, oridata=(words, mentions), modelname=self.savename)
        _, _, macro = loose_macro(labels_test, preds)
        _, _, micro = loose_micro(labels_test, preds)
        return acc, macro, micro

    def rate_decay(self,i,epochs,mini,mx):
        now_rate = mx - ((mx-mini)/epochs)*i
        if now_rate<-1e-3:
            return 'error'
        else:
            return now_rate
    def rate_grow(self,i,epochs,mini,mx):
        now_rate = mini + ((mx-mini)/epochs)*i
        if now_rate<-1e-3:
            return 'error'
        else:
            return now_rate

    def update_loss(self,s_loss,epoch_loss,epoch_ids):
        gamma = 0.2 #EMA loss discount factor
        for i in range(epoch_ids.shape[0]):
            idx = epoch_ids[i]
            if s_loss[idx] ==0.0:
                s_loss[idx] = epoch_loss[i]
            else:
                s_loss[idx] = s_loss[idx]*(1-gamma)+epoch_loss[i]*gamma
        return s_loss

    def get_select_ids(self, s_loss):
        order_loss = np.sort(s_loss, kind='mergesort')
        split_idx = int(s_loss.shape[0] * self.hparams.save_rate)
        split_value = order_loss[split_idx]
        select_ids = []
        for i in range(len(self.train_set)):
            if s_loss[i] == 0.0 or s_loss[i] < 1e-19:
                continue
            if s_loss[i] <= split_value:
                select_ids.append(i)
        self.logger.info('one label loss:\t%.5f\tmulti label loss:\t%.5f\tAll label loss:\t%.5f' %
                         (float(np.mean(s_loss[self.one_label_ids])), float(np.mean(s_loss[self.multi_label_ids])),
                          float(np.mean(s_loss))))
        return select_ids

    def refit(self):
        self.logger.info('Params')
        self._print_param_dict(self.params_dict)
        self.logger.info('Evaluation for each epoch')
        self.logger.info('\t\tEpoch\t\tAcc\t\tMacro\t\tMicro\t\tTAcc\t\tTMacro\t\tTMicro')
        sess = self.create_session()
        
        print('retraining times: ', self.cv_runs)
        sess.run(tf.global_variables_initializer())

        max_base_on_valid = ()

        va_acc_list = []
        va_macro_list = []
        va_micro_list = []

        ls_rate = self.model.label_smoothing
        ancestor_rate = self.model.ancestor_rate
        ls_mini_rate= 0.0
        #anc_mini_rate= 0.1

        for i in range(self.cv_runs):
            if self.cv_runs > 1 and i != 0:
                print('reopen sess...')
                sess.close()
                sess = self.create_session()
                sess.run(tf.global_variables_initializer())
            max_va_acc = -1
            all_losses =np.zeros(len(self.train_set))
            #
            self.model.label_smoothing = ls_rate
            self.model.ancestor_rate =ancestor_rate
            for epoch in range(1, self.num_epochs + 1):
                if epoch<=self.warm_epochs:
                    train_batches = data_utils.batch_iter(self.train_set, self.batch_size, 1,self.one_label_ids+self.multi_label_ids)
                    epoch_ids, epoch_losses = self.model.evaluate(sess, train_batches)
                    all_losses = self.update_loss(all_losses,epoch_losses,epoch_ids)
                elif epoch<=self.e_1:
                    num_batches_per_epoch = int((len(self.one_label_ids) - 1) / self.batch_size) + 1
                    for batch_num in range(num_batches_per_epoch):
                        train_batch = data_utils.batch_iter2(self.train_set, self.batch_size, 1, self.one_label_ids,
                                                             all_losses)
                        batch_ids, batch_losses = self.model.evaluate_batch(sess, train_batch)
                        all_losses = self.update_loss(all_losses, batch_losses, batch_ids)

                else:
                    if epoch<=self.e_2:
                        self.model.label_smoothing = self.rate_decay(epoch-self.e_1, self.e_2-self.e_1, ls_mini_rate, ls_rate)

                    num_batches_per_epoch = int((len(self.one_label_ids) - 1) / self.batch_size) + 1
                    for batch_num in range(num_batches_per_epoch):
                        train_batch = data_utils.batch_iter2(self.train_set, self.batch_size, 1, self.one_label_ids, all_losses)
                        batch_ids, batch_losses = self.model.evaluate_batch(sess, train_batch)
                        all_losses = self.update_loss(all_losses, batch_losses, batch_ids)


                preds, maxtype = self.model.predict(sess, self.test_set)
                acc, macro, micro = self.get_scores(maxtype)
                vapreds, _ = self.model.predict(sess, self.valid_set)
                vaacc, vamacro, vamicro = self.get_scores(_, target='vatestset')
                cmp = vaacc
                #
                if cmp >= max_va_acc:
                    max_va_acc = cmp
                    max_base_on_valid = (epoch, acc, macro, micro, max_va_acc)
                    self.logger.info(
                        '\tep\t%d\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f' %
                        (epoch, vaacc, vamacro, vamicro, acc, macro, micro, max_va_acc,self.model.label_smoothing,self.model.ancestor_rate))
                else:
                    self.logger.info(
                        '\tep\t%d\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t-.---\t\t%.3f\t\t%.3f' %
                        (epoch, vaacc, vamacro, vamicro, acc, macro, micro,self.model.label_smoothing,self.model.ancestor_rate))

            va_acc_list.append(max_base_on_valid[1])
            va_macro_list.append(max_base_on_valid[2])
            va_micro_list.append(max_base_on_valid[3])
            self.logger.info('\tMax\t%d\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f' % (
                max_base_on_valid[0], max_base_on_valid[1], max_base_on_valid[2], max_base_on_valid[3], max_base_on_valid[4]))

        meanvaacc = np.mean(va_acc_list)
        meanvamacro = np.mean(va_macro_list)
        meanvamicro = np.mean(va_micro_list)
        stdvaacc = np.std(va_acc_list)
        stdvamacro = np.std(va_macro_list)
        stdvamicro = np.std(va_micro_list)

        self.logger.info('\tCV\t{:.1f}±{:.1f}\t{:.1f}±{:.1f}\t{:.1f}±{:.1f}'.format(meanvaacc * 100, stdvaacc * 100,
                                                                                    meanvamacro * 100, stdvamacro * 100,
                                                                                    meanvamicro * 100,
                                                                                    stdvamicro * 100))
        sess.close()
        tf.reset_default_graph()