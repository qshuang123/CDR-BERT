import os

import keras
from keras import Input, Model, optimizers
from keras.activations import softmax
from keras.backend import batch_dot, tf
from keras.models import load_model
from keras.layers import Dense, Dropout, Bidirectional, GRU, Embedding, TimeDistributed, GlobalMaxPooling1D, \
    concatenate, Lambda, Dot, Permute, Concatenate, Multiply, Add
from keras.optimizers import RMSprop
from sklearn.metrics import f1_score
import pickle
import numpy as np
from keras import backend as K
import logging
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
from utils.batch_gather import batch_gather
from keras.callbacks import Callback
logging.basicConfig(level=logging.INFO, filename='bert_baseline_hua.log')
from utils.CyclicLR import CyclicLR
from utils.threshold import threshold_search, f1, count, load_pkl, save_file
from keras_bert import load_vocabulary, load_trained_model_from_checkpoint, Tokenizer


pretrained_path = 'scibert_scivocab_uncased'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

batch_size = 16

max_sentence_length=512
bert_out_shape = 768
learning_rate = 5e-5
min_learning_rate = 1e-5

# def unchanged_shape(input_shape):
#     """Function for Lambda layer"""
#     return input_shape
#
#
# def soft_attention_alignment(input_1, input_2):
#     """Align text representation with neural soft attention"""
#
#     attention = Dot(axes=-1)([input_1, input_2])
#
#     w_att_1 = Lambda(lambda x: softmax(x, axis=1),
#                      output_shape=unchanged_shape)(attention)
#     w_att_2 = Lambda(lambda x: softmax(x, axis=2), output_shape=unchanged_shape)(attention)
#     w_att_2 = Permute((2, 1))(w_att_2)
#     in1_aligned = Dot(axes=1)([w_att_1, input_1])
#     in2_aligned = Dot(axes=1)([w_att_2, input_2])
#     return in1_aligned, in2_aligned
#
#
# def entity_attention(input):
#     """Align text representation with neural soft attention"""
#
#     pos = K.reshape(input[0], (-1, max_sentence_number, max_sentence_length, 1))
#     pos = K.repeat_elements(pos, 512, -1)
#     # 对200求最大
#     result = pos*input[1]
#     result1 = K.reshape(result, (-1, max_sentence_length, 512))
#     result1 = GlobalMaxPooling1D()(result1)
#     result2 = K.reshape(result1,(-1,max_sentence_number,512))
#     # result2 = GlobalMaxPooling1D()(result2)
#     return result2
#
# def entity_attention_output_shape(input_shape):
#     shape = list(input_shape[0])
#     shape[2] = 512
#     return tuple(shape)

#
# def get_context(input,key):
#
#     entity = GlobalMaxPooling1D()(input[1])
#     con = K.reshape(input[0], (-1, max_sentence_length*max_sentence_number, 512))
#     atten = K.batch_dot(entity, con, axes=[1, 2])
#     atten_topk, topk_pos = K.tf.nn.top_k(atten, key)
#     print("attentopk", atten_topk)
#     print("topk_pos", topk_pos)
#     context = batch_gather(con, topk_pos)
#     top_atten = K.softmax(atten_topk)
#     result1 = K.batch_dot(top_atten, context, axes=[1, 1])
#     return result1
#
#
# def get_context_output_shape(input_shape):
#     shape = list(input_shape[0])
#     shape[1] = 512
#     shape.pop(2)
#     shape.pop(2)
#     assert len(shape) == 2
#     return tuple(shape)


class data_generator:
    def __init__(self, data, batch_size=batch_size):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            # X_indices, X_segments, X_pos1, X_pos2, Y = [], [], [], [], []
            X_indices, X_segments,Y = [], [], []
            for i in idxs:
                d = self.data[i]
                indices = d[0]
                segments = d[1]
                # pos1 = d[2]
                # pos2 = d[3]
                X_indices.append(indices)
                X_segments.append(segments)
                # X_pos1.append(pos1)
                # X_pos2.append(pos2)
                Y.append(d[2])
                if len(X_indices) == self.batch_size or i == idxs[-1]:
                    # yield [np.array(X_indices), np.array(X_segments), np.array(X_pos1), np.array(X_pos2)], np.array(Y)
                    # [X_indices, X_segments, X_pos1, X_pos2, Y] = [], [], [], [], []
                    yield [np.array(X_indices), np.array(X_segments)], np.array(Y)
                    [X_indices, X_segments, Y] = [], [], []

class Evaluate(Callback):
    def __init__(self):

        self.best = 0.
        self.passed = 0
        self.stage = 0
    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """

        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1

    def on_epoch_end(self, epoch, logs=None):
        print("pridict_dev")
        # pred_dev_y = model.predict([dev_x_indices, dev_x_segments, dev_entity1_pos, dev_entity2_pos],
        #                            batch_size=batch_size, verbose=0)
        pred_dev_y = model.predict([dev_x_indices, dev_x_segments],
                                   batch_size=batch_size, verbose=0)
        # best_threshold, best_scores = threshold_search(y_develop,pred_dev_y)
        precision, recall, best_score = f1(dev_y, (pred_dev_y > 0.5).astype(int))
        print("epoch###################################################################################################")
        print("-    dev F1 Score: {:.4f}".format(best_score))
        logging.info("-    dev F1 Score: " + str(best_score))
        # pred_test_y = model.predict([test_x_indices, test_x_segments, test_entity1_pos, test_entity2_pos],
        #                             batch_size=batch_size, verbose=0)
        pred_test_y = model.predict([test_x_indices, test_x_segments],
                                    batch_size=batch_size, verbose=0)
        p, r, f = f1(test_y, (pred_test_y > 0.5).astype(int))
        print("-    Val F1 Score: {:.4f}".format(f))
        logging.info("-    Val p Score:" + str(p))
        logging.info("-    Val r Score:" + str(r))
        logging.info("-    Val F1 Score:" + str(f))
        if f > 0.6:
            save_file("bert_hua_" + str(f) + "_predict", pred_test_y)
            model.save('./models/bert_hua_' + str(f) + '.h5')


def bert_cea_model(drop,k):
    inputs_indices = Input(shape=(max_sentence_length,), name='inputs_indices')
    print(inputs_indices.shape)
    inputs_segment = Input(shape=(max_sentence_length,), name='inputs_segment')
    print(inputs_segment.shape)
    # position1 = Input(shape=(max_sentence_number, max_sentence_length,), name='position1')
    # position2 = Input(shape=(max_sentence_number, max_sentence_length,), name='position2')
    # entity1_mask = Input(shape=(max_sentence_number, max_sentence_length,), name='entity1_mask')
    # print(entity1_mask.shape)
    # entity2_mask = Input(shape=(max_sentence_number, max_sentence_length,), name='entity2_mask')
    # print(entity2_mask.shape)

    # indices_reshape = Lambda(lambda x: tf.reshape(x, [-1, max_sentence_length]))(inputs_indices)  # B*23,452
    # segment_reshape = Lambda(lambda x: tf.reshape(x, [-1, max_sentence_length]))(inputs_segment)  # B*23,452
    context_vector = bert_model([inputs_indices, inputs_segment])
    for l in bert_model.layers:
        l.trainable = True
    x = Lambda(lambda x: x[:, 0])(context_vector)
    # x = Lambda(lambda x: tf.reshape(x, [-1, max_sentence_number, max_sentence_length, bert_out_shape]))(context_vector)

    # lstm = TimeDistributed(Bidirectional(GRU(256, return_sequences=True, dropout=drop)))(x)
    # print(lstm.shape)

    # entity1_lstm = Lambda(entity_attention, output_shape=entity_attention_output_shape)([entity1_mask, lstm])
    # entity2_lstm = Lambda(entity_attention, output_shape=entity_attention_output_shape)([entity2_mask, lstm])

    # entity1_atten, entity2_atten = soft_attention_alignment(entity1_lstm, entity2_lstm)
    #
    # entity1_out = GlobalMaxPooling1D()(entity1_atten)
    # entity2_out = GlobalMaxPooling1D()(entity2_atten)
    #
    # context1_rep = Lambda(get_context, output_shape=get_context_output_shape, arguments={'key': k})([lstm, entity1_lstm])
    # context2_rep = Lambda(get_context, output_shape=get_context_output_shape, arguments={'key': k})([lstm, entity2_lstm])

    # res = concatenate([context1_rep,context2_rep, entity1_out, entity2_out])
    res = Dense(64)(x)
    res = Dropout(drop)(res)

    out = Dense(1, activation='sigmoid', name='output')(res)
    # model = Model([inputs_indices, inputs_segment, entity1_mask, entity2_mask], out)
    model = Model([inputs_indices, inputs_segment], out)
    model.summary()
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":

    # train_entity1_pos = load_pkl("train_pos1")
    # dev_entity1_pos = load_pkl("dev_pos1")
    # test_entity1_pos = load_pkl("test_pos1")
    # train_entity2_pos = load_pkl("train_pos2")
    # dev_entity2_pos = load_pkl("dev_pos2")
    # test_entity2_pos = load_pkl("test_pos2")

    train_x_indices = load_pkl('train_indices')
    train_x_segments = load_pkl('train_segments')
    train_y = load_pkl('train_y')
    dev_x_indices = load_pkl('dev_indices')
    dev_x_segments = load_pkl('dev_segments')
    dev_y = load_pkl('dev_y')
    test_x_indices = load_pkl('test_indices')
    test_x_segments = load_pkl('test_segments')
    test_y = load_pkl('tests_y')

    train = []
    for i in range(len(train_x_indices)):
        temp=[]
        temp.append(train_x_indices[i])
        temp.append(train_x_segments[i])
        # temp.append(train_entity1_pos[i])
        # temp.append(train_entity2_pos[i])
        temp.append(train_y[i])
        train.append(temp)

    train_D = data_generator(train)
    evaluator = Evaluate()

    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    top_k = [100,200,300,400,500]
    # for k in top_k:
    k=400
    drop = 0.3
    model = bert_cea_model(drop, k)

    model.fit_generator(train_D.__iter__(),
                              steps_per_epoch=610,
                              epochs=50,
                              callbacks=[evaluator])







