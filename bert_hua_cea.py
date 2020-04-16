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
logging.basicConfig(level=logging.INFO, filename='bert_hua_cea.log')
from utils.CyclicLR import CyclicLR
from utils.threshold import threshold_search, f1, count, load_pkl, save_file
from keras_bert import load_vocabulary, load_trained_model_from_checkpoint, Tokenizer


pretrained_path = 'scibert_scivocab_uncased'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

batch_size = 16
bert_out_shape = 768
max_sentence_length=512
bert_out_shape = 768
learning_rate = 5e-5
min_learning_rate = 1e-5


def unchanged_shape(input_shape):
    """Function for Lambda layer"""
    return input_shape


def soft_attention_alignment(input_1, input_2, attention_mask):
    """Align text representation with neural soft attention"""

    attention_scores = Dot(axes=-1)([input_1, input_2])
    attention_scores = keras.layers.Add()([attention_mask, attention_scores])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention_scores)
    w_att_2 = Lambda(lambda x: softmax(x, axis=2), output_shape=unchanged_shape)(attention_scores)
    w_att_2 = Permute((2, 1))(w_att_2)
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])

    return in1_aligned, in2_aligned


def entity_extract(input):
    """Align text representation with neural soft attention"""

    pos = K.reshape(input[1], (-1,  max_sentence_length, 1))
    pos = K.repeat_elements(pos, bert_out_shape, -1)
    result = pos*input[0]
    return result


def entity_extract_output_shape(input_shape):
    shape = list(input_shape[0])
    return tuple(shape)


def attention_mask_fun(input):
    """Align text representation with neural soft attention"""

    mask1 = K.reshape(input[0], (-1, max_sentence_length, 1))
    mask2 = K.reshape(input[1], (-1, max_sentence_length, 1))
    result = keras.layers.Dot(axes=-1)([mask1, mask2])
    result = (1.0 - tf.cast(result, tf.float32)) * -100000.0

    return result


def attention_mask_output_shape(input_shape):
    shape = list(input_shape[0])
    shape.append(1)
    shape[2] = max_sentence_length
    return tuple(shape)


def get_context(input,key):

    entity = GlobalMaxPooling1D()(input[1])
    atten = K.batch_dot(entity, input[0], axes=[1, 2])
    atten_topk, topk_pos = K.tf.nn.top_k(atten, key)
    print("attentopk", atten_topk)
    print("topk_pos", topk_pos)
    context = batch_gather(input[0], topk_pos)
    top_atten = K.softmax(atten_topk)
    result1 = K.batch_dot(top_atten, context, axes=[1, 1])
    return result1


def get_context_output_shape(input_shape):
    shape = list(input_shape[0])
    shape[1] = bert_out_shape
    shape.pop(2)
    assert len(shape) == 2
    return tuple(shape)


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
            X_indices, X_segments, X_pos1, X_pos2, Y = [], [], [], [], []

            for i in idxs:
                d = self.data[i]
                indices = d[0]
                segments = d[1]
                pos1 = d[2]
                pos2 = d[3]
                X_indices.append(indices)
                X_segments.append(segments)
                X_pos1.append(pos1)
                X_pos2.append(pos2)
                Y.append(d[4])
                if len(X_indices) == self.batch_size or i == idxs[-1]:
                    yield [np.array(X_indices), np.array(X_segments), np.array(X_pos1), np.array(X_pos2)], np.array(Y)
                    [X_indices, X_segments, X_pos1, X_pos2, Y] = [], [], [], [], []

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
        pred_dev_y = model.predict([dev_x_indices, dev_x_segments, dev_entity1_pos, dev_entity2_pos],
                                   batch_size=batch_size, verbose=0)

        # best_threshold, best_scores = threshold_search(y_develop,pred_dev_y)
        precision, recall, best_score = f1(dev_y, (pred_dev_y > 0.5).astype(int))
        print("epoch###################################################################################################")
        print("-    dev F1 Score: {:.4f}".format(best_score))
        logging.info("-    dev F1 Score: " + str(best_score))
        pred_test_y = model.predict([test_x_indices, test_x_segments, test_entity1_pos, test_entity2_pos],
                                    batch_size=batch_size, verbose=0)

        p, r, f = f1(test_y, (pred_test_y > 0.5).astype(int))
        print("-    Val F1 Score: {:.4f}".format(f))
        logging.info("-    Val p Score:" + str(p))
        logging.info("-    Val r Score:" + str(r))
        logging.info("-    Val F1 Score:" + str(f))
        if f > 0.6:
            save_file("bert_hua_cea_" + str(f) + "_predict", pred_test_y)
            model.save('./models/bert_hua_cea_' + str(f) + '.h5')


def bert_gea_model(drop):
    inputs_indices = Input(shape=(max_sentence_length,), name='inputs_indices')
    inputs_segment = Input(shape=(max_sentence_length,), name='inputs_segment')

    entity1_mask = Input(shape=(max_sentence_length,), name='entity1_mask', )
    entity2_mask = Input(shape=(max_sentence_length,), name='entity2_mask')

    context_vector = bert_model([inputs_indices, inputs_segment])
    for l in bert_model.layers:
        l.trainable = True

    # CLS_encode = Lambda(lambda x: x[:, 0])(context_vector)

    entity1_encodes = Lambda(entity_extract, output_shape=entity_extract_output_shape)([context_vector, entity1_mask])
    entity2_encodes = Lambda(entity_extract, output_shape=entity_extract_output_shape)([context_vector, entity2_mask])

    attention_mask = Lambda(attention_mask_fun, output_shape=attention_mask_output_shape)([entity1_mask, entity2_mask])

    entity1_atten, entity2_atten = soft_attention_alignment(entity1_encodes, entity2_encodes, attention_mask)

    entity1_vector = GlobalMaxPooling1D()(entity1_atten)
    entity2_vector = GlobalMaxPooling1D()(entity2_atten)

    context1_rep = Lambda(get_context, output_shape=get_context_output_shape, arguments={'key': k})(
        [context_vector, entity1_encodes])
    context2_rep = Lambda(get_context, output_shape=get_context_output_shape, arguments={'key': k})(
        [context_vector, entity2_encodes])

    res = concatenate([context1_rep,context2_rep,entity1_vector,entity2_vector])

    res = Dense(64)(res)
    res = Dropout(drop)(res)

    out = Dense(1, activation='sigmoid', name='output')(res)
    model = Model([inputs_indices, inputs_segment, entity1_mask, entity2_mask], out)

    model.summary()
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":

    train_entity1_pos = load_pkl("train_pos1")
    dev_entity1_pos = load_pkl("dev_pos1")
    test_entity1_pos = load_pkl("test_pos1")
    train_entity2_pos = load_pkl("train_pos2")
    dev_entity2_pos = load_pkl("dev_pos2")
    test_entity2_pos = load_pkl("test_pos2")

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
        temp.append(train_entity1_pos[i])
        temp.append(train_entity2_pos[i])
        temp.append(train_y[i])
        train.append(temp)

    train_D = data_generator(train)
    evaluator = Evaluate()

    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    top_k = [100,200,300,400,500]
    # for k in top_k:
    k = 30
    drop = 0.3
    model = bert_gea_model(drop)

    model.fit_generator(train_D.__iter__(),
                              steps_per_epoch=610,
                              epochs=50,
                              callbacks=[evaluator])




