# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.model_fn import EstimatorSpec
import os
import modeling
from utils import *
import config
import tensorflow as tf
import pickle
import re
'''
config includes two dictionary

model_base_dir means your finetuned bert model dir
label_path needed if your task is seq tagging
allow_growth is gpu options
'''
model_base = config.model_config['model_base_dir']
label_path  = config.model_config['label_path']
max_seq_length = config.predict_config['max_seq_length']
allow_growth = config.predict_config['allow_growth']
output_dir = config.predict_config['output_dir']


def input_fn_builder(text):
    '''

    :param text: list of sentence; List length means batch size
    :return:For Estimator generator
    '''
    def gen():
        tokenizer = tokenization.PosTokenizer(vocab_file=os.path.join(model_base, 'vocab.txt'))
        is_tokenized = all(isinstance(el, list) for el in text)
        tmp_f = list(convert_lst_to_features(text, max_seq_length, tokenizer,
                                                         is_tokenized, False))
        yield {
            'input_ids': [f.input_ids for f in tmp_f],
            'input_mask': [f.input_mask for f in tmp_f],
            'segment_ids': [f.segment_ids for f in tmp_f]
        }

    def input_fn():
        return (tf.data.Dataset.from_generator(
            gen,
            output_types={'input_ids': tf.int32,
                          'input_mask': tf.int32,
                          'segment_ids': tf.int32},
            output_shapes={
                'input_ids': (None, max_seq_length),
                'input_mask': (None, max_seq_length),
                'segment_ids': (None, max_seq_length)}).prefetch(32))

    return input_fn



def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, num_labels, use_one_hot_embeddings):
    """
    You can change your return here

    Default is for seq tagging  task


    """
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)
    final_hidden = model.get_sequence_output()
    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())
    with tf.variable_scope("loss"):
        final_hidden_matrix = tf.reshape(final_hidden, [batch_size * seq_length, hidden_size])
        logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        logits = tf.reshape(logits, [batch_size, seq_length, num_labels])
        log_probs = tf.nn.log_softmax(logits, axis=-1)
    return (logits)


def model_fn_builder(bert_config, num_labels, init_checkpoint,
                     use_one_hot_embeddings):
    """
    Here You can change the estimator output format and value

    """

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        # tf.logging.info("*** Features ***")
        # for name in sorted(features.keys()):
        #     tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        logits = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            predict_output = {'values': predictions}
            export_outputs = {'predictions': tf.estimator.export.PredictOutput(predict_output)}

            output_spec = EstimatorSpec(
                mode=mode,
                predictions=predict_output,
                export_outputs=export_outputs)

        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

        return output_spec

    return model_fn

def run_seg_without_record(estimator,label_list,predict_examples,output_dir):
    '''

    :param estimator: Created estimator
    :param label_list: Sequence tagging task label list
    :param predict_examples: list of sentence (length means batch)
    :param output_dir: output_path
    :return: Write File
    '''
    result = estimator.predict(input_fn=input_fn_builder(predict_examples))

    output_predict_file = os.path.join(output_dir,"results.tsv")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
        tf.logging.info("***** Predict results *****")
        for i,prediction in enumerate(result):
            tp = ''.join(list(re.sub('\s{2,}',' ',predict_examples[i].lower())))
            pred =[label_list[p] for p in prediction['values'].tolist()[1:len(tp)+1]]
            k2 = [p for p, v in enumerate(pred) if v[0] == "B" or v[0] == "S"]
            op = [''.join(tp[k2[t]:(len(tp))]) if t == (len(k2) - 1) else ''.join(tp[k2[t]:k2[t + 1]]) for
                  t, p in enumerate(k2) if t < len(k2)]
            writer.write('{}\n'.format(re.sub('\n','',' '.join(op))))

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    bert_config = modeling.BertConfig.from_json_file(os.path.join(model_base,'bert_config.json'))
    init_checkpoint = model_base
    label_list = pickle.load(open(label_path,'rb'))

    if max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (max_seq_length, bert_config.max_position_embeddings))
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=init_checkpoint,
        use_one_hot_embeddings=False)
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = allow_growth
    estimator = Estimator(model_fn=model_fn, config=RunConfig(session_config= sess_config))
    predict_examples = [
                           '話說山東登州府東門外有一座大山，名叫蓬萊山。山上有個閣子，名叫蓬萊閣。這閣造得畫棟飛雲，珠簾捲雨，十分壯麗。西面看城中人戶，煙雨萬家；東面看海上波濤，崢嶸千里。所以城中人士往往於下午攜尊挈酒，在閣中住宿，準備次日天來明時，看海中出日。習以為常，這且不表。卻說那年有個遊客，名叫老殘。此人原姓鐵，單名一個英字，號補殘。因慕懶殘和尚煨芋的故事，遂取這「殘」字做號。大家因他為人頗不討厭，契重他的意思，都叫他老殘。不知不覺，這「老殘」二字便成了個別號了。他年紀不過三十多歲，原是江南人氏。當年也曾讀過幾句詩書，因八股文章做得不通，所以學也未曾進得一個，教書沒人要他，學生意又嫌歲數大，不中用了。其先，他的父親原也是個三四品的官，因性情迂拙，不會要錢，所以做了二十年實缺，回家仍是賣了袍褂做的盤川。你想，可有餘資給他兒子應用呢？這老殘既無祖業可守，又無行當可做，自然「飢寒」二字漸漸的相逼來了。正在無可如何，可巧天不絕人，來了一個搖串鈴的道士，說是曾受異人傳授，能治百病，街上人找他治病，百治百效。所以這老殘就拜他為師，學了幾個口訣。從此也就搖個串鈴，替人治病餬口去了，奔走江湖近二十年。',
                           '話說山東登州府東門外有一座大山，名叫蓬萊山。山上有個閣子，名叫蓬萊閣。這閣造得畫棟飛雲，珠簾捲雨，十分壯麗。西面看城中人戶，煙雨萬家；東面看海上波濤，崢嶸千里。所以城中人士往往於下午攜尊挈酒，在閣中住宿，準備次日天來明時，看海中出日。習以為常，這且不表。卻說那年有個遊客，名叫老殘。此人原姓鐵，單名一個英字，號補殘。因慕懶殘和尚煨芋的故事，遂取這「殘」字做號。大家因他為人頗不討厭，契重他的意思，都叫他老殘。不知不覺，這「老殘」二字便成了個別號了。他年紀不過三十多歲，原是江南人氏。當年也曾讀過幾句詩書，因八股文章做得不通，所以學也未曾進得一個，教書沒人要他，學生意又嫌歲數大，不中用了。其先，他的父親原也是個三四品的官，因性情迂拙，不會要錢，所以做了二十年實缺，回家仍是賣了袍褂做的盤川。你想，可有餘資給他兒子應用呢？這老殘既無祖業可守，又無行當可做，自然「飢寒」二字漸漸的相逼來了。正在無可如何，可巧天不絕人，來了一個搖串鈴的道士，說是曾受異人傳授，能治百病，街上人找他治病，百治百效。所以這老殘就拜他為師，學了幾個口訣。從此也就搖個串鈴，替人治病餬口去了，奔走江湖近二十年。'] * 12
    run_seg_without_record(estimator,label_list,predict_examples,output_dir)

if __name__ == "__main__":
    tf.app.run()
