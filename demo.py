# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A demo script showing how to use the uisrnn package on toy data."""

import numpy as np
from functools import partial
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import os

mp = mp.get_context('forkserver')
from uisrnn import utils
import uisrnn

NUM_WORKERS = 4

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def diarization_experiment(model_args, training_args, inference_args):
    """Experiment pipeline.

  Load data --> train model --> test model --> output result

  Args:
    model_args: model configurations
    training_args: training configurations
    inference_args: inference configurations
  """
    # data loading
    train_data = np.load('./data/PV_training_data.npz', allow_pickle=True)
    test_data = np.load('./data/PV_testing_data_nooverlap.npz', allow_pickle=True)
    train_sequences = train_data['train_sequences']
    train_cluster_id = train_data['train_cluster_ids'].tolist()
    test_sequences = test_data['test_sequences']
    test_cluster_ids = test_data['test_cluster_ids']
    test_file_names = test_data['test_file_ids']
    test_times = test_data['test_time_list']

    train_sequences = [utils.dataNorm(t) for t in train_sequences]
    test_sequences = [utils.dataNorm(t) for t in test_sequences]

    print('Number of train sequences: %d' % len(train_sequences))
    print('Number of test sequences: %d' % len(test_sequences))
    # model init
    model = uisrnn.UISRNN(model_args)
    if training_args.epochs == 0:
        model.load(model_args.model_name)  # to load a checkpoint
    else:
        # tensorboard writer init
        writer = SummaryWriter()  ###log_dir = "name of the folder"

    # training
    print('Training')
    for epoch in range(training_args.epochs):
        print('epoch %d' % epoch)
        stats = model.fit(train_sequences, train_cluster_id, training_args)

        # add to tensorboard
        for loss, cur_iter in stats:
            for loss_name, loss_value in loss.items():
                writer.add_scalar('loss/' + loss_name, loss_value, cur_iter)
        # save the model
        model.save(model_args.model_name + '_' + str(epoch))

    # testing
    predicted_cluster_ids = []
    test_record = []

    # predict sequences in parallel
    model.rnn_model.share_memory()
    pool = mp.Pool(NUM_WORKERS, maxtasksperchild=None)

    pred_gen = pool.imap(
        func=partial(model.predict, args=inference_args),
        iterable=test_sequences)

    print('Collect predictions and score predictions')
    for idx, predicted_cluster_id in enumerate(pred_gen):
        print(idx, test_file_names[idx], len(test_cluster_ids[idx]), end=' ', flush=True)

        accuracy = uisrnn.compute_sequence_match_accuracy(
            test_cluster_ids[idx], predicted_cluster_id)
        predicted_cluster_ids.append((test_file_names[idx], test_times[idx], predicted_cluster_id))

        print('Accuracy = %.2f' % accuracy)
        test_record.append((accuracy, len(test_cluster_ids[idx])))

    # close multiprocessing pool
    pool.close()
    if training_args.epochs > 0:
        # close tensorboard writer
        writer.close()

    print('Finished diarization experiment')
    print(uisrnn.output_result(model_args, training_args, inference_args, test_record))

    return predicted_cluster_ids


def write_rttm_files(predictions, use_seg=False):
    """Outputs predictions in rttm formats. Adjacent clusters are merged
    Options:
    use_seg: if true it uses the speaker change information (segments) taking the most frequent id in the segment

    """
    for filename, times, ids in predictions:
        clusters = []
        if use_seg: # use segmentation information
            idx = 0
            for t in times:
                id_list=ids[idx:idx+len(t)]
                idx=idx+len(t)
                clusters.append((t[0], t[-1], max(set(id_list),key=id_list.count)))
        else: # cluster every observations
            time_v = np.hstack(times)
            for t, spk in zip(time_v, ids):
                if len(clusters) > 0 and (t - clusters[-1][1] <= 0.1) and clusters[-1][2] == spk:
                    clusters[-1] = (clusters[-1][0], t + float(0.1), spk)
                else:
                    clusters.append((t + 0.001, t + float(0.1), spk))

        # write rttm file...
        with open('rttms/' + filename + '.rttm', 'w') as file_object:
            for c in clusters:
                file_object.write(
                    'SPEAKER %s 1 %.4f %.4f <NA> <NA> %s <NA>\n' % (filename, c[0], (c[1] - c[0]), str(c[2])))


def main():
    """The main function."""
    model_args, training_args, inference_args = uisrnn.parse_arguments()
    predictions = diarization_experiment(model_args, training_args, inference_args)
    write_rttm_files(predictions)


if __name__ == '__main__':
    main()
