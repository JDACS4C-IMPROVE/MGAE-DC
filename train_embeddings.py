import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, mean_squared_error
import tensorflow.compat.v1 as tf

# --------------------------------------------------------------------
# Helper functions from original code
# --------------------------------------------------------------------

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def get_classification_stats(net_labels, net_preds):
    net_preds_binary = [ 1 if x >= 0.5 else 0 for x in net_preds ]
    net_auc = roc_auc_score(net_labels, net_preds)
    precision, recall, _ = precision_recall_curve(net_labels, net_preds)
    net_aupr = auc(recall, precision)
    net_acc = accuracy_score(net_preds_binary, net_labels)
    net_f1 = f1_score(net_labels, net_preds_binary)
    net_precision = precision_score(net_labels, net_preds_binary, zero_division=0)
    net_recall = recall_score(net_labels, net_preds_binary)
    fnoutput = [net_auc,net_acc, net_aupr, net_f1, net_precision, net_recall]
    return fnoutput

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
# --------------------------------------------------------------------
# New helper functions based on original code
# --------------------------------------------------------------------

def construct_synergy_networks(data, cellslist, drugslist, indexs_all, params):
    """
    Constructs networks for drug synergy prediction from pre-filtered training or validation data.

    Args:
        data (pd.DataFrame): Pre-filtered dataframe (either train or validation data).
        cellslist (list): List of cell lines to process.
        drugslist (list): List of unique drugs for indexing.
        indexs_all (list): List of all possible index pairs.
        params (dict): Dictionary of parameter values, including column names.

    Returns:
        dict: Dictionary containing adjacency matrices, labels, and indices.
    """

    networks = {
        "d_net1_norm": {}, "d_net2_norm": {}, "d_net3_norm": {},
        "d_net1_labels": {}, "d_net2_labels": {}, "d_net3_labels": {},
        "d_net1_index": {}, "d_net2_index": {}, "d_net3_index": {},
        "d_net1_pos_weight": {}, "d_net2_pos_weight": {}, "d_net3_pos_weight": {}
    }

    drugscount = len(drugslist)
    
    for cellidx, cellname in enumerate(cellslist):
        print(f'Constructing networks for {cellname}')
        each_data = data[data[params['canc_col_name']] == cellname]

        # Initialize adjacency matrices
        net_adj_train = {1: np.zeros((drugscount, drugscount)),
                         2: np.zeros((drugscount, drugscount)),
                         3: np.zeros((drugscount, drugscount))}
        
        # Store positive sample positions
        net_train_pos = {1: [], 2: [], 3: []}

        for each in each_data.values:
            drug1, drug2, cell, synergy = each[
            [data.columns.get_loc(params["drug_col_name_1"]),
             data.columns.get_loc(params["drug_col_name_2"]),
             data.columns.get_loc(params["canc_col_name"]),
             data.columns.get_loc(params["y_col_name"])]
            ]
            drugidx1, drugidx2 = drugslist.index(drug1), drugslist.index(drug2)
            if drugidx2 < drugidx1:
                drugidx1, drugidx2 = drugidx2, drugidx1
            
            # Determine network type based on synergy
            net_type = 1 if float(synergy) >= params['additive_max'] else 2 if float(synergy) > params['additive_min'] else 3

            net_adj_train[net_type][drugidx1, drugidx2] = 1
            net_train_pos[net_type].extend([[drugidx1, drugidx2], [drugidx2, drugidx1]])

        for net in [1, 2, 3]:
            net_adj_train[net] = sp.csr_matrix(net_adj_train[net])
            net_adj_train[net] += net_adj_train[net].T
            net_adj_norm_train = preprocess_graph(net_adj_train[net])

            # Store normalized adjacency matrices
            networks[f"d_net{net}_norm"][cellidx] = net_adj_norm_train

            # Compute labels
            net_labels_train = net_adj_train[net] + sp.eye(net_adj_train[net].shape[0])
            net_labels_train = sparse_to_tuple(net_labels_train)
            net_labels_train_pos = [indexs_all.index(x) for x in net_labels_train[0].tolist()]
            net_labels_train_neg = [x for x in range(len(indexs_all)) if x not in net_labels_train_pos]

            networks[f"d_net{net}_labels"][cellidx] = [net_labels_train_pos, net_labels_train_neg]

            # Compute class weights
            pos_weight = (net_adj_train[net].shape[0] ** 2 - net_adj_train[net].sum()) / net_adj_train[net].sum()
            networks[f"d_net{net}_pos_weight"][cellidx] = pos_weight

            # Compute indices
            net_index_train_pos = [indexs_all.index(x) for x in net_train_pos[net]]
            net_index_train_neg = [indexs_all.index(x) for x in net_train_pos[3 if net != 3 else 2] + net_train_pos[2 if net != 2 else 1]]

            networks[f"d_net{net}_index"][cellidx] = [net_index_train_pos, net_index_train_neg]

    return networks


def train_model(sess, model, opt, placeholders, drug_feat, train_networks, val_networks, params, resultspath, cellscount):
    """
    Trains a deep learning model for drug synergy networks and returns the minimum validation loss.

    Args:
        sess (tf.Session): TensorFlow session.
        model (tf.Model): Deep learning model.
        opt (tf.Optimizer): Optimizer for training.
        placeholders (dict): TensorFlow placeholders for input data.
        drug_feat (tf.Tensor): Input features for drugs.
        train_networks (dict): Processed adjacency matrices and labels for training.
        val_networks (dict): Processed adjacency matrices and labels for validation.
        params (dict): Dictionary of parameter values.
        resultspath (str): Path to save model checkpoints.
        cellscount (int): Number of cell lines.

    Returns:
        tuple: (loss_history, min_loss, feed_dict), where
            - loss_history (dict) contains training and validation loss history.
            - min_loss (float) is the lowest validation loss achieved.
            - feed_dict (dict): Final dictionary used for feeding values into TensorFlow.
    """

    # Initialize TensorFlow session
    sess.run(tf.global_variables_initializer())

    # Track best validation performance
    min_loss = float('inf')
    saver = tf.train.Saver(max_to_keep=1)

    loss_history = {"train": [], "valid": []}
    # Training loop
    for epoch in range(params["epochs"]):
        feed_dict = {
            placeholders['features']: drug_feat,
            placeholders['dropout']: params["dropout"]
        }

        # Add adjacency matrices for training
        for net in range(1, 4):
            feed_dict.update({placeholders[f'net{net}_adj_norm_{cellidx}']: train_networks[f"d_net{net}_norm"][cellidx] 
                              for cellidx in range(cellscount)})

        # Run training step
        _, train_loss = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        loss_history["train"].append(train_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1:04d} | Training Loss: {train_loss:.4f}")

        # Compute validation loss
        feed_dict[placeholders['dropout']] = 0  # No dropout for validation
        res1, res2 = sess.run([model.reconstructions_common, model.reconstructions_specific], feed_dict=feed_dict)

        valid_loss = 0
        for cellidx in range(cellscount):
            for net in range(1, 4):  # Loop over three networks
                net_index_pos, net_index_neg = val_networks[f"d_net{net}_index"][cellidx]
                pos_neg_weight = len(net_index_neg) / len(net_index_pos)

                net_labels_pos = [1] * len(net_index_pos)
                net_labels_neg = [0] * len(net_index_neg)

                net_preds_common_pos = [sigmoid(res1[cellidx][net - 1].reshape(-1, 1)[x][0]) for x in net_index_pos]
                net_preds_common_neg = [sigmoid(res1[cellidx][net - 1].reshape(-1, 1)[x][0]) for x in net_index_neg]
                net_loss_common = pos_neg_weight * mean_squared_error(net_preds_common_pos, net_labels_pos) + \
                                  mean_squared_error(net_preds_common_neg, net_labels_neg)

                net_preds_specific_pos = [sigmoid(res2[cellidx][net - 1].reshape(-1, 1)[x][0]) for x in net_index_pos]
                net_preds_specific_neg = [sigmoid(res2[cellidx][net - 1].reshape(-1, 1)[x][0]) for x in net_index_neg]
                net_loss_specific = pos_neg_weight * mean_squared_error(net_preds_specific_pos, net_labels_pos) + \
                                    mean_squared_error(net_preds_specific_neg, net_labels_neg)

                net_loss = net_loss_common + net_loss_specific
                valid_loss += net_loss

        loss_history["valid"].append(valid_loss)

        # Save best model and track min_loss
        if valid_loss < min_loss:
            min_loss = valid_loss
            saver.save(sess, f"{resultspath}/best_model.ckpt")

    print("Training Complete! Restoring Best Model...")
    saver.restore(sess, f"{resultspath}/best_model.ckpt")

    return loss_history, min_loss, feed_dict
