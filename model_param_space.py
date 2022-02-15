param_space_nfetc_wikim = {
    'wpe_dim': 85,
    'lr': 0.0002,
    'state_size': 180,
    'hidden_layers': 0,
    'hidden_size': 0,
    'dense_keep_prob': 0.7,
    'rnn_keep_prob': 0.9,
    'l2_reg_lambda': 0.0000,
    'batch_size': 512,
    'num_epochs': 20,
    'alpha': 0.4,
    'filter':True, # Whether to distinguish between "one label and multi-label data"
    'label_smoothing':0.1, #ls
    'ancestor_rate':0.1, #ancestor rate
    'warm_epochs':10,
    'e_1':30,
    'e_2':50
}

param_space_nfetc_ontonotes = {
    'wpe_dim': 70,
    'hidden_layers': 2,
    'hidden_size': 1024,
    'dense_keep_prob': 0.7,
    'rnn_keep_prob': 0.6,
    'num_epochs': 50,
    'lr': 0.0006,
    'state_size': 1024,
    'l2_reg_lambda': 0.0000,
    'batch_size': 512,
    'alpha': 0,#0.2, # 0 rep No hier loss
    'filter':True, # Whether to distinguish between "one label and multi-label data"
    'label_smoothing':0.1, #ls
    'ancestor_rate':0.1, #ancestor rate
    'e_1':35, #one label train end epoch
}

param_space_nfetc_bbn = {
    'lr': 0.0007,  # learning rate
    'state_size': 300,  # LSTM dim
    'l2_reg_lambda': 0.000,  # l2 factor
    'alpha': 0.0,  # control the hier loss
    'wpe_dim': 20,  # position embedding dim
    'hidden_layers': 1,  # number of hidden layer of the classifier
    'hidden_size': 560,  # hidden layer dim of the classifier
    'dense_keep_prob': 0.3,  # dense dropout rate of the feature extractor
    'rnn_dense_dropout': 0.3,  # useless
    'rnn_keep_prob': 1.0,  # rnn output droput rate
    'batch_size': 512,
    'num_epochs': 20,
    'filter':True, # Whether to distinguish between "one label and multi-label data"
    'label_smoothing':0.1, #ls
    'ancestor_rate':0.1, #ancestor rate
    'warm_epochs':10,
    'e_1':30,
    'e_2':50
}

param_space_dict = {
    'ontonotes': param_space_nfetc_ontonotes,
    'bbn': param_space_nfetc_bbn,
    'wikim': param_space_nfetc_wikim,
}
