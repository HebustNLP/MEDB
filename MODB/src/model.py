from .util import *

class BertForModel(BertPreTrainedModel):
    def __init__(self, config, _num_labels):
        super(BertForModel, self).__init__(config)
        # 兼容新版 transformers：无 tied weights 时需提供空 dict
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        self.num_labels = _num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size,_num_labels)
        self.feature_transform = nn.Linear(config.hidden_size, 768)
        self.init_weights()
        
    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, centroids = None):

        encoded_layer_12 = self.bert(input_ids, token_type_ids, attention_mask, output_hidden_states = True, return_dict=True).last_hidden_state

        pooled_output = self.dense(encoded_layer_12.mean(dim = 1))
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if feature_ext:
            pooled_output = self.feature_transform(pooled_output)
            return pooled_output
        else:
            if mode == 'train':
                loss = nn.CrossEntropyLoss()(logits,labels)
                return loss
            else:
                pooled_output = self.feature_transform(pooled_output)
                return pooled_output, logits
