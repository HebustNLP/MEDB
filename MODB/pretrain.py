from model import *
from dataloader import *
from util import *
from myloss import *
from adaptive_boundary_loss import AdaptiveBoundaryLoss

from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
# from utils import util


class PretrainModelManager:

    def __init__(self, args, data):

        self.model = BertForModel.from_pretrained(args.bert_model, cache_dir="", _num_labels=data.num_labels)

        if args.freeze_bert_parameters:
            for name, param in self.model.bert.named_parameters():
                param.requires_grad = False
                if "encoder.layer.11" in name or "pooler" in name:
                    param.requires_grad = True

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.num_train_optimization_steps = int(
            len(data.train_examples) / args.train_batch_size) * args.num_train_epochs

        self.optimizer, self.scheduler = self.get_optimizer(args)
        self.optimizer2, self.scheduler2 = self.get_optimizer2(args)

        self.best_eval_score = 0
        self.clusterLoss = clusterLoss(args, data)
        self.gb_centroids = None
        self.gb_radii = None
        self.gb_labels = None

    def eval(self, args, data):

        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.num_labels)).to(self.device)

        for batch in tqdm(data.eval_dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                _, logits = self.model(input_ids, segment_ids, input_mask,
                                       mode='eval')
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(
            dim=1)
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)

        return acc



    def train(self, args, data):
        wait = 0
        best_model = None
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            total_feats = []
            total_labels = []

            for step, batch in enumerate(tqdm(data.train_dataloader,
                                              desc="Iteration")):
                batch = tuple(t.to(self.device) for t in
                              batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    loss1 = self.model(input_ids, segment_ids, input_mask, label_ids,
                                       mode="train")

                    self.optimizer.zero_grad()
                    loss1.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    tr_loss += loss1.item()
                    # util.summary_writer.add_scalar("Loss/loss1", loss1.item(), step+ epoch*batch_number)
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    features = self.model(input_ids, segment_ids, input_mask, feature_ext="True")
                    total_feats.append(features.detach().cpu())
                    total_labels.append(label_ids.detach().cpu())


            accumulated_features = torch.cat(total_feats, dim=0).to(self.device)
            accumulated_labels = torch.cat(total_labels, dim=0).to(self.device)
            self.gb_centroids, self.gb_radii, self.gb_labels= self.clusterLoss.forward(args,accumulated_features,accumulated_labels, type='ball', select=False)
                    
            for step, batch in enumerate(tqdm(data.train_dataloader,
                                              desc="Iteration")):
                batch = tuple(t.to(self.device) for t in
                              batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    features = self.model(input_ids, segment_ids, input_mask, feature_ext="True")
                    loss2 = self.clusterLoss.forward(args,features,label_ids, type='loss', select=False)
                    self.optimizer2.zero_grad()
                    loss2.backward()
                    # util.summary_writer.add_scalar("Loss/loss11", loss2.item(), step + epoch * batch_number)

                    self.optimizer2.step()
                    self.scheduler2.step()


            loss = tr_loss / nb_tr_steps
            print('train_loss', loss)
            eval_score = self.eval(args, data)
            print('eval_score', eval_score)
            if eval_score > self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break

        self.model = best_model
        if args.save_model:
            self.save_model(args)
        return self.gb_centroids, self.gb_radii, self.gb_labels

    def calculate_granular_balls(self, args, data):

        with torch.set_grad_enabled(False):
            for epoch in trange(int(1), desc="Epoch"):
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                memory_bank = []
                memory_bank_label = []

                for step, batch in enumerate(tqdm(data.train_dataloader,
                                                desc="Iteration")):
                    batch = tuple(t.to(self.device) for t in
                                batch)
                    input_ids, input_mask, segment_ids, label_ids = batch
                    batch_number = len(data.train_dataloader)


                    features = self.model(input_ids, segment_ids, input_mask, feature_ext="True")
                    memory_bank.append(features.detach().cpu())
                    memory_bank_label.append(label_ids.cpu())

                    if (step + 1) == batch_number:
                        accumulated_features = torch.cat(memory_bank, dim=0).to(self.device)
                        accumulated_labels = torch.cat(memory_bank_label, dim=0).to(self.device)
                        self.gb_centroids, self.gb_radii, self.gb_labels = self.clusterLoss.forward(args,
                                                                                                        accumulated_features,
                                                                                                        accumulated_labels,
                                                                                                        select=True, type='ball')
        return self.gb_centroids, self.gb_radii, self.gb_labels

    def train_adaptive_boundary(self, args, data, gb_centroids, gb_radii, gb_labels):
        """
        在多粒度球基础上训练动态自适应决策边界（可学习 delta，可选椭球）。
        返回：gb_centroids, 学到的半径 delta, gb_labels, boundary_loss 模块（含 get_rotate_matrix 供推理用）。
        """
        setattr(args, 'num_labels', data.num_labels)
        boundary_loss = AdaptiveBoundaryLoss(args, gb_centroids, gb_radii, gb_labels, device=self.device)
        boundary_loss.to(self.device)
        opt_boundary = torch.optim.Adam(
            boundary_loss.parameters(),
            lr=getattr(args, 'lr3', getattr(args, 'lr2', 1e-4))
        )
        n_epochs = getattr(args, 'adaptive_boundary_epochs', 5)
        self.model.eval()
        for _ in trange(n_epochs, desc="Adaptive boundary"):
            boundary_loss.train()
            for batch in tqdm(data.train_dataloader, desc="Boundary iter"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                    _, _, _, _, loss = boundary_loss(features, label_ids)
                    opt_boundary.zero_grad()
                    loss.backward()
                    opt_boundary.step()
        learned_delta = boundary_loss.delta.detach()
        return gb_centroids, learned_delta, gb_labels, boundary_loss

    def get_optimizer(self, args):

        param_optimizer = list(self.model.named_parameters())
        param_optimizer = [p for p in param_optimizer if p[1].requires_grad]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},

            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}

        ]
        num_warmup_steps = int(args.warmup_proportion * self.num_train_optimization_steps)
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.lr
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.num_train_optimization_steps
        )
        return optimizer, scheduler

    def get_optimizer2(self, args):

        param_optimizer = list(self.model.named_parameters())
        param_optimizer = [p for p in param_optimizer if p[1].requires_grad]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},

            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}

        ]
        num_warmup_steps = int(args.warmup_proportion * self.num_train_optimization_steps)
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.lr2
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.num_train_optimization_steps
        )
        return optimizer, scheduler

    def save_model(self, args):

        if not os.path.exists(args.pretrain_dir):
            os.makedirs(args.pretrain_dir)
        self.save_model = self.model.module if hasattr(self.model,
                                                       'module') else self.model

        model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model_config_file = os.path.join(args.pretrain_dir, CONFIG_NAME)
        torch.save(self.save_model.state_dict(), model_file)
        with open(model_config_file, "w") as f:
            f.write(self.save_model.config.to_json_string())


def calculate_distances(a, b):
    distances = torch.sqrt(torch.sum((a[:, None, :] - b[None, :, :]) ** 2, dim=2))
    return distances

