# experiments/exp_term_forecasting.py
from __future__ import annotations

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import time
import warnings
import numpy as np
from torch.optim import lr_scheduler

from utils.llm_explain import build_numeric_bundle, explain_forecast_with_llm

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    # ------------------- plumbing -------------------
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_for_params = self.model.module if hasattr(self.model, "module") else self.model
        base, pde = [], []
        for name, p in model_for_params.named_parameters():
            if not p.requires_grad:
                continue
            if ("pde." in name) or ("pde_block" in name):
                pde.append(p)
            else:
                base.append(p)
        if len(pde) == 0:
            print("[opt] PDE param group empty – training with base LR only.")
            param_groups = [{"params": base, "lr": self.args.learning_rate, "weight_decay": 1e-4}]
        else:
            print(f"[opt] PDE params: {len(pde)}")
            param_groups = [
                {"params": base, "lr": self.args.learning_rate,       "weight_decay": 1e-4},
                {"params": pde,  "lr": self.args.learning_rate*0.3,   "weight_decay": 1e-4},
            ]
        return torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    def _select_criterion(self):
        return nn.MSELoss()

    def _freeze_pde(self, flag: bool = True):
        model_for_params = self.model.module if hasattr(self.model, "module") else self.model
        for n, p in model_for_params.named_parameters():
            if ("pde." in n) or ("pde_block" in n):
                p.requires_grad = (not flag)

    # ------------------- validation -------------------
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None; batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x) if not self.args.output_attention else \
                          self.model(batch_x, batch_x_mark, None, batch_y_mark)[0]

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
        self.model.train()
        return float(np.average(total_loss))

    # ------------------- training -------------------
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # warm start: freeze PDE for 2 epochs
        self._freeze_pde(flag=True)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate
        )

        for epoch in range(self.args.train_epochs):
            if epoch == 2:
                self._freeze_pde(flag=False)

            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None; batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x) if not self.args.output_attention else \
                                  self.model(batch_x, batch_x_mark, None, batch_y_mark)[0]
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y_cut = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y_cut) + F.l1_loss(outputs, batch_y_cut)
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    outputs = self.model(batch_x) if not self.args.output_attention else \
                              self.model(batch_x, batch_x_mark, None, batch_y_mark)[0]
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_cut = batch_y[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y_cut) + F.l1_loss(outputs, batch_y_cut)
                    loss.backward()
                    model_optim.step()

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = float(np.average(train_loss))
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch+1}, Steps: {train_steps} | "
                  f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    # ------------------- testing + LLM explanation -------------------
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds, trues = [], []
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None; batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # forward
                outputs = self.model(batch_x) if not self.args.output_attention else \
                          self.model(batch_x, batch_x_mark, None, batch_y_mark)[0]

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_cut = batch_y[:, -self.args.pred_len:, f_dim:]

                # optional LLM explanation on first batch only
                if getattr(self.args, "explain", False) and i == 0:
                    try:
                        model_for_read = self.model.module if hasattr(self.model, "module") else self.model
                        bw = None
                        if hasattr(model_for_read, "last_explain"):
                            le = model_for_read.last_explain if isinstance(model_for_read.last_explain, dict) else {}
                            bw_list = None
                            if "branch_weights_mean" in le:
                                bw_list = le["branch_weights_mean"]
                            elif "branch_weights" in le:
                                bw_list = le["branch_weights"]
                            if bw_list is not None:
                                K = len(bw_list)
                                B, Tp, N = outputs.shape
                                bw = torch.tensor(bw_list, device=outputs.device).view(1, 1, K).expand(B, N, K)

                        var_names = getattr(test_data, "cols", None)
                        units = getattr(self.args, "units", None)
                        bundle = build_numeric_bundle(
                            preds=outputs,
                            trues=batch_y_cut,
                            inputs=batch_x,
                            branch_weights=bw,
                            var_names=var_names,
                            units=units
                        )
                        text = explain_forecast_with_llm(
                            bundle,
                            provider=getattr(self.args, "explain_provider", "openai"),
                            model=getattr(self.args, "explain_model", "gpt-4o-mini"),
                            api_key=getattr(self.args, "llm_api_key", None),
                        )
                        os.makedirs("./test_results/explanations", exist_ok=True)
                        with open(f"./test_results/explanations/{self.args.model_id}_batch0.txt", "w") as f:
                            f.write(text + "\n")
                        print("\n[LLM EXPLANATION]\n" + text + "\n")
                    except Exception as e:
                        print(f"[explain] skipped: {e}")

                # to numpy for metrics
                outputs_np = outputs.detach().cpu().numpy()
                batch_y_np = batch_y_cut.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    outputs_np = test_data.inverse_transform(outputs_np)
                    batch_y_np = test_data.inverse_transform(batch_y_np)

                preds.append(outputs_np)
                trues.append(batch_y_np)

                if i % 200 == 0:
                    input_np = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input_np[0, :, -1], batch_y_np[0, :, -1]), axis=0)
                    pd = np.concatenate((input_np[0, :, -1], outputs_np[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        result_folder = './results/' + setting + '/'
        os.makedirs(result_folder, exist_ok=True)

        mae, mse, rmse, mape, mspe, rse = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, rse:{}, rmse:{}, mape{}, mspe{}\n\n'.format(mse, mae, rse, rmse, mape, mspe))
        return
