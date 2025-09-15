import os
import torch
from torch import Tensor
from typing_extensions import override
from typing import Any, Callable, Dict, Optional, Tuple

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_prefixed_message, rank_zero_warn

from src.utility import log


class ComparingEarlyStopping(Callback):
    mode_dict = {"min": torch.lt, "max": torch.gt}

    order_dict = {"min": "<", "max": ">"}

    def __init__(
        self,
        loss_weights: dict,
        monitor1: str,
        monitor2: str,
        save_filepath: str,
        warmup: int = 0,
        min_delta: float = 0.0,
        patience1: int = 3,
        patience2: int = 3,
        update1: bool = True,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
        check_finite: bool = True,
        check_on_train_epoch_end: Optional[bool] = None,
        log_rank_zero_only: bool = False
    ):
        super().__init__()
        self.warmup = warmup
        self.loss_weights = loss_weights
        self.monitor1 = monitor1
        self.monitor2 = monitor2
        self.min_delta = min_delta
        self.patience1 = patience1
        self.patience2 = patience2
        self.verbose = verbose
        self.mode = mode
        self.strict = strict
        self.check_finite = check_finite
        
        self.wait_count = 0
        self.should_stop = False
        self.stopped_epoch = 0
        self._check_on_train_epoch_end = check_on_train_epoch_end
        self.log_rank_zero_only = log_rank_zero_only
        self.update1 = update1

        if self.mode not in self.mode_dict:
            raise MisconfigurationException(f"`mode` can be {', '.join(self.mode_dict.keys())}, got {self.mode}")

        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        torch_inf = torch.tensor(torch.inf)
        self.best_score1 = torch_inf if self.monitor_op == torch.lt else -torch_inf
        self.best_score2 = torch_inf if self.monitor_op == torch.lt else -torch_inf
        
        self.save_filepath = save_filepath

    @property
    @override
    def state_key(self) -> str:
        return self._generate_state_key(monitor1=self.monitor1, monitor2=self.monitor2, mode=self.mode)

    @override
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        if self._check_on_train_epoch_end is None:
            # if the user runs validation multiple times per training epoch or multiple training epochs without
            # validation, then we run after validation instead of on train epoch end
            self._check_on_train_epoch_end = trainer.val_check_interval == 1.0 and trainer.check_val_every_n_epoch == 1

    def _validate_condition_metric(self, logs: Dict[str, Tensor]) -> bool:
        monitor_val1 = logs.get(self.monitor1)
        monitor_val2 = logs.get(self.monitor2)

        error_msg = (
            "Early stopping conditioned on metric `{}` which is not available."
            " Pass in or modify your `EarlyStopping` callback to use any of the following:"
            f' `{"`, `".join(list(logs.keys()))}`'
        )

        not_exitst_monitor = []
        if monitor_val1 is None:
            not_exitst_monitor.append(self.monitor1)
        if monitor_val2 is None:
            not_exitst_monitor.append(self.monitor2)
        
        if monitor_val1 is None or monitor_val2 is None:
            if self.strict:
                raise RuntimeError(error_msg.format(",".join(not_exitst_monitor)))
            if self.verbose > 0:
                rank_zero_warn(error_msg.format(",".join(not_exitst_monitor)), category=RuntimeWarning)
            return False

        return True

    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]

    @override
    def state_dict(self) -> Dict[str, Any]:
        return {
            "update1": self.update1,
            "stopped_epoch": self.stopped_epoch,
            "wait_count": self.wait_count,
            "best_score1": self.best_score1,
            "best_score2": self.best_score2,
            "patience1": self.patience1,
            "patience2": self.patience2,
            "should_stop": self.should_stop
        }

    @override
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.update1 = state_dict["update1"]
        self.stopped_epoch = state_dict["stopped_epoch"]
        self.wait_count = state_dict["wait_count"]
        self.best_score1 = state_dict["best_score1"]
        self.best_score2 = state_dict["best_score2"]
        self.patience1 = state_dict["patience1"]
        self.patience2 = state_dict["patience2"]
        self.should_stop = state_dict["should_stop"]
        
    def _should_skip_check(self, trainer: "pl.Trainer") -> bool:
        from lightning.pytorch.trainer.states import TrainerFn

        return trainer.state.fn != TrainerFn.FITTING or trainer.sanity_checking

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when fit begins."""
        self._train_branch(trainer, pl_module, 'main', y_effect=False if self.warmup>0 else True)
        
    @override
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer, pl_module)

    @override
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer, pl_module)
    
    def _run_early_stopping_check(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        
        logs = trainer.callback_metrics

        if trainer.fast_dev_run or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
            logs
        ):  # short circuit if metric not present
            return

        current1 = logs[self.monitor1].squeeze()
        current2 = logs[self.monitor2].squeeze()
        if trainer.current_epoch < self.warmup:
            if self.monitor_op(current1 - self.min_delta, self.best_score1.to(current1.device)):
                self.best_score1 = current1
            if self.monitor_op(current2 - self.min_delta, self.best_score2.to(current2.device)):
                self.best_score2 = current2
            if trainer.current_epoch+1 == self.warmup//2:
                self._train_branch(trainer, pl_module, 'aux', y_effect=False)
            elif trainer.current_epoch+1 == self.warmup:
                arl_start_from = 'main'
                if self.monitor_op(self.best_score1.to(current1.device), self.best_score2.to(current1.device)):
                    arl_start_from = 'aux'
                self._train_branch(trainer, pl_module, arl_start_from, y_effect=True)
                self._average_model(trainer, pl_module)
            return
        
        should_stop, should_switch, reason = self._evaluate_stopping_criteria(current1, current2)
        
        if should_stop:
            if self.should_stop:
                # stop every ddp process if any world process decides to stop
                should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
                trainer.should_stop = trainer.should_stop or should_stop
                if should_stop:
                    self.stopped_epoch = trainer.current_epoch
            else:
                self.should_stop = True
                trainer.save_checkpoint(os.path.join(self.save_filepath, f"epoch_{trainer.current_epoch}_EStop_match.ckpt"))
                self._switch(trainer, pl_module, y_effect=False)
                
        elif should_switch:
            self._switch(trainer, pl_module, y_effect=True)
            self._average_model(trainer, pl_module)
        
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)
    
    def _switch(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, y_effect: bool):
        self.wait_count = 0
        branch = 'aux' if self.update1 else 'main'
        self._train_branch(trainer, pl_module, branch, y_effect=y_effect)
    
    def _average_model(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # load optimizer if lr_scheduler is applied
        # raise NotImplementedError
        if self.update1:
            training_model = 'branchs.0'
            fixxed_model = 'branchs.1'
        else:
            training_model = 'branchs.1'
            fixxed_model = 'branchs.0'
        
        model_weights = pl_module.state_dict()
        for k in model_weights.keys():
            if training_model in k:
                fixxed_k = k.replace(training_model, fixxed_model)
                model_weights[k].data.copy_((model_weights[k] + model_weights[fixxed_k]) / 2.)
        
    def _evaluate_stopping_criteria(self, current1: Tensor, current2: Tensor) -> Tuple[bool, Optional[str]]:
        should_switch = False
        should_stop = False
        reason = ""
        if self.check_finite and not torch.isfinite(current1) and not torch.isfinite(current2):
            should_stop = True
            reason = (
                f"Monitored metric {self.monitor1}, {self.monitor2} = {current1}, {current2} is not finite."
                f" Previous best value was {self.best_score1:.3f}, {self.best_score2:.3f}. Signaling Trainer to stop."
            )
        elif self.update1 and self.monitor_op(current1 - self.min_delta, self.best_score1.to(current1.device)):
            reason = self._improvement_message(current1, self.best_score1)
            self.best_score1 = current1
            self.wait_count = 0
            if self.monitor_op(current1, self.best_score2) and not self.should_stop:
                should_switch = True
                reason += f" Outperform {self.monitor2}: {self.best_score2:.3f}."
        elif not self.update1 and self.monitor_op(current2 - self.min_delta, self.best_score2.to(current2.device)):
            reason = self._improvement_message(current2, self.best_score2)
            self.best_score2 = current2
            self.wait_count = 0
            if self.monitor_op(current2, self.best_score1) and not self.should_stop:
                should_switch = True
                reason += f" Outperform {self.monitor1}: {self.best_score1:.3f}."
        else:
            self.wait_count += 1
            patience = self.patience1 if self.update1 else self.patience2
            if self.wait_count >= patience:
                if self.update1:
                    should_stop = True
                    reason = (
                        f"Monitored metric {self.monitor1} did not improve in the last {self.wait_count} records."
                        f" Best scores: {self.best_score1:.3f}, {self.best_score2:.3f}. Signaling Trainer to stop trining this branch."
                    )
                else:
                    should_stop = True
                    reason = (
                        f"Monitored metric {self.monitor2} did not improve in the last {self.wait_count} records."
                        f" Best scores: {self.best_score1:.3f}, {self.best_score2:.3f}. Signaling Trainer to stop trining this branch."
                    )
        
        return should_stop, should_switch, reason

    def _improvement_message(self, current: Tensor, best: Tensor) -> str:
        """Formats a log message that informs the user about an improvement in the monitored score."""
        monitor = self.monitor1 if self.update1 else self.monitor2
        if torch.isfinite(best):
            msg = (
                f"Metric {monitor} improved by {abs(best - current):.3f} >="
                f" min_delta = {abs(self.min_delta)}. New best score: {current:.3f}."
            )
        else:
            msg = f"Metric {monitor} improved. New best score: {current:.3f}."
        return msg

    @staticmethod
    def _log_info(trainer: "pl.Trainer", message: str, log_rank_zero_only) -> None:
        
        rank = trainer.global_rank if trainer.world_size > 1 else None
        message = rank_prefixed_message(message, rank)
        if rank is None or not log_rank_zero_only or rank == 0:
            log.info(f"\n**\nARL log:{message}\n**\n")
            
    def _train_branch(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, branch: str, y_effect: bool):
        if self.verbose:
            if not y_effect:
                self._log_info(trainer, f"Start pretraining {branch} branch", self.log_rank_zero_only)
            else:
                self._log_info(trainer, f"Switch to train {branch} branch", self.log_rank_zero_only)
        if branch == 'main':
            self._train_main(pl_module, y_effect)
        elif branch == 'aux':
            self._train_aux(pl_module, y_effect)
        else:
            raise NotImplementedError
        log.debug((f"Training loss: y_pred: {pl_module.arl_loss.loss_weights['y_pred']}"
                    f", y_effect: {pl_module.arl_loss.loss_weights['y_effect']}"
                    f", y_aux: {pl_module.arl_loss.loss_weights['y_aux']}"
                    ))

    def _train_main(self, pl_module: pl.LightningModule, y_effect: bool):
        
        self.update1 = True
        pl_module.arl_loss.update1 = True
        pl_module.arl_loss.loss_weights["y_pred"] = self.loss_weights["y_pred"]
        pl_module.arl_loss.loss_weights["y_effect"] = self.loss_weights["y_effect"] if y_effect else 0. 
        pl_module.arl_loss.loss_weights["y_aux"] = 0.
        pl_module.train_main()
        
    def _train_aux(self, pl_module: pl.LightningModule, y_effect: bool):
        
        self.update1 = False
        pl_module.arl_loss.update1 = False
        pl_module.arl_loss.loss_weights["y_pred"] = 0.
        pl_module.arl_loss.loss_weights["y_effect"] = self.loss_weights["y_effect"] if y_effect else 0. 
        pl_module.arl_loss.loss_weights["y_aux"] = self.loss_weights["y_aux"]
        pl_module.train_aux()
        