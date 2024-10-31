from accelerate import Accelerator
from omegaconf import OmegaConf, open_dict
from bitlinear import bitlinearize, set_lambda_
import datetime
import hydra
import torch
import time
import wandb
import aimrun

from .utils import (
    setup_basics,
    train,
    predict,
    eval as _eval,
    get_lr_scheduler,
    get_optimizer,
    get_tokenizer,
    get_model,
    get_dataloaders,
    get_config,
)


@hydra.main(config_path="configs", config_name="default", version_base='1.1')
def main(args):
    accelerator = Accelerator(
        cpu=args.device == "cpu",
        mixed_precision=args.precision,
    )
    logger = setup_basics(accelerator, args)
    config = get_config(args)
    model = get_model(args, config, logger)
    if args.quantization_warmup_steps is not None:
        if args.quantization_warmup_offset == 0 or args.quantization_warmup_prequantize:
            bitlinearize(model, replacements=args.bitlinear)
            print(f"Quantized to bitlinear at step 0, setting lambda_ to 0.0.")
            set_lambda_(model, lambda_=0.0)
    print(model)

    tokenizer = get_tokenizer(args)
    optimizer = get_optimizer(model, args, logger)
    lr_scheduler = get_lr_scheduler(optimizer, args, logger)
    train_dataloader, test_dataloader = get_dataloaders(tokenizer, config, args)

    logger.log_args(args)

    (
        model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        test_dataloader,
    ) = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, test_dataloader
    )

    if args.model.checkpoint_path and args.model.load_accelerator:
        accelerator.load_state(args.model.checkpoint_path)

    if args.model.compile:
        model = torch.compile(model)

    with open_dict(args):
        args.current_train_step = 1
        args.current_epoch = 1
        args.last_log = time.time()

    if args.eval_only:
        model.eval()
        with torch.no_grad():
            _eval(model, test_dataloader, logger, args, tokenizer, accelerator)
    elif args.predict_only:
        model.eval()
        with torch.no_grad():
            predict(model, test_dataloader, logger,
                    args, tokenizer)
    else:
        if args.wandb is not None:
            if accelerator.is_main_process:
                run_name = f"{args.model.name} lr={args.optim.base_lr} bl={args.bitlinear} {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
                wandb.init(project=args.wandb, name=run_name)
            accelerator.wait_for_everyone()
        if args.aim.experiment is not None:
            aimrun.init(repo=args.aim.repo, experiment=args.aim.experiment, args=OmegaConf.to_container(args, resolve=True), sync_repo=args.aim.sync_repo, sync_args=args.aim.sync_args)
        train(model, train_dataloader, test_dataloader, accelerator,
              lr_scheduler, optimizer, logger, args, tokenizer)
        if args.wandb is not None:
            if accelerator.is_main_process:
                wandb.finish()
            accelerator.wait_for_everyone()
        aimrun.close()

    logger.finish()


if __name__ == "__main__":
    main()
