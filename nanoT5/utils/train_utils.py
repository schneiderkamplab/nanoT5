import torch
import time
import evaluate
from .logging_utils import Averager
from datasets.iterable_dataset import IterableDataset
import wandb
import aimrun
from bitlinear import bitlinearize, set_lambda_
import math
from .t11_model import T5SequenceNorm

def maybe_save_checkpoint(model, accelerator, optimizer, args):
    if (
        args.current_train_step > args.optim.total_steps
        or args.current_train_step % args.checkpoint.every_steps == 0
    ):
        model.eval()
        if args.optim.name == 'adamwschedulefree':
            optimizer.eval()
        output_dir = f'checkpoint-{args.mode}-{args.current_train_step}'
        accelerator.save_state(output_dir=output_dir)
        model.train()
        if args.optim.name == 'adamwschedulefree':
            optimizer.train()


def maybe_eval_predict(model, dataloader, logger, args, tokenizer, accelerator, optimizer):
    if (
        args.current_train_step > args.optim.total_steps
        or args.current_train_step % args.eval.every_steps == 0
    ):
        model.eval()
        if args.optim.name == 'adamwschedulefree':
            optimizer.eval()

        with torch.no_grad():
            eval(model, dataloader, logger, args, tokenizer, accelerator)

            if args.mode == 'ft':
                predict(
                    model, dataloader, logger, args, tokenizer
                )

        args.last_log = time.time()
        model.train()
        if args.optim.name == 'adamwschedulefree':
            optimizer.train()

def maybe_logging(averager, args, model, optimizer, logger):
    if args.current_train_step % args.logging.every_steps == 0:
        stats = extra_stats(args, model, optimizer)

        averager.update(stats)
        averaged_stats = averager.average()

        logger.log_stats(
            stats=averaged_stats,
            step=args.current_train_step,
            args=args,
            prefix='train/'
        )

        args.last_log = time.time()

        # gather_running_statistics(model)

def gather_running_statistics(model, prefix="", stats=dict()):
    for name, module in model.named_children():
        qual_name = prefix + "." + name
        module.__qualname__ = qual_name
        if isinstance(module, T5SequenceNorm):
            stats[qual_name + ".mean"] = module.running_mean
            stats[qual_name + ".var"] = module.running_var
        else:
            gather_running_statistics(module, prefix=qual_name, stats=stats)
    return stats



def maybe_grad_clip_and_grad_calc(accelerator, model, args):
    if args.optim.grad_clip > 0:
        grad_l2 = accelerator.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=args.optim.grad_clip,
            norm_type=2,
        )
    else:
        grad_l2 = None

    if args.logging.grad_l2:
        if grad_l2 is None:
            grad_l2 = (
                sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
            )

        return {'grad_l2': grad_l2}
    else:
        return {}


def extra_stats(args, model, optimizer):
    stats = {}

    if args.logging.weights_l2:
        weights_l2 = sum(p.detach().norm(2).item() ** 2 for p in model.parameters()) ** 0.5
        stats['weights_l2'] = weights_l2

    stats['lr'] = optimizer.param_groups[0]['lr']
    stats['seconds_per_step'] = (time.time() - args.last_log) / args.logging.every_steps

    return stats


def forward(model, batch, args, tokenizer, calc_acc, eval):
    outputs = model(**batch)
    loss = outputs.loss

    stats = {}
    stats['loss'] = loss.detach().float().item()

    if calc_acc:
        correct = (outputs.logits.argmax(-1) == batch["labels"]).sum().item()
        accuracy = correct / batch["labels"].numel()
        stats['accuracy'] = accuracy

    if args.current_train_step % args.eval.every_steps == 0:
        # print(f"INPUT    : {tokenizer.decode(batch['input_ids'][0])}")
        # print(f"LABEL    : {tokenizer.decode(batch['labels'][0])}")
        # print(f"PREDICTED: {tokenizer.decode(outputs.logits.argmax(-1)[0])}")
        pass

    return loss, stats


def eval(model, dataloader, logger, args, tokenizer, accelerator):
    args.last_log = time.time()
    averager = Averager()

    for batch_id, batch in enumerate(dataloader, start=0):
        if batch_id == args.eval.corrected_steps * args.optim.grad_acc:
            break

        _, stats = forward(model, batch, args, tokenizer, calc_acc=True, eval=True)
        averager.update(stats)

    averager.update({'time': time.time() - args.last_log})
    averaged_stats = averager.average()
    # print(f"averaged_stats: {averaged_stats}")
    if args.wandb is not None:
        if accelerator.is_main_process:
            wandb.log({"eval_loss": averaged_stats["loss"]})
        accelerator.wait_for_everyone()
    aimrun.track({"eval_loss": averaged_stats["loss"]}, step=args.current_train_step, epoch=args.current_epoch)

    logger.log_stats(
        stats=averaged_stats,
        step=args.current_train_step,
        args=args,
        prefix='eval/'
    )


def predict(model, dataloader, logger, args, tokenizer):
    args.last_log = time.time()
    metric = evaluate.load('rouge')
    samples_seen = 0

    def decode(preds):
        preds[preds == -100] = tokenizer.pad_token_id
        preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        preds = [pred.strip() for pred in preds]
        return preds

    for step, batch in enumerate(dataloader):
        predictions = model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=args.data.max_target_len,
            generation_config=model.generation_config,
        )
        predictions = decode(predictions)
        references = decode(batch["labels"])
        # print('\n'.join(repr(x) for x in zip(predictions, references)))

        # If we are in a multiprocess environment, the last batch has duplicates
        if step == len(dataloader) - 1:
            predictions = predictions[: len(dataloader.dataset) - samples_seen]
            references = references[: len(dataloader.dataset) - samples_seen]
        else:
            samples_seen += len(references)

        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute(use_stemmer=True, use_aggregator=False)
    rougeL = sum(eval_metric["rougeL"]) * 100 / len(eval_metric["rougeL"])

    logger.log_stats(
        stats={
            "rougeL": rougeL,
            "time": time.time() - args.last_log,
        },
        step=args.current_train_step,
        args=args,
        prefix="test/",
    )


def train(model, train_dataloader, test_dataloader, accelerator, lr_scheduler,
          optimizer, logger, args, tokenizer):
    model.train()
    if args.optim.name == 'adamwschedulefree':
        optimizer.train()

    if hasattr(args.optim, "skip_steps") and args.optim.skip_steps > 0:
        print(f"Skipping {args.optim.skip_steps} steps.")
        while args.current_train_step <= args.optim.skip_steps:
            for batch_id, batch in enumerate(train_dataloader, start=1):
                if args.current_train_step > args.optim.skip_steps:
                    break
                if batch_id % args.optim.grad_acc == 0:
                    args.current_train_step += 1
                    if args.current_train_step % args.logging.every_steps == 0:
                        print(f"Skipping step {args.current_train_step}.")
            else:
                print(f"At end of epoch {args.current_epoch} after {args.current_train_step} skipped steps.")
                args.current_epoch += 1
        print(f"Skipping done, starting at step {args.current_train_step}.")
        if args.quantization_warmup_steps is not None:
            current_step = args.current_train_step-args.quantization_warmup_offset
            if current_step == 0:
                if args.quantization_warmup_prequantize:
                    print(f"Model was pre-quantized to bitlinear - doing nothing at step {args.current_train_step}, steeing lambda_ to 0.0.")
                else:
                    bitlinearize(model, replacements=args.bitlinear)
                    print(f"Quantized to bitlinear at step {args.current_train_step}, setting lambda_ to 0.0.")
                set_lambda_(model, lambda_=0.0)
            elif current_step < args.quantization_warmup_steps:
                def sigmoid(x):
                    return 1 / (1 + math.exp(-x))
                lambda_ = 2*(sigmoid(current_step*5/args.quantization_warmup_steps))-1
                if args.current_train_step % args.logging.every_steps == 0 and batch_id % args.optim.grad_acc == 0:
                    print(f"Setting lambda_ to {lambda_} for step {args.current_train_step}.")
                set_lambda_(model, lambda_=lambda_)
            elif current_step >= args.quantization_warmup_steps:
                #if args.current_train_step % args.logging.every_steps == 0 and batch_id % args.optim.grad_acc == 0:
                print(f"Finishing warmup at step {args.current_train_step}, setting lambda_ to 1.0.")
                set_lambda_(model, lambda_=1.0)
                print(model)

    train_averager = Averager()
    step_averager = Averager()

    while args.current_train_step <= args.optim.total_steps:
        if isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(args.current_epoch)

        # In case there is a remainder from previous epoch, we need to reset the optimizer
        optimizer.zero_grad(set_to_none=True)

        for batch_id, batch in enumerate(train_dataloader, start=1):
            if args.current_train_step > args.optim.total_steps:
                break

            # Set quantization rate lambda_.
            if args.quantization_warmup_steps is not None:
                current_step = args.current_train_step-args.quantization_warmup_offset
                if current_step < 0:
                    if args.current_train_step % args.logging.every_steps == 0 and batch_id % args.optim.grad_acc == 0:
                        print(f"Waiting for quantization warmup from {args.quantization_warmup_offset} at step {args.current_train_step} steps, setting lambda_ to 0.0.")
                    set_lambda_(model, lambda_=0.0)
                elif current_step == 0:
                    if args.quantization_warmup_prequantize:
                        print(f"Model was pre-quantized to bitlinear - doing nothing at step {args.current_train_step}, steeing lambda_ to 0.0.")
                    else:
                        bitlinearize(model, replacements=args.bitlinear)
                        print(f"Quantized to bitlinear at step {args.current_train_step}, setting lambda_ to 0.0.")
                    set_lambda_(model, lambda_=0.0)
                elif current_step < args.quantization_warmup_steps:
                    def sigmoid(x):
                        return 1 / (1 + math.exp(-x))
                    lambda_ = 2*(sigmoid(current_step*5/args.quantization_warmup_steps))-1
                    if args.current_train_step % args.logging.every_steps == 0 and batch_id % args.optim.grad_acc == 0:
                        print(f"Setting lambda_ to {lambda_} for step {args.current_train_step}.")
                    set_lambda_(model, lambda_=lambda_)
                elif current_step == args.quantization_warmup_steps:
                    #if args.current_train_step % args.logging.every_steps == 0 and batch_id % args.optim.grad_acc == 0:
                    print(f"Finishing warmup at step {args.current_train_step}, setting lambda_ to 1.0.")
                    set_lambda_(model, lambda_=1.0)
                    print(model)

            loss, stats = forward(model, batch, args, tokenizer, calc_acc=True, eval=False)
            accelerator.backward(loss / args.optim.grad_acc)
            train_averager.update(stats)
            step_averager.update({"train_loss": stats["loss"], "train_accuracy": stats["accuracy"]})

            if batch_id % args.optim.grad_acc == 0:
                avg_stats = step_averager.average()
                if args.wandb is not None:
                    if accelerator.is_main_process:
                        wandb.log(avg_stats)
                    accelerator.wait_for_everyone()
                aimrun.track(avg_stats, step=args.current_train_step, epoch=args.current_epoch)
                stats = maybe_grad_clip_and_grad_calc(accelerator, model, args)
                train_averager.update(stats)

                optimizer.step()
                lr_scheduler.step() if args.optim.lr_scheduler != 'reduce_on_plateau' else lr_scheduler.step(avg_stats["train_loss"])
                optimizer.zero_grad(set_to_none=True)

                maybe_logging(train_averager, args, model, optimizer, logger)
                maybe_eval_predict(model, test_dataloader, logger, args, tokenizer, accelerator, optimizer)
                maybe_save_checkpoint(model, accelerator, optimizer, args)

                args.current_train_step += 1

        args.current_epoch += 1

    maybe_eval_predict(model, test_dataloader, logger, args, tokenizer, accelerator, optimizer)
    maybe_save_checkpoint(model, accelerator, optimizer, args)
