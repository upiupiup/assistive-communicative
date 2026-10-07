import sys, json, time


def main():
    cfg = json.load(open(sys.argv[1], encoding="utf-8"))
    res = {"model_id": cfg["model_id"], "batch_size": cfg["batch_size"], "ok": False}
    try:
        import torch
        from transformers import WhisperForConditionalGeneration
        from peft import LoraConfig, get_peft_model

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = WhisperForConditionalGeneration.from_pretrained(
            cfg["model_id"], torch_dtype=torch.float16
        )
        model.config.forced_decoder_ids = None
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        lcfg = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lcfg)
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        model.to("cuda")

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        mel = model.config.num_mel_bins
        B = cfg["batch_size"]
        feats = torch.randn(B, mel, 3000, dtype=torch.float16, device="cuda")
        labels = torch.randint(100, 5000, (B, 64), device="cuda")

        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        t0 = time.time()
        for _ in range(3):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_features=feats, labels=labels).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_s = (time.time() - t0) / 3

        res.update(
            ok=True,
            num_mel_bins=mel,
            total_params_m=round(total / 1e6, 1),
            trainable_params_m=round(trainable / 1e6, 2),
            trainable_pct=round(100 * trainable / total, 3),
            peak_alloc_mb=round(torch.cuda.max_memory_allocated() / 1024**2),
            peak_reserved_mb=round(torch.cuda.max_memory_reserved() / 1024**2),
            sec_per_step=round(step_s, 2),
        )
    except Exception as e:
        res.update(ok=False, error=(type(e).__name__ + ": " + str(e))[:400])
    sys.stdout.write("###RESULT###" + json.dumps(res))


main()
