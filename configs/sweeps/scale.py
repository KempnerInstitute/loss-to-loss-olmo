import yaml
import numpy as np
from olmo.model import OLMo
from olmo.config import ModelConfig

train_sets = [
    "fineweb-100b",
    "starcoder",
    "proof-pile-2",  # ~50B toks
    "fineweb-edu-100b",
    "slimpajama-chunk1",  # ~65B toks
    "smollm-corpus",  # ~60B toks
]

sweep_config = {
    "wandb": {"group": "new-scale-big-1"},
    "sweep": [],
    "save_num_checkpoints_to_keep": 1,
    "save_num_unsharded_checkpoints_to_keep": 1,
    "save_interval": 1000,
    "save_interval_unsharded": 100000,
    "data": {
        "paths": train_sets,
    },
    "global_train_batch_size": 1024,
}


iso_flops = [int(n) for n in np.geomspace(2e17, 1e19, 6)] + [int(2.2e19), int(4.84e19), int(1e21)]

model_sizes = [
    (256, 4),
    (320, 5),
    (384, 6),
    (448, 7),  # 50M
    (512, 8),
    (576, 9),
    (640, 10),
    (704, 11),
    (768, 12),
    (832, 13),
    (896, 14),
    (960, 15),
    (1024, 16),  # 260M
    (1088, 17),
    (1152, 18),
    (1216, 19),  # 450M
    (1280, 20),
    (1344, 21),
    (1408, 22),  # 610M
    (1536, 24),
    (1664, 26),
    (1792, 28),
    (1920, 30),
    (2048, 32),  # 1.7B
    (2176, 34),
    (2304, 36),  # 2.4B
    (2432, 38),  # 2.9B
    (2560, 40),  # 3.3B
]


model_defaults = {
    "max_sequence_length": 512,
    "mlp_ratio": 4,
    "rope": True,
    "attention_layer_norm": True,
    "attention_layer_norm_with_affine": True,
    "multi_query_attention": False,
    "include_bias": False,
    "block_type": "sequential",
    "layer_norm_type": "default",
    "layer_norm_with_affine": True,
    "bias_for_layer_norm": False,
    "activation_type": "gelu",
    "attention_dropout": 0.0,
    "residual_dropout": 0.0,
    "embedding_dropout": 0.0,
    "vocab_size": 32000,
    "embedding_size": 32000,
    "eos_token_id": 2,
    "pad_token_id": 2,
    "init_device": "meta",
    "init_fn": "mitchell",
    "weight_tying": False,
}


def get_model_config_and_size(d_model, n_layers):
    head_size = 64

    model_config = ModelConfig(**model_defaults)
    model_config.d_model = d_model
    model_config.n_layers = n_layers
    model_config.n_heads = d_model // head_size
    model = OLMo(model_config, False)
    params = model.num_params()
    fwd_flops = model.num_fwd_flops

    # Hard code batch rules
    # if params < 50e6:
    #     bs = 64
    # elif params < 300e6:
    #     bs = 32

    # Rough estimate of proper batch size (note: note quite right)
    # TODO: fix this approximation
    seq_len = model_defaults["max_sequence_length"]
    activation_memory = 2 * (n_layers + 1) * (seq_len * d_model) * (4 + 4 + 2)  # Attention + ff hidden + norm
    attention_memory = 2 * n_layers * seq_len**2 * d_model // head_size
    output_memory = 2 * 4 * seq_len * 32000  # vocab size
    model_memory = 3 * 4 * params
    # Using 75 GB to allow for fudge factor
    bs = int((70e9 - model_memory) / (activation_memory + attention_memory + output_memory))
    bs = min(2 ** int(np.log2(bs)), 128)  # Otherwise small models fo OOM
    # bs = 2 ** int(np.log2(bs))
    print(f"Computed batch size: {bs}")
    print(f"Memory: {(model_memory + bs * (activation_memory+attention_memory+output_memory)) / 1e9} GB")
    print(
        f"Breakdown: {model_memory / 1e9} GB, {bs * activation_memory / 1e9} GB, {bs * attention_memory / 1e9} GB, {bs * output_memory / 1e9} GB"
    )

    print(f"Model size: {params} parameters, {fwd_flops} flops, (d: {d_model}, l: {n_layers})")

    return (
        params,
        fwd_flops,
        {
            "model": {
                "d_model": d_model,
                "n_heads": d_model // head_size,
                "n_layers": n_layers,
                "max_sequence_length": model_defaults["max_sequence_length"],
            },
            "device_train_microbatch_size": bs,
            "device_eval_batch_size": min(bs, 64),  # Avoid OOM in eval bc no compiler
        },
    )


def expand_config(
    total_flops,
    params,
    fwd_flops,
    config,
    global_bs=sweep_config["global_train_batch_size"],
    seq_len=model_defaults["max_sequence_length"],
):
    tok_bs = seq_len * global_bs

    steps = int(total_flops / (6 * params) / tok_bs)

    # steps = int(total_flops / (3 * fwd_flops) / global_bs)

    tokens = steps * tok_bs

    config["total_flops"] = total_flops
    config["max_duration"] = steps
    config["scheduler"] = {"t_warmup": steps // 5}
    config["eval_interval"] = steps // 5
    print(f"FLOPs: {float(flops)}, tokens: {tokens}, steps: {steps}")
    return config, tokens, steps


if __name__ == "__main__":
    max_tokens = 0
    for flops in iso_flops:
        for d, l in model_sizes:
            if flops == 1e21 and l != 40:
                continue

            params, fwd_flops, config = get_model_config_and_size(d, l)
            config, tokens, steps = expand_config(flops, params, fwd_flops, config)
            data_ratio = tokens / params  # expect to be around 20
            print(f"Data ratio: {data_ratio}")
            if data_ratio > 2 and data_ratio < 200:  # 20 +/- 10x -> 570 jobs
                sweep_config["sweep"].append(config)
                print(f"Added config!")
                max_tokens = max(max_tokens, tokens)

    print(f"Total sweep size: {len(sweep_config['sweep']) * len(train_sets)}")
    print(f"Max tokens needed: {max_tokens}")

    with open("configs/sweeps/scale.yaml", "w") as f:
        yaml.dump(sweep_config, f)
