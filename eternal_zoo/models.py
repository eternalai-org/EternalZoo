HASH_TO_MODEL = {
    "bafkreiacd5mwy4a5wkdmvxsk42nsupes5uf4q3dm52k36mvbhgdrez422y": "qwen3-embedding-0.6b",
    "bafkreia7nzedkxlr6tebfxvo552zq7cba6sncloxwyivfl3tpj7hl5dz5u": "qwen3-embedding-4b",
    "bafkreib6pws5dx5ur6exbhulmf35twfcizdkxvup4cklzprlvaervfz5zy": "qwen3-1.7b",
    "bafkreiekokvzioogj5hoxgxlorqvbw2ed3w4mwieium5old5jq3iubixza": "qwen3-4b",
    "bafkreid5z4lddvv4qbgdlz2nqo6eumxwetwmkpesrumisx72k3ahq73zpy": "qwen3-8b",
    "bafkreiclwlxc56ppozipczuwkmgnlrxrerrvaubc5uhvfs3g2hp3lftrwm": "qwen3-14b",
    "bafkreihq4usl2t3i6pqoilvorp4up263yieuxcqs6xznlmrig365bvww5i": "qwen3-32b",
    "bafkreieroiopteqmtbjadlnpq3qkakdu7omvtuavs2l2qbu46ijnfdo2ly": "qwen3-30b-a3b",
    "bafkreie4uj3gluik5ob2ib3cm2pt6ww7n4vqpmjnq6pas4gkkor42yuysa": "qwen3-235b-a22b",
    "bafkreib6thkvzddxkxtgkeslioreae66uef42gtxzy4wh7cyzf6fmlq3rm": "qwen3-coder-480b-a35b",
    "bafkreiaevddz5ssjnbkmdrl6dzw5sugwirzi7wput7z2ttcwnvj2wiiw5q": "gemma-3-4b",
    "bafkreic2bkjuu3fvdoxnvusdt4in6fa6lubzhtjtmcp2zvokvfjpyndakq": "gemma-3-12b",
    "bafkreihi2cbsgja5dwa5nsuixicx2x3gbcnh7gsocxbmjxegtewoq2syve": "gemma-3-27b",
    "bafkreihz3mz422vpoy7sccwj5tujkerxbjxdlmsqqf3ridxbe3m6ipnq5i": "gemma-3n-e4b",
    "bafkreiaztifhss23cftya3bkorbsenzohol2oc3dvngo2srbbosko6gmme": "lfm2-1.2b",
    "bafkreidrdplo7mcfhrvocaa26yge6kmxmuwrexm5rffnzo5lbe6fkhjuvq": "openreasoning-nemotron-32b",
    "bafkreih4xgr5t7yc3yooz6i6usgpwhggaobspmgut4rnu42gi6cv77o4em": "devstral-small",
    "bafkreibokz6tdke7k3eozsro3hh3luyqbub7tzdawpswtt7q6bzfg36fw4": "dolphin-3.0-llama3.1-8b",
    "bafkreiaha3sjfmv4affmi5kbu6bnayenf2avwafp3cthhar3latmfi632u": "flux-dev",
    "bafkreibks5pmc777snbo7dwk26sympe2o24tpqfedjq6gmgghwwu7iio34": "flux-schnell",
    "bafkreidbaksrogxispjejczfj36vtf5uzsbjt7irspl6kckynz5u2ugzke": "flux-dev-nsfw",
    "bafkreihiaeosw2jlyvzo7od46ihe4iwutgmppqj5d7z74g25qljlcmcikq": "flux-dev-18-loras",
    "bafkreidnd2n2sp3gw6c4iutvgdtupqa4qlpsznpjnwqmsna2ko3uhv4fce": "nsfw-lab",
    "bafkreidl2y42rs2ymhydn7gojikgv657yy73yldu3nanjsljeepen6ftsy": "lora-lab"
}

MODEL_TO_HASH = {model: hash for hash, model in HASH_TO_MODEL.items()}

FEATURED_MODELS = {
    "qwen3-embedding-0.6b": {
        "repo": "Qwen/Qwen3-Embedding-0.6B-GGUF",
        "model": "Qwen3-Embedding-0.6B-Q8_0.gguf",
        "task": "embed"
    },
    "qwen3-embedding-4b": {
        "repo": "Qwen/Qwen3-Embedding-4B-GGUF",
        "model": "Qwen3-Embedding-4B-Q8_0.gguf",
        "task": "embed"
    },
    "qwen3-1.7b": {
        "repo": "Qwen/Qwen3-1.7B-GGUF",
        "model": "Qwen3-1.7B-Q8_0.gguf",
        "task": "chat"
    },
    "qwen3-4b": {
       "repo": "Qwen/Qwen3-4B-GGUF",
       "model": "Qwen3-4B-Q8_0.gguf",
       "task": "chat"
    },
    "qwen3-8b": {
        "repo": "Qwen/Qwen3-8B-GGUF",
        "model": "Qwen3-8B-Q8_0.gguf",
        "task": "chat"
    },
    "qwen3-14b": {
        "repo": "Qwen/Qwen3-14B-GGUF",
        "model": "Qwen3-14B-Q8_0.gguf",
        "task": "chat"
    },
    "qwen3-32b": {
        "repo": "Qwen/Qwen3-32B-GGUF",
        "model": "Qwen3-32B-Q8_0.gguf",
        "task": "chat"
    },
    "qwen3-30b-a3b": {
        "repo": "Qwen/Qwen3-30B-GGUF",
        "model": "Qwen3-30B-A3B-Q8_0.gguf",
        "task": "chat"
    },
    "qwen3-30b-a3b-instruct-2507": {
        "repo": "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
        "model": "Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf",
        "task": "chat"
    },
    "qwen3-30b-a3b-thinking-2507": {
        "repo": "unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF",
        "model": "Qwen3-30B-A3B-Thinking-2507-Q8_0.gguf",
        "task": "chat"
    },
    "qwen3-coder-30b-a3b-instruct": {
        "repo": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        "model": "Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf",
        "task": "chat"
    },
    "qwen3-235b-a22b": {
        "repo": "unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF",
        "pattern": "Q4_K_M",
        "task": "chat"
    },
    "qwen3-coder-480b-a35b": {
        "repo": "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF",
        "pattern": "Q4_K_M",
        "task": "chat"
    },
    "gemma-3-4b": {
        "repo": "lmstudio-community/gemma-3-4B-it-qat-GGUF",
        "model": "gemma-3-4B-it-QAT-Q4_0.gguf",
        "projector": "mmproj-model-f16.gguf",
        "task": "chat"
    },
    "gemma-3-12b": {
        "repo": "lmstudio-community/gemma-3-12B-it-qat-GGUF",
        "model": "gemma-3-12B-it-QAT-Q4_0.gguf",
        "projector": "mmproj-model-f16.gguf",
        "task": "chat"
    },
    "gemma-3-27b": {
        "repo": "lmstudio-community/gemma-3-27B-it-qat-GGUF",
        "model": "gemma-3-27B-it-QAT-Q4_0.gguf",
        "projector": "mmproj-model-f16.gguf",
        "task": "chat"
    },
    "gemma-3n-e4b": {
        "repo": "unsloth/gemma-3n-E4B-it-GGUF",
        "model": "gemma-3n-E4B-it-Q8_0.gguf",
        "task": "chat"
    },
    "lfm2-1.2b": {
        "repo": "LiquidAI/LFM2-1.2B-GGUF",
        "model": "LFM2-1.2B-Q8_0.gguf",
        "task": "chat"
    },
    "openreasoning-nemotron-32b": {
        "repo": "lmstudio-community/OpenReasoning-Nemotron-32B-GGUF",
        "model": "OpenReasoning-Nemotron-32B-Q8_0.gguf",
        "task": "chat"
    },
    "devstral-small": {
        "repo": "mistralai/Devstral-Small-2507_gguf",
        "model": "Devstral-Small-2507-Q8_0.gguf",
        "task": "chat"
    },
    "dolphin-3.0-llama3.1-8b": {
        "repo": "dphn/Dolphin3.0-Llama3.1-8B-GGUF",
        "model": "Dolphin3.0-Llama3.1-8B-Q8_0.gguf",
        "task": "chat"
    },
    "flux-dev": {
       "repo": "NikolaSigmoid/FLUX.1-dev",
       "task": "image-generation",
       "architecture": "flux-dev"
    },
    "flux-schnell": {
        "repo": "NikolaSigmoid/FLUX.1-schnell",
        "task": "image-generation",
        "architecture": "flux-schnell"
    },
    "flux-krea-dev": {
        "repo": "NikolaSigmoid/FLUX.1-Krea-dev",
        "task": "image-generation",
        "architecture": "flux-dev"
    },
    "flux-dev-nsfw": {
        "repo": "NikolaSigmoid/FLUX.1-dev-NSFW-Master",
        "task": "image-generation",
        "lora": True,
        "base_model": "flux-dev"
    },
    "flux-dev-18-loras": {
        "repo": "NikolaSigmoid/FLUX.1-dev-18-loras",
        "task": "image-generation",
        "lora": True,
        "base_model": "flux-dev"
    },
    "nsfw-lab": {
       "repo": "NikolaSigmoid/NSFW-Lab",
       "task": "image-generation",
       "lora": True,
       "base_model": "flux-schnell"
    },
    "lora-lab": {
        "repo": "NikolaSigmoid/lora-lab",
        "task": "image-generation",
        "lora": True,
        "base_model": "flux-dev"
    }
}