from src.architectures import arch_1l, arch_2l, arch_3l, arch_4l, arch_5l

ARCHITECTURE_REGISTRY = {
    "overlap": {
        1: {
            "con": arch_1l.arch_con_1l,
            "cat": arch_1l.arch_cat_1l,
            "bin": arch_1l.arch_cat_1l,
        },
        2: {
            "con": arch_2l.arch_con_2l,
            "cat": arch_2l.arch_cat_2l,
            "bin": arch_2l.arch_cat_2l,
        },
        3: {
            "con": arch_3l.arch_con_3l,
            "cat": arch_3l.arch_cat_3l,
            "bin": arch_3l.arch_cat_3l,
        },
        
        4: {
            "con": arch_4l.arch_con_4l,
            "cat": arch_4l.arch_cat_4l,
            "bin": arch_4l.arch_cat_4l,
        },

        5: {
            "con": arch_5l.arch_con_5l,
            "cat": arch_5l.arch_cat_5l,
            "bin": arch_5l.arch_cat_5l,
        },
    },
}

CONV_PARAMS = {
    "MNIST": {
        "overlap": {
            "14x14": {
                1: {
                    "encoder": [dict(kernel_size=4, stride=2, padding=1)],# 28 -> 14
                    "decoder": [dict(kernel_size=4, stride=2, padding=1)], 
                },
                4: {
                    "encoder": [
                        dict(kernel_size=4, stride=2, padding=1),  # 28 → 14
                        dict(kernel_size=3, stride=1, padding=1),  # 14 → 14
                        dict(kernel_size=3, stride=1, padding=1),  # 14 → 14
                        dict(kernel_size=1, stride=1, padding=0),  # 14 → 14
                    ],
                    "decoder": [
                        dict(kernel_size=1, stride=1, padding=0),  # 14 → 14
                        dict(kernel_size=3, stride=1, padding=1),  # 14 → 14
                        dict(kernel_size=3, stride=1, padding=1),  # 14 → 14
                        dict(kernel_size=4, stride=2, padding=1),  # 14 → 28
                    ],
                }
            },
            "7x7": {
                2: {
                    "encoder": [dict(kernel_size=4, stride=2, padding=1)] * 2,
                    "decoder": [dict(kernel_size=4, stride=2, padding=1)] * 2, 
                },
                3: {
                    "encoder": [
                        dict(kernel_size=4, stride=2, padding=1),  # 32→16
                        dict(kernel_size=4, stride=2, padding=1),  # 16→8
                        dict(kernel_size=3, stride=1, padding=1),  # 8→8 (capacity without shrinking)
                    ],
                    "decoder": [
                        dict(kernel_size=4, stride=2, padding=1),  # 32→16
                        dict(kernel_size=4, stride=2, padding=1),  # 16→8
                        dict(kernel_size=3, stride=1, padding=1),  # 8→8 (capacity without shrinking)
                    ]
                },
                4: {
                    "encoder": [
                        dict(kernel_size=4, stride=2, padding=1),  # 28 → 14
                        dict(kernel_size=4, stride=2, padding=1),  # 14 → 7
                        dict(kernel_size=3, stride=1, padding=1),  # 7 → 7 (no change)
                        dict(kernel_size=1, stride=1, padding=0),  # 7 → 7 (no change)
                    ],
                    "decoder": [ 
                        dict(kernel_size=1, stride=1, padding=0),       # 7 → 7
                        dict(kernel_size=3, stride=1, padding=1),       # 7 → 7
                        dict(kernel_size=4, stride=2, padding=1),       # 7 → 14
                        dict(kernel_size=4, stride=2, padding=1),       # 14 → 28
                    ],
                },
            },
            "4x4": {
                3: {
                    "encoder": [dict(kernel_size=4, stride=2, padding=1)] * 2
                            + [dict(kernel_size=3, stride=2, padding=1)],
                    "decoder": [dict(kernel_size=3, stride=2, padding=1)]
                            + [dict(kernel_size=4, stride=2, padding=1)] * 2,
                },
                4: {
                    "encoder": [
                        dict(kernel_size=4, stride=2, padding=1),  # 28 → 14
                        dict(kernel_size=4, stride=2, padding=1),  # 14 → 7
                        dict(kernel_size=4, stride=2, padding=2),  # 7 → 4
                        dict(kernel_size=3, stride=1, padding=1),  # 4 → 4
                    ],

                    "decoder": [ 
                        dict(kernel_size=3, stride=1, padding=1),                         # 4 → 4
                        dict(kernel_size=4, stride=2, padding=2, output_padding=1),       # 4 → 7
                        dict(kernel_size=4, stride=2, padding=1, output_padding=0),       # 7 → 14
                        dict(kernel_size=4, stride=2, padding=1, output_padding=0),       # 14 → 28 ✅
                    ],
                }
            },
            "2x2": {
                4: {
                    "encoder": [dict(kernel_size=4, stride=2, padding=1)] * 2
                             + [dict(kernel_size=3, stride=2, padding=1),
                                dict(kernel_size=2, stride=2, padding=0)],
                    "decoder": [dict(kernel_size=2, stride=2, padding=0),
                                dict(kernel_size=3, stride=2, padding=1)]
                             + [dict(kernel_size=4, stride=2, padding=1)] * 2 ,
                }
            },
            "1x1": {
                4: {
                    "encoder": [
                            dict(kernel_size=4, stride=2, padding=1),  # 28 → 14
                            dict(kernel_size=4, stride=2, padding=1),  # 14 → 7
                            dict(kernel_size=3, stride=2, padding=1),  # 7 → 4
                            dict(kernel_size=4, stride=4, padding=0),  # 4 → 1
                        ],
                    "decoder": [
                            dict(kernel_size=4, stride=4, padding=0),                          # 1 → 4
                            dict(kernel_size=3, stride=2, padding=1, output_padding=0),       # 4 → 7
                            dict(kernel_size=4, stride=2, padding=1, output_padding=0),       # 7 → 14
                            dict(kernel_size=4, stride=2, padding=1, output_padding=0),       # 14 → 28
                        ],
                },
                5: {
                    "encoder": [dict(kernel_size=4, stride=2, padding=1)] * 2
                             + [dict(kernel_size=3, stride=2, padding=1)] 
                             + [dict(kernel_size=2, stride=2, padding=0)] * 2,

                    "decoder": [dict(kernel_size=2, stride=2, padding=0)] * 2
                             + [dict(kernel_size=3, stride=2, padding=1, output_padding=0)]
                             + [dict(kernel_size=4, stride=2, padding=1)] * 2 ,
                }
            }
            
        },
        "tiles": {
            2: {
                "encoder": [dict(kernel_size=2, stride=2, padding=0)] * 2,
                "decoder": [dict(kernel_size=2, stride=2, padding=0)] * 2, 
            },
            3: {
                "encoder": [
                    dict(kernel_size=2, stride=2, padding=0),
                    dict(kernel_size=2, stride=2, padding=0),
                    dict(kernel_size=2, stride=2, padding=1),
                ],
                "decoder": [ 
                    dict(kernel_size=2, stride=2, padding=0, output_padding=1),
                    dict(kernel_size=2, stride=2, padding=0),
                    dict(kernel_size=2, stride=2, padding=1),
                ],
            },
        },
    },
    "SVHN": {
        "overlap": {
            "16x16": {
                4: {
                    "encoder": [
                        dict(kernel_size=4, stride=2, padding=1),  # 32 → 16
                        dict(kernel_size=3, stride=1, padding=1),
                        dict(kernel_size=3, stride=1, padding=1),
                        dict(kernel_size=1, stride=1, padding=0),
                    ],
                    "decoder": [
                        dict(kernel_size=1, stride=1, padding=0),
                        dict(kernel_size=3, stride=1, padding=1),
                        dict(kernel_size=3, stride=1, padding=1),
                        dict(kernel_size=4, stride=2, padding=1),  # 16 → 32
                    ],
                }
            },
            "8x8": {
                2: {
                    "encoder": [dict(kernel_size=4, stride=2, padding=1)] * 2,
                    "decoder": [dict(kernel_size=4, stride=2, padding=1)] * 2, # Identical to encoder for now
                },
                3: {
                    "encoder": [
                        dict(kernel_size=4, stride=2, padding=1),  # 32→16
                        dict(kernel_size=4, stride=2, padding=1),  # 16→8
                        dict(kernel_size=3, stride=1, padding=1),  # 8→8 (capacity without shrinking)
                    ],
                    "decoder": [
                        dict(kernel_size=4, stride=2, padding=1),  # 32→16
                        dict(kernel_size=4, stride=2, padding=1),  # 16→8
                        dict(kernel_size=3, stride=1, padding=1),  # 8→8 (capacity without shrinking)
                    ]
                },
                4: {
                    "encoder": [
                        dict(kernel_size=4, stride=2, padding=1),
                        dict(kernel_size=4, stride=2, padding=1),
                        dict(kernel_size=3, stride=1, padding=1),
                        dict(kernel_size=1, stride=1, padding=0),
                    ],
                    "decoder": [ 
                        dict(kernel_size=1, stride=1, padding=0),
                        dict(kernel_size=3, stride=1, padding=1),
                        dict(kernel_size=4, stride=2, padding=1),
                        dict(kernel_size=4, stride=2, padding=1),
                    ],
                },
            },
            "4x4": {
                3: {
                    "encoder": [dict(kernel_size=4, stride=2, padding=1)] * 3,
                    "decoder": [dict(kernel_size=4, stride=2, padding=1)] * 3, # Identical to encoder for now
                },
                4: {
                    "encoder": [
                        dict(kernel_size=4, stride=2, padding=1),
                        dict(kernel_size=4, stride=2, padding=1),
                        dict(kernel_size=4, stride=2, padding=1),
                        dict(kernel_size=3, stride=1, padding=1),
                    ],
                    "decoder": [ 
                        dict(kernel_size=3, stride=1, padding=1),
                        dict(kernel_size=4, stride=2, padding=1),
                        dict(kernel_size=4, stride=2, padding=1),
                        dict(kernel_size=4, stride=2, padding=1),
                    ],
                },
            },
            "2x2": {
                4: {
                    "encoder": [dict(kernel_size=4, stride=2, padding=1)] * 3
                            + [dict(kernel_size=2, stride=2, padding=0)],
                    "decoder": [dict(kernel_size=4, stride=2, padding=1)] * 3 # Identical to encoder for now
                            + [dict(kernel_size=2, stride=2, padding=0)],
                },
            },
            "1x1": {
                4: {
                    "encoder": [dict(kernel_size=4, stride=2, padding=1)] * 3
                            + [dict(kernel_size=4, stride=4, padding=0)],
                    "decoder": [dict(kernel_size=4, stride=2, padding=1)] * 3 # Identical to encoder for now
                            + [dict(kernel_size=4, stride=4, padding=0)],
                },
                5: {
                    "encoder": [dict(kernel_size=4, stride=2, padding=1)] * 5,

                    "decoder": [dict(kernel_size=4, stride=2, padding=1)] * 5,
                }
            }
        },
        "tiles": {
            2: {
                "encoder": [dict(kernel_size=2, stride=2, padding=0)] * 2,
                "decoder": [dict(kernel_size=2, stride=2, padding=0)] * 2, # Identical to encoder for now
            },
            3: {
                "encoder": [dict(kernel_size=2, stride=2, padding=0)] * 3,
                "decoder": [dict(kernel_size=2, stride=2, padding=0)] * 3, # Identical to encoder for now
            },
        },
    },
    "celeba": {
        "overlap": {
            "16x16": {
                4: {
                "encoder": [
                        dict(kernel_size=4, stride=2, padding=1),
                        dict(kernel_size=4, stride=2, padding=1),
                        dict(kernel_size=4, stride=2, padding=1),
                        dict(kernel_size=3, stride=1, padding=1),
                    ],
                "decoder": [ 
                        dict(kernel_size=3, stride=1, padding=1),
                        dict(kernel_size=4, stride=2, padding=1),
                        dict(kernel_size=4, stride=2, padding=1),
                        dict(kernel_size=4, stride=2, padding=1),
                    ],
                },
            },
            "32x32": {
                2: {
                    "encoder": [dict(kernel_size=4, stride=2, padding=1)] * 2,
                    "decoder": [dict(kernel_size=4, stride=2, padding=1)] * 2, # Identical to encoder for now
                },
                4: {
                "encoder": [
                        dict(kernel_size=4, stride=2, padding=1),
                        dict(kernel_size=4, stride=2, padding=1),
                        dict(kernel_size=3, stride=1, padding=1),
                        dict(kernel_size=1, stride=1, padding=0),
                    ],
                "decoder": [ 
                        dict(kernel_size=1, stride=1, padding=0),
                        dict(kernel_size=3, stride=1, padding=1),
                        dict(kernel_size=4, stride=2, padding=1),
                        dict(kernel_size=4, stride=2, padding=1),
                    ],
                },
            },
        },
    },
}
# 3 - 32 - 64 - 128 - 256
def get_architecture(config):
    data_cnf = config["input_data"]
    pixel = data_cnf["pixel_representation"]["type"]
    dataset = data_cnf["dataset"]
    model = config.get("vqvae", config.get("vae", config.get("ae", config.get("vqvae_rec", config.get("model", None)))))


    arch = model["architecture"]
    n_layers = len(model["filters"]) + 1
    lshape = model["latent_shape"]
    ls = f'{lshape["height"]}x{lshape["width"]}'


    try:
        architecture = ARCHITECTURE_REGISTRY[arch][n_layers][pixel]      
        params = CONV_PARAMS[dataset][arch][ls][n_layers]
        return architecture, params["encoder"], params["decoder"]
    except KeyError:
        raise ValueError(f"Unknown configuration for dataset: {dataset}, "
                         f"pixel rep: {pixel}, "
                         f"architecture: {arch} "
                         f"and layer filters: { model['filters']}")