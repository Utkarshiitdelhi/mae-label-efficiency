| Part | Parameters | Compute per image | Share of pretraining compute |
| --- | --- | --- | --- |
| Encoder (16 visible tokens) | 5.35 M | 87 M MACs | 61% |
| Decoder (64 tokens) | 0.82 M | 55 M MACs | 39% |
| Encoder at fine-tune (64 tokens) | 5.35 M | 359 M MACs | n/a |

Decoder / encoder parameters: 15.4%. Pretraining encoder cost vs running it on all 64 tokens: 4.1x cheaper.
