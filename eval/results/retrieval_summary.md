# Retrieval (17 questions, top 5 chunks)

| setup | n | hit_rate | mrr | context_recall | context_precision |
|---|---|---|---|---|---|
| vector | 17 | 0.94 | 0.83 | 0.87 | 0.79 |
| hybrid | 17 | 0.94 | 0.85 | 0.87 | 0.82 |
| full | 17 | 0.94 | 0.85 | 0.81 | 0.86 |

By question type, hybrid setup (the default):

| type | n | hit_rate | mrr | context_recall | context_precision |
|---|---|---|---|---|---|
| single_hop_specific_query_synthesizer | 8 | 0.88 | 0.69 | 0.88 | 0.69 |
| multi_hop_specific_query_synthesizer | 3 | 1.00 | 1.00 | 0.83 | 0.90 |
| multi_hop_abstract_query_synthesizer | 3 | 1.00 | 1.00 | 1.00 | 0.90 |
| manual_teacher_question | 3 | 1.00 | 1.00 | 0.78 | 1.00 |
