# Answers (17 of 17 questions answered)

Pipeline: full pipeline (rewrite, router, rulebook search and/or SQL), model gemini-3.1-flash-lite, search setup hybrid, judge gemini-3.5-flash-lite

| all | n | faithfulness | answer_relevancy | factual_correctness |
|---|---|---|---|---|
| all | 17 | 0.89 | 0.89 | 0.56 |

By question type:

| type | n | faithfulness | answer_relevancy | factual_correctness |
|---|---|---|---|---|
| single_hop_specific_query_synthesizer | 8 | 0.94 | 0.89 | 0.56 |
| multi_hop_specific_query_synthesizer | 3 | 0.63 | 0.92 | 0.50 |
| multi_hop_abstract_query_synthesizer | 3 | 1.00 | 0.90 | 0.70 |
| manual_teacher_question | 3 | 0.93 | 0.85 | 0.48 |

Routes chosen: rules 15, both 2
