"""Benchmark categories and original extraction/judge protocols."""

import re


BENCHMARKS = {'structcot': {'size': 5600,
               'test_file': 'structcot_test.json',
               'categories': [('Strategic Game and Planning', 'Strategy Planning'),
                              ('Spatial Perception and Embodied Planning', 'Spatial Planning'),
                              ('Logical and Abstract Reasoning', 'Logic'),
                              ('Mathematical and Algorithmic Reasoning', 'Math'),
                              ('Natural Science and Domain-Specific Reasoning', 'Science'),
                              ('Visual Search', 'Visual Search'),
                              ('Image Restoration and Jigsaw Reasoning', 'Jigsaw Restoration')]},
 'zebracot': {'size': 4000,
              'test_file': 'zebracot_test.json',
              'categories': [('2D Visual Reasoning', '2D'),
                             ('3D Visual Reasoning', '3D'),
                             ('Scientific Reasoning', 'Science'),
                             ('Visual Logic & Strategic Games', 'Strategy')]}}


EXTRACTION_PROMPTS = {
    "structcot": """You are an expert answer extractor. Your strictly single task is to extract the final conclusion from the provided response to a given question. You must NOT judge whether the response is correct or wrong; only extract what the response claims is the answer.

Extraction Rules:
1. If the response contains an explicit final answer, extract the final answer content exactly.
2. If the final answer is multi-line code, HTML, a list, a table, or structured text, keep the full content.
3. If the response selects a specific multiple-choice option letter, output only the single uppercase letter.
4. If no usable answer exists, output "Z".

Output Format:
Directly output the extracted answer ONLY.

Question: {question}
Answer: {response}
""",
    "zebracot": """You are an expert answer extractor. Your strictly single task is to extract the final conclusion from the provided response to a given question. You must NOT judge whether the response is correct or wrong; only extract what the response claims is the answer.

Extraction Rules:
1. [Multiple-Choice Tasks]:
   - If the response selects a specific option letter, output ONLY the single uppercase letter.
   - If the response gives both an option letter and its content, return only the letter.

2. [Open-Ended / Value Tasks]:
   - Extract the final requested answer only.
   - Do not add explanation.
   - Keep the answer concise and faithful.

3. [No Valid Answer]:
   - If no usable answer exists, output "Z".

Output Format:
Directly output the extracted answer ONLY.

Question: {question}
Answer: {response}""",
}


JUDGE_PROMPTS = {
    "structcot": """You are an AI assistant who will help judge whether a model answer is consistent with the standard answer for a given question.

Your task:
Given a question, a standard answer, and a model answer, determine whether the model answer is consistent with the standard answer.

Evaluation Criteria:

1. Multiple-choice questions
If the question contains explicit options such as A/B/C/D:
- The predicted answer must match the correct option.
- Acceptable forms include the option letter, the option letter with description, or the option content itself if it clearly matches the standard answer.

2. Numerical or counting questions
- The predicted numeric value must match the ground truth numeric value.
- Equivalent formats are acceptable, such as 36% and 36 percent.
- If the numeric value differs, mark Incorrect.

3. Open-ended answers
- Mark Correct if the prediction conveys the same answer as the standard answer.
- Minor wording differences are acceptable.

4. Procedural or step-based tasks
- Mark Correct if the predicted sequence or final result is consistent with the standard answer.
- Mark Incorrect if key actions/results contradict the standard answer.

5. Strict incorrect cases
Mark Incorrect if:
- the key entity/object differs;
- the selected option differs;
- the numeric value differs;
- the predicted answer is empty, missing, or only says it cannot answer;
- the predicted answer contradicts the standard answer.

Output format:
Output ONLY one label:
Correct
Incorrect
""",
    "zebracot": """You are an AI assistant who will help judge whether a model answer is consistent with the standard answer for a given question.

Your task:
Given a question, a standard answer, and a model answer, determine whether the model answer is consistent with the standard answer.

Evaluation Criteria for Answer Matching

1. Multiple-choice questions
If the question contains explicit options (e.g., A/B/C/D):
- The predicted answer must match the correct option.
- Acceptable forms include:
  - The option letter only (e.g., A)
  - The option letter with description (e.g., A. Qc2)
  - A sentence that clearly identifies the correct option (e.g., The answer is A.)
  - The option content itself, if it clearly matches the correct option content

2. True/False or option-equivalent questions
If the answer is expected to match one of several discrete options:
- The predicted answer must correspond to the same option or the same option content as the ground truth.
- Minor wording differences are acceptable only if the selected option/content is the same.

3. Numerical or counting questions
For problems where the numeric value is a key part of the answer (e.g., counting, calculation, measurements):
- The predicted numeric value must match the ground truth value.
- Equivalent formats are acceptable:
  - 30 days vs 30 Days
  - 36% vs 36 percent
- If the numeric value differs, mark it Incorrect even if the reasoning is similar.

4. Procedural or step-based tasks
For tasks involving sequences of actions or steps (e.g., robot planning, navigation, manipulation):
- The predicted sequence should achieve the same goal and include the required actions in the ground truth.
- Additional intermediate steps are allowed if they do NOT introduce incorrect operations and do NOT change the intended outcome.
- Mark Incorrect if:
  - required actions are missing
  - key actions are replaced with different actions
  - extra actions cause additional state changes beyond the ground truth requirements

5. Open-ended descriptive answers
For other question types:
- Mark Correct if the prediction conveys the same meaning and answers the question correctly, even if the wording differs.
- Minor paraphrasing is acceptable.

6. Strict incorrect conditions
Mark Incorrect if any of the following occurs:
- The key entity/object differs from the ground truth
- The numeric value differs
- The selected option differs
- The described actions/results contradict the ground truth

Output format:
- Output ONLY one label:
  - Correct
  - Incorrect
""",
}


def normalize_final_answer(answer):
    answer = answer.strip()
    if not answer:
        return None
    match = re.fullmatch(r"([A-Z])[\.\)、:：]\s*", answer)
    return match.group(1) if match else answer


def extract_by_rules(benchmark, text):
    if not text.strip():
        return None
    if benchmark == "zebracot":
        return None
    for pattern in (
        r"<answer>\s*([\s\S]*?)\s*</answer>",
        r"<\\answer>\s*([\s\S]*?)\s*</\\answer>",
        r"Final\s+Answer\s*[:：]\s*([\s\S]*)",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_final_answer(match.group(1))
    for pattern in (
        r"(?:\*\*)?\bAnswer\s*[:：](?:\*\*)?\s*([\s\S]*)",
        r"The\s+answer\s+is\s*[:：]?\s*([\s\S]*)",
    ):
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if matches:
            return normalize_final_answer(matches[-1].group(1))
    return None
