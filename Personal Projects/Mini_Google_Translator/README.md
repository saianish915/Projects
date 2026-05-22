# Mini Google Translator (English to Telugu)

## Motivation

Telugu speakers commonly code-switch between Telugu and English in
everyday conversation, embedding English words naturally into Telugu
sentences. As a native Telugu speaker, this pattern is deeply familiar:
phrases like "Adi really good idea" or "Nenu tomorrow school ki veltanu"
reflect how the language is actually spoken in daily life rather than
the formal pure-Telugu found in textbooks.

This project was built with that personal motivation. Rather than treating
Telugu as an abstract low-resource language problem, this translator was
designed to handle natural input that reflects how Telugu speakers actually
communicate, mixing English words into Telugu sentences naturally and
producing clean Telugu output.

---

## Hypothesis

A pretrained neural machine translation model trained on broad multilingual
data can be fine-tuned on a domain-specific English-Telugu parallel corpus
to produce meaningful Telugu translations. If the pretrained model has
already learned general linguistic structure and the fine-tuning data
provides sufficient Telugu-specific signal, the resulting model should
outperform the pretrained baseline on everyday conversational sentences
including those with mixed English-Telugu input.

---

## Background

Telugu is a Dravidian language spoken by approximately 82 million people,
primarily in the Indian states of Andhra Pradesh and Telangana. Despite
being one of the most widely spoken languages in India, Telugu is
significantly underrepresented in commercial translation systems compared
to European languages.

This project builds a lightweight English to Telugu translation system
by fine-tuning the Helsinki-NLP MarianMT model on a curated parallel
corpus of 155,798 English-Telugu sentence pairs. MarianMT is a family
of transformer-based sequence-to-sequence models trained by the Language
Technology Research Group at the University of Helsinki on the OPUS
parallel corpus collection.

The project demonstrates that meaningful neural machine translation for
low-resource language pairs can be achieved through transfer learning
from a pretrained multilingual model without training from scratch.

---

## Dataset

Source: English-Telugu parallel corpus
File format: Custom delimiter format (sentence pairs separated by ++++$++++)
Raw size: 155,798 sentence pairs
Train split: 90% (approximately 140,218 pairs)
Test split: 10% (approximately 15,580 pairs)

Sample sentence pairs:

| English | Telugu |
|---------|--------|
| His legs are long. | అతని కాళ్ళు పొడవుగా ఉన్నాయి. |
| I swim in the sea every day. | నేను ప్రతి రోజు సముద్రంలో ఈత కొడతాను. |
| Smoke filled the room. | పొగ గదిని నింపింది. |

### Data Cleaning

The raw file used a non-standard delimiter format where each line contained
an English sentence and its Telugu translation separated by ++++$++++.
Cleaning steps applied:

- Lines not containing the delimiter were skipped as malformed
- Both sides were stripped of whitespace after splitting
- Any residual delimiter characters were removed from the text
- Rows where either side was fewer than 2 characters were dropped

---

## Methodology

### Model Selection

Helsinki-NLP/opus-mt-en-dra was selected as the base pretrained model.
This model was trained on English to Dravidian language pairs from the
OPUS corpus, making it the most appropriate starting point for English
to Telugu translation as Telugu is a Dravidian language.

Using a pretrained model rather than training from scratch provides:

- Pretrained knowledge of English grammar and syntax
- General sequence-to-sequence translation capability
- Significantly reduced training time and data requirements
- A stronger baseline than a randomly initialized model

### Tokenization

MarianTokenizer was used with a maximum sequence length of 128 tokens.
Sequences longer than 128 tokens were truncated and shorter sequences
were padded to a fixed length. The language token >>te<< was prepended
to source sentences to signal Telugu as the target language.

### Fine-Tuning Setup

The pretrained model was fine-tuned using the Hugging Face Seq2SeqTrainer
with the following configuration:

| Parameter | Value |
|-----------|-------|
| Learning rate | 5e-5 |
| Batch size (train) | 8 |
| Batch size (eval) | 8 |
| Weight decay | 0.01 |
| Epochs | 3 |
| Predict with generate | True |
| Mixed precision (fp16) | Disabled |

DataCollatorForSeq2Seq was used to handle dynamic padding during
training, ensuring efficient batch processing without wasting compute
on excessive padding.

---

## Experiments

### Experiment 1: Pretrained Baseline

Before fine-tuning, the pretrained Helsinki-NLP/opus-mt-en-dra model
was tested directly on a sample sentence:

Input: >>te<< This is a book.
Output: [paste your pretrained baseline output here]

This established the translation quality before any fine-tuning on
the custom corpus, providing a comparison point for measuring the
impact of fine-tuning.

### Experiment 2: Data Parsing and Cleaning

The raw data file required a custom parser to handle the non-standard
++++$++++ delimiter format. Lines without the delimiter were skipped
rather than raising errors, making the parser robust to malformed entries.
After cleaning, the dataset was converted to the Hugging Face Dataset
format with the translation column structure expected by MarianMT.

### Experiment 3: Fine-Tuning on English-Telugu Corpus

The model was fine-tuned for 3 epochs on 140,218 training pairs with
evaluation every 500 steps. The fine-tuned model was saved locally
and reloaded for inference testing.

### Experiment 4: Translation Quality on Conversational Sentences

The fine-tuned model was tested on 8 conversational sentences reflecting
everyday Telugu-English usage patterns:

| English Input | Telugu Output |
|---------------|---------------|
| What is your name? | [paste output] |
| How are you? | [paste output] |
| I am learning machine translation. | [paste output] |
| This is a book. | [paste output] |
| She likes mangoes. | [paste output] |
| We are going to school. | [paste output] |
| I don't know. | [paste output] |

---

## Results

Model: Helsinki-NLP/opus-mt-en-dra fine-tuned on English-Telugu corpus
Training pairs: 140,218
Evaluation pairs: 15,580
Training epochs: 3
Final model saved to: en_te_translator_model/

[Paste your training loss values here after running]

---

## Conclusion and Takeaways

1. Personal motivation produces better research questions

Building a translator for a language you speak natively changes how
you evaluate the output. A native Telugu speaker immediately recognizes
whether a translation sounds natural, formal, or wrong in ways that
an automated BLEU score cannot capture. The most meaningful evaluation
of this model is whether it produces translations that sound like
something a Telugu speaker would actually say.

2. Code-switching is the real use case

Formal Telugu-only input is not how most Telugu speakers communicate.
The practical use case for this translator is handling code-switched
input where English words appear naturally within Telugu sentences.
The current model handles English-only input but future work would
train explicitly on code-switched data to handle mixed-language input
more robustly.

3. Transfer learning makes low-resource NMT practical

Training a neural machine translation model from scratch for a
low-resource language pair like English-Telugu would require hundreds
of millions of sentence pairs and significant compute. Fine-tuning a
pretrained MarianMT model on 155,798 pairs achieves meaningful
translation quality in a fraction of the time and cost, demonstrating
the power of transfer learning for NLP tasks.

4. Language family matters for model selection

Selecting opus-mt-en-dra specifically because Telugu is a Dravidian
language was a deliberate and important modeling decision. A model
pretrained on European language pairs would have little relevant
prior knowledge for Telugu's agglutinative morphology and
subject-object-verb word order. Domain-appropriate pretraining is
as important as the fine-tuning data itself.

5. Sequence length is a modeling constraint

Capping sequences at 128 tokens truncates longer sentences which
may affect translation quality for complex multi-clause sentences.
Telugu sentences tend to be longer than their English equivalents
due to agglutination, meaning some semantic content may be lost
during truncation. Increasing max_length to 256 would reduce
truncation at the cost of increased memory usage.

6. Future Work

The most impactful improvements would be:

- Training on code-switched data that mixes English and Telugu
  to reflect how the language is actually spoken
- Adding Romanized Telugu input support so users can type
  Telugu phonetically without a Telugu keyboard
- Computing BLEU score on the test set for objective quality measurement
- Pushing the fine-tuned model to the Hugging Face Hub for
  public access and community evaluation
- Increasing batch size and enabling fp16 mixed precision to
  accelerate training

7. Limitations

- No quantitative evaluation metric such as BLEU score was computed
  to measure translation quality objectively
- Batch size of 8 is small and training would benefit from larger
  batches with more GPU memory
- fp16 mixed precision training was disabled which significantly
  slows training on GPU
- The model was not pushed to the Hugging Face Hub for public access
- Translation quality for code-switched Telugu-English input was not
  evaluated
- 3 epochs may be insufficient for full convergence on this dataset size

---

## Tech Stack

Python              Data processing and model training
pandas              Dataset loading and cleaning
transformers        MarianMT model, tokenizer, Seq2SeqTrainer
datasets            Hugging Face Dataset format and train test split
sentencepiece       Subword tokenization for MarianMT
sacrebleu           BLEU score evaluation
torch               PyTorch backend for model training

---

## How To Run

1. Place english_telugu_data.txt in your working directory

2. Install dependencies:
   pip install transformers datasets sentencepiece sacrebleu pandas

3. Run the notebook cells in order:
   - Data parsing and cleaning
   - Dataset preparation and tokenization
   - Fine-tuning with Seq2SeqTrainer
   - Inference and translation testing

4. Output files generated:
   cleaned_en_te_final.csv
   en_te_translator_model/

Note: A GPU runtime is strongly recommended. Training on CPU will
be extremely slow for 140,000 sentence pairs over 3 epochs.
