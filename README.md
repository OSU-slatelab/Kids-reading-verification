# Kids-reading-verification

Project page for the Kids' reading verification project. For a given word in a reading passage, and the kids’ speech:
* Detect in real time whether a mispronunciation was made.
* Categorize the mispronunciation.

## Pretraining
An attention-based ASR pretraining with teacher forcing is used. To run:
<code>run_pt.sh</code>

## Finetuning
Co-train in-domain ASR with detection. To run (5-fold cross-validation):
<code>run_slu.sh</code>
