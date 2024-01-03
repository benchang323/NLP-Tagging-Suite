# Hello

## Create Models

### HMM

#### ic_hmm

`python3 tag.py ../data/icdev -t ../data/icsup --save_path ic_hmm.pkl`

#### ic_hmm_raw

`python3 tag.py ../data/icdev -t ../data/{icsup,icraw} --save_path ic_hmm_raw.pkl`

#### en_hmm

`python3 tag.py ../data/endev -l ../data/words-50.txt -t ../data/ensup --save_path en_hmm.pkl`

#### en_hmm_raw

`python3 tag.py ../data/endev -l ../data/words-50.txt -t ../data/{ensup,enraw} --save_path en_hmm_raw.pkl`

#### en_hmm_awesome

(here awesome should have all mothods)

`python3 tag.py ../data/endev -l ../data/words-50.txt -t ../data/ensup --save_path en_hmm_awesome.pkl --awesome`

### CRF

#### ic_crf

`python3 tag.py ../data/icdev -t ../data/icsup --crf --save_path ic_crf.pkl`

#### en_crf

`python3 tag.py ../data/endev -l ../data/words-50.txt -t ../data/ensup --crf --save_path en_crf.pkl`

#### en_crf_raw

`python3 tag.py ../data/endev -l ../data/words-50.txt -t ../data/{ensup,enraw} --crf --save_path en_crf_raw.pkl`

#### en_crf_birnn

`python3 tag.py ../data/endev -l ../data/words-50.txt -t ../data/ensup --crf --birnn --save_path en_crf_birnn.pkl`

## Abalation Study

### Base Model

`python3 tag.py ../data/endev -l ../data/words-50.txt -t ../data/ensup --save_path base.pkl`

### Improved Model (with all lexicon methods)

`python3 tag.py ../data/endev -l ../data/words-50.txt -t ../data/ensup --awesome --save_path all.pkl`

### Improved Model (with additional log_counts in lexicon)

make sure topass true to only log_counts in lexicon

`python3 tag.py ../data/endev -l ../data/words-50.txt -t ../data/ensup --awesome --save_path base_log_counts.pkl`

### Improved Model (with additional affixes in lexicon)

`python3 tag.py ../data/endev -l ../data/words-50.txt -t ../data/ensup --awesome --save_path base_affixes.pkl`

### Improved Model (with additional word counts in lexicon)

`python3 tag.py ../data/endev -l ../data/words-50.txt -t ../data/ensup --awesome --save_path base_word_couts.pkl`