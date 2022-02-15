#NFETC-CLHLS

The code for DASFAA(2022) paper `A Three-Stage Curriculum Learning Framework with Hierarchical Label Smoothing for Fine-Grained Entity Typing`

##Request

* numpy==1.15.4
* tensorflow==1.14.0
* pandas==0.23.4
* gensim==3.8.3
* scikit_learn==0.23.1

##Data

For generate `wikim`(wiki with the improved hierarchy) raw data, obtained from [NFETC](https://github.com/billy-inn/NFETC).  
For generate `OntoNotes` and `BBN` raw data, download data in `data` directory using `download.sh`.
Run command:
`sed -i "1i 2196017 300" data/glove.840B.300d.txt` to insert a line "2196017 300" to the head of `data/glove.840.300d.txt`.
You also can be obtained from [NFETC-CLSC](https://github.com/herbertchen1/NFETC-CLSC)

##Train and Evaluation

Run `python eval.py -m <model_name> -d <data_name> -r <runs> -g <gpu>` and the scores for each run and the average scores are recorded in one log file stored in folder `log`.

* `<data_name>` choices: `wikim, ontonotes, bbn`
* `<model_name>` choices: `wikim, ontonotes, bbn`
* `<runs>`: the number of repetitions of the experiment.

##Acknowledge

Code is based on previous work: [NFETC-AR](https://www.ijcai.org/proceedings/2020/0527.pdf), [NFETC-CLSC](https://github.com/herbertchen1/NFETC-CLSC) and [NFETC](https://github.com/billy-inn/NFETC), many thanks to them.  
For more implementation details, please read the source code.
