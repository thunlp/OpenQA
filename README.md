# Open-QA

The source codes for paper "Denoising Distantly Supervised Open-Domain Question Answering", which is modified based on the [code](https://github.com/facebookresearch/DrQA) of paper " Reading Wikipedia to Answer Open-Domain Questions."


Evaluation Results
==========

<table border=0 cellpadding=0 cellspacing=0 width=705 style='border-collapse:
 collapse;table-layout:fixed;width:527pt'>
 <col width=183 style='mso-width-source:userset;mso-width-alt:5845;width:137pt'>
 <col width=87 span=6 style='width:65pt'>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl66 width=183 style='height:16.0pt;width:137pt'>Dataset</td>
  <td colspan=2 class=xl66 width=174 style='border-left:none;width:130pt'>Quasar-T</td>
  <td colspan=2 class=xl66 width=174 style='border-left:none;width:130pt'>SearchQA</td>
  <td colspan=2 class=xl66 width=174 style='border-left:none;width:130pt'>TrivialQA</td>
 <td colspan=2 class=xl66 width=174 style='border-left:none;width:130pt'>SQuAD</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl66 style='height:16.0pt;border-top:none'>Models</td>
  <td class=xl66 style='border-top:none;border-left:none'>EM</td>
  <td class=xl66 style='border-top:none;border-left:none'>F1</td>
  <td class=xl66 style='border-top:none;border-left:none'>EM</td>
  <td class=xl66 style='border-top:none;border-left:none'>F1</td>
  <td class=xl66 style='border-top:none;border-left:none'>EM</td>
  <td class=xl66 style='border-top:none;border-left:none'>F1</td>
  <td class=xl66 style='border-top:none;border-left:none'>EM</td>
  <td class=xl66 style='border-top:none;border-left:none'>F1</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl67 style='height:16.0pt;border-top:none'>GA<font
  class="font7"> (Dhingra et al., 2017)</font></td>
  <td class=xl67 style='border-top:none'>26.4</td>
  <td class=xl69 style='border-top:none'>26.4</td>
  <td class=xl67 style='border-top:none;border-left:none'>-</td>
  <td class=xl69 style='border-top:none'>-</td>
  <td class=xl68 style='border-top:none'>-</td>
  <td class=xl69 style='border-top:none'>-</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl70 style='height:16.0pt'>BiDAF <font class="font7">(Seo
  et al., 2017)</font></td>
  <td class=xl70>25.9</td>
  <td class=xl71>28.5</td>
  <td class=xl70 style='border-left:none'>28.6</td>
  <td class=xl71>34.6</td>
  <td class=xl65>-</td>
  <td class=xl71>-</td>
  <td class=xl65>-</td>
  <td class=xl71>-</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl70 style='height:16.0pt'>AQA <font class="font7">(Buck
  et al., 2017)</font></td>
  <td class=xl70>-</td>
  <td class=xl71>-</td>
  <td class=xl70 style='border-left:none'>40.5</td>
  <td class=xl71>47.4</td>
  <td class=xl65>-</td>
  <td class=xl71>-</td>
  <td class=xl65>-</td>
  <td class=xl71>-</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl70 style='height:16.0pt'>R^3<font class="font7"> (Wang
  et al., 2018a)</font></td>
  <td class=xl70>35.3</td>
  <td class=xl71>41.7</td>
  <td class=xl70 style='border-left:none'>49</td>
  <td class=xl71>55.3</td>
  <td class=xl65>47.3</td>
  <td class=xl71>53.7</td>
 
  <td class=xl65>37.5</td>
  <td class=xl71>29.1</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl72 style='height:16.0pt'>Our Model</td>
  <td class=xl75>42.2</td>
  <td class=xl74>49.3</td>
  <td class=xl75 style='border-left:none'>58.8</td>
  <td class=xl74>64.5</td>
  <td class=xl73>48.7</td>
  <td class=xl74>56.3</td>
 
  <td class=xl73>36.6</td>
  <td class=xl74>28.7</td>
 </tr>
</table>

Data
==========

We provide Quasar-T, SearchQA and TrivialQA  dataset we used for the task in data/ directory. We preprocess the original data to make it satisfy the input format of our codes, and can be download at [here](https://thunlp.oss-cn-qingdao.aliyuncs.com/OpenQA_data.tar.gz).

To run our code, the dataset should be put in the folder data/ using the following format:

datasets/
+ train.txt, dev.txt, test.txt:  format for each line: \{"question": quetion, "answers":[answer1, answer2, ...]\}.

+ train.json, dev.json, test.json: format [\{"question": question, "document":document1\},\{"question": question, "document":document2\}, ...]. 

embeddings/
+ glove.840B.300d.txt: word vectors obtained from [here](https://nlp.stanford.edu/projects/glove/).

corenlp/
+ all jar files from Stanford Corenlp.

Codes
==========

The source codes of our models are put in the folders src/.

Train and Test
==========
For training and test, you need to:

1. Pre-train the paragraph reader:	python main --batch-size 256 --model-name quasart_reader --num-epochs 10 --dataset quasart --mode reader

2. Pre-train the paragraph selector: 	python main --batch-size 64 --model-name quasart_selector --num-epochs 10 --dataset quasart --mode selector --pretrained models/quasart_reader

3. Train the whole model: python main --batch-size 32 --model-name quasart_all --num-epochs 10 --dataset quasart --mode all --pretrained models/quasart_selector



Cite
==========

If you use the code, please cite the following paper:

1. Yankai Lin, Haozhe Ji, Zhiyuan Liu, and Maosong Sun. 2018. Denoising Distantly Supervised Open-Domain Question Answering. In Proceedings of ACL. pages 1736--1745. [[pdf]](http://www.thunlp.org/~lyk/publications/acl2018_denoising.pdf)

Reference
=========

1. Bhuwan Dhingra, Hanxiao Liu, Zhilin Yang, William Cohen,  and Ruslan Salakhutdinov.  2017.  Gated-attention readers for text comprehension. In Proceedings of ACL. pages 1832--1846.

2. Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, and Hannaneh Hajishirzi. 2017.  Bidirectional attention flow for machine comprehension. In Proceedings of ICLR.

3. Christian Buck,  Jannis Bulian, Massimiliano Ciaramita, Andrea Gesmundo, Neil Houlsby, Wojciech Gajewski, and Wei Wang. 2017. Ask the right questions: Active question reformulation with reinforcement learning. arXiv preprint arXiv:1705.07830.

4. Shuohang Wang, Mo Yu, Xiaoxiao Guo, Zhiguo Wang,Tim Klinger, Wei Zhang, Shiyu Chang, Gerald Tesauro, Bowen Zhou, and Jing Jiang. 2018.  R3: Reinforced ranker-reader for open-domain question answering. In Proceedings of AAAI. pages 5981--5988.
