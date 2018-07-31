# Open-QA

The source codes for paper "Denoising Distantly Supervised Open-Domain Question Answering".


Evaluation Results
==========

<table border=0 cellpadding=0 cellspacing=0 width=609 style='border-collapse:
 collapse;table-layout:fixed;width:455pt'>
 <col width=87 span=7 style='width:65pt'>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl66 width=87 style='height:16.0pt;width:65pt'>Dataset</td>
  <td colspan=2 class=xl66 width=174 style='border-left:none;width:130pt'>Quasar-T</td>
  <td colspan=2 class=xl66 width=174 style='border-left:none;width:130pt'>SearchQA</td>
  <td colspan=2 class=xl66 width=174 style='border-left:none;width:130pt'>TrivialQA</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl66 style='height:16.0pt;border-top:none'>Models</td>
  <td class=xl66 style='border-top:none;border-left:none'>EM</td>
  <td class=xl66 style='border-top:none;border-left:none'>F1</td>
  <td class=xl66 style='border-top:none;border-left:none'>EM</td>
  <td class=xl66 style='border-top:none;border-left:none'>F1</td>
  <td class=xl66 style='border-top:none;border-left:none'>EM</td>
  <td class=xl66 style='border-top:none;border-left:none'>F1</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl67 style='height:16.0pt;border-top:none'>GA</td>
  <td class=xl67 style='border-top:none'>26.4</td>
  <td class=xl69 style='border-top:none'>26.4</td>
  <td class=xl67 style='border-top:none;border-left:none'>-</td>
  <td class=xl69 style='border-top:none'>-</td>
  <td class=xl68 style='border-top:none'>-</td>
  <td class=xl69 style='border-top:none'>-</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl70 style='height:16.0pt'>BiDAF</td>
  <td class=xl70>25.9</td>
  <td class=xl71>28.5</td>
  <td class=xl70 style='border-left:none'>28.6</td>
  <td class=xl71>34.6</td>
  <td class=xl65>-</td>
  <td class=xl71>-</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl70 style='height:16.0pt'>AQA</td>
  <td class=xl70>-</td>
  <td class=xl71>-</td>
  <td class=xl70 style='border-left:none'>40.5</td>
  <td class=xl71>47.4</td>
  <td class=xl65>-</td>
  <td class=xl71>　</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl70 style='height:16.0pt'>R^3</td>
  <td class=xl70>35.3</td>
  <td class=xl71>41.7</td>
  <td class=xl70 style='border-left:none'>49</td>
  <td class=xl71>55.3</td>
  <td class=xl65>47.3</td>
  <td class=xl71>53.7</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl72 style='height:16.0pt'>Our Model</td>
  <td class=xl75>42.2</td>
  <td class=xl74>49.3</td>
  <td class=xl75 style='border-left:none'>58.8</td>
  <td class=xl74>64.5</td>
  <td class=xl73>48.7</td>
  <td class=xl74>56.3</td>
 </tr>
 <tr height=126 style='height:96.0pt;mso-xlrowspan:6'>
  <td height=126 colspan=7 style='height:96.0pt;mso-ignore:colspan'></td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 style='height:16.0pt'></td>
  <td class=xl65></td>
  <td colspan=5 style='mso-ignore:colspan'></td>
 </tr>
</table>

Data
==========


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

[Lin et al., 2018] Yankai Lin, Haozhe Ji, Zhiyuan Liu, Maosong Sun. Denoising Distantly Supervised Open-Domain Question Answering. The 56th Annual Meeting of the Association for Computational Linguistics (ACL 2018). [[pdf]](http://www.thunlp.org/~lyk/publications/acl2018_denoising.pdf)