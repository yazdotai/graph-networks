# Graph Neural Networks and Recommendations
A list of interesting graph neural networks (GNN) material with a primary interest in recommendations and [tensorflow](https://github.com/tensorflow/tensorflow) that is continually updated and refined

  - [TensorFlow Implementations](#tensorflow-implementations)
  - [Articles](#articles)
  - [Videos](#videos)
  - [Public Datasets](#public-datasets)
  - [Recommendation Algorithms](#recommendation-algorithms)
  - [Research Papers](#research-papers)
    - [Relational Representation Learning](#relational-representation-learning)
    - [Survey papers](#survey-papers)
    - [Models](#models)
    - [Applications](#applications)


![graph neural networks](https://user-images.githubusercontent.com/130832/52767777-6354c800-3023-11e9-9032-3a5a89190996.png)

## TensorFlow Implementations

- **[Graph Nets in TensorFlow by DeepMind](https://github.com/deepmind/graph_nets)**

- **[Colab Notebook For Graph Nets and Item Connections / Recommendations](https://colab.research.google.com/github/deepmind/graph_nets/blob/master/graph_nets/demos/sort.ipynb)**

- **[Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://github.com/mdeff/cnn_graph)**

- **[Semi-Supervised Classification with Graph Convolutional Networks](https://github.com/tkipf/gcn)**

- **[GraphSAGE](https://github.com/williamleif/GraphSAGE)**

- **[Large-Scale Learnable Graph Convolutional Networks](https://github.com/divelab/lgcn/)**

- **[RippleNet](https://github.com/hwwang55/RippleNet)**

- **[MKR (multi-task learning for knowledge graph enhanced recommendation)](https://github.com/hwwang55/MKR)**

- **[DeepRec](https://github.com/cheungdaven/DeepRec)**

- **[OpenRec](https://github.com/ylongqi/openrec)**

- **[Graph Attention Networks](https://github.com/PetarV-/GAT)**

- **[Variational Graph Auto-Encoder](https://github.com/limaosen0/Variational-Graph-Auto-Encoders)**

- **[Adversarially Regularized Graph Autoencoder](https://github.com/Ruiqi-Hu/ARGA)**

- **[Deep Recursive Network Embedding with Regular Equivalence](https://github.com/tadpole/DRNE)**

- **[DeepWalk](https://github.com/triandicAnt/GraphEmbeddingRecommendationSystem)**

- **[GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Model](https://github.com/snap-stanford/GraphRNN)**

- **[Diffusion Convolutional Recurrent Neural Network](https://github.com/liyaguang/DCRNN)**

- **[Spatio-Temporal Graph Convolutional Networks](https://github.com/VeritasYin/STGCN_IJCAI-18)**

## Articles

- **[A Gentle Introduction to Graph Neural Networks (Basics, DeepWalk, and GraphSage)](https://towardsdatascience.com/a-gentle-introduction-to-graph-neural-network-basics-deepwalk-and-graphsage-db5d540d50b3)**

- **[PinSage: A new graph convolutional neural network for web-scale recommender systems](https://medium.com/pinterest-engineering/pinsage-a-new-graph-convolutional-neural-network-for-web-scale-recommender-systems-88795a107f48)**

- **[Model-Based Machine Learning and Making Recommendations](http://mbmlbook.com/Recommender.html)**

- **[Machine Learning for Recommender systems from Recombee](https://medium.com/recombee-blog/machine-learning-for-recommender-systems-part-1-algorithms-evaluation-and-cold-start-6f696683d0ed)**

- **[How Does Spotify Know You So Well?](https://medium.com/s/story/spotifys-discover-weekly-how-machine-learning-finds-your-new-music-19a41ab76efe)**

## Videos

- **[Intro to Graph Convolutional Networks](https://www.youtube.com/watch?v=UAwrDY_Bcdc)**

- **[Graph Convolutional Networks for Node Classification](https://www.youtube.com/watch?v=LFSR27BaNIQ)**

- **[Jure Leskovec - Large-scale Graph Representation Learning](https://www.youtube.com/watch?v=oQL4E1gK3VU)**

- **[Attention in Neural Networks](https://www.youtube.com/watch?v=W2rWgXJBZhU)**

- **[Michael Bronstein - Geometric deep learning on graphs: going beyond Euclidean data](https://www.youtube.com/watch?v=b187J4ndZWY)**

- **[Yann LeCun - Graph Embedding, Content Understanding, and Self-Supervised Learning](https://www.youtube.com/watch?v=UGPT64wo7lU)**

- **[DeepWalk: Turning Graphs Into Features via Network Embeddings with Neo4j](https://www.youtube.com/watch?v=aZNtHJwfIVg)**

- **[What is a Random Walk?](https://www.youtube.com/watch?v=stgYW6M5o4k)**

## Public Datasets
- [Recommender Systems Datasets](https://cseweb.ucsd.edu/~jmcauley/datasets.html)
- [GroupLens](https://grouplens.org/)
  - [MovieLens](https://grouplens.org/datasets/movielens/)
  - [HetRec2011](https://grouplens.org/datasets/hetrec-2011/)
  - [WikiLens](https://grouplens.org/datasets/wikilens/)
  - [Book-Crossing](https://grouplens.org/datasets/book-crossing/)
  - [Jester](https://grouplens.org/datasets/jester/)
  - [EachMovie](https://grouplens.org/datasets/eachmovie/)
- [Amazon Product Data](http://jmcauley.ucsd.edu/data/amazon/)
  - Books, Electronics, Movies, etc.
- [SNAP Datasets](https://snap.stanford.edu/data/index.html)
- [#nowplaying Dataset](http://dbis-nowplaying.uibk.ac.at/)
- [Last.fm Datasets](http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/index.html)
- [Million Song Dataset](https://labrosa.ee.columbia.edu/millionsong/)
- [Frappe](http://baltrunas.info/research-menu/frappe)
- [Yahoo! Webscope Program](https://webscope.sandbox.yahoo.com/)
  - music ratings, movie ratings, etc.
- [Yelp Dataset Challenge](https://www.yelp.com/dataset/challenge)
- [MovieTweetings](https://github.com/sidooms/MovieTweetings)
- [Foursquare](https://archive.org/details/201309_foursquare_dataset_umn)
- [Epinions](http://jmcauley.ucsd.edu/data/epinions)
- [Google Local](http://jmcauley.ucsd.edu/data/googlelocal/)
  - location, phone number, time, rating, addres, GPS, etc.
- [CiteUlike-t](http://www.wanghao.in/CDL.htm)
- [LibimSeTi](http://www.occamslab.com/petricek/data/)
- [Scholarly Paper Recommendation Datasets](http://www.comp.nus.edu.sg/~sugiyama/SchPaperRecData.html)
- [Netflix Prize Data Set](http://academictorrents.com/details/9b13183dc4d60676b773c9e2cd6de5e5542cee9a)
- [FilmTrust,CiaoDVD](https://www.librec.net/datasets.html)
- [Chicago Entree](http://archive.ics.uci.edu/ml/datasets/Entree+Chicago+Recommendation+Data)
- [Douban](http://socialcomputing.asu.edu/datasets/Douban)
- [BibSonomy](https://www.kde.cs.uni-kassel.de/bibsonomy/dumps)
- [Delicious](http://www.dai-labor.de/en/competence_centers/irml/datasets/)
- [Foursquare](https://archive.org/details/201309_foursquare_dataset_umn)
- [MACLab LJ Datasets](http://mac.citi.sinica.edu.tw/LJ#.Ww_hbFOFNE5)
- Kaggle::Datasets
  - [Steam Video Games](https://www.kaggle.com/tamber/steam-video-games/data)
  - [Anime Recommendations Database](https://www.kaggle.com/CooperUnion/anime-recommendations-database)

**Movies Recommendation**:
* *MovieLens* - Movie Recommendation Data Sets http://www.grouplens.org/node/73
* *Yahoo!* - Movie, Music, and Images Ratings Data Sets http://webscope.sandbox.yahoo.com/catalog.php?datatype=r
* *Jester* - Movie Ratings Data Sets (Collaborative Filtering Dataset) http://www.ieor.berkeley.edu/~goldberg/jester-data/
* *Cornell University* - Movie-review data for use in sentiment-analysis experiments http://www.cs.cornell.edu/people/pabo/movie-review-data/

**Music Recommendation**:
* *Last.fm* - Music Recommendation Data Sets http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/index.html
* *Yahoo!* - Movie, Music, and Images Ratings Data Sets http://webscope.sandbox.yahoo.com/catalog.php?datatype=r
* *Audioscrobbler* - Music Recommendation Data Sets http://www-etud.iro.umontreal.ca/~bergstrj/audioscrobbler_data.html
* *Amazon* - Audio CD recommendations http://131.193.40.52/data/


**Books Recommendation**:
* *Institut für Informatik, Universität Freiburg* - Book Ratings Data Sets http://www.informatik.uni-freiburg.de/~cziegler/BX/


**Food Recommendation**:
* *Chicago Entree* - Food Ratings Data Sets http://archive.ics.uci.edu/ml/datasets/Entree+Chicago+Recommendation+Data


**Merchandise Recommendation**:
* *Amazon* - Product Recommendation Data Sets http://131.193.40.52/data/


**Healthcare Recommendation**:
* *Nursing Home* - Provider Ratings Data Set http://data.medicare.gov/dataset/Nursing-Home-Compare-Provider-Ratings/mufm-vy8d
* *Hospital Ratings* - Survey of Patients Hospital Experiences http://data.medicare.gov/dataset/Survey-of-Patients-Hospital-Experiences-HCAHPS-/rj76-22dk


**Dating Recommendation**:
* *www.libimseti.cz* - Dating website recommendation (collaborative filtering) http://www.occamslab.com/petricek/data/


**Scholarly Paper Recommendation**:
* *National University of Singapore* - Scholarly Paper Recommendation http://www.comp.nus.edu.sg/~sugiyama/SchPaperRecData.html



## Recommendation Algorithms

- Basic of Recommender Systems
  - [Wikipedia](https://en.wikipedia.org/wiki/Recommender_system)
- Nearest Neighbor Search
  - [Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
  - [sklearn.neighbors](http://scikit-learn.org/stable/modules/neighbors.html)
  - [Benchmarks of approximate nearest neighbor libraries](https://github.com/erikbern/ann-benchmarks)
- Classic Matrix Facotirzation
  - [Matrix Factorization: A Simple Tutorial and Implementation in Python](http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/)
  - [Matrix Factorization Techiques for Recommendaion Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
- Singular Value Decomposition (SVD)
  - [Wikipedia](https://en.wikipedia.org/wiki/Singular-value_decomposition)
- SVD++
  - [Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model](http://www.cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf)
- Content-based CF / Context-aware CF
  - there are so many ...
- Advanced Matrix Factorization
  - [Probabilistic Matrix Factorization](https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf)
  - [Fast Matrix Factorization for Online Recommendation with Implicit Feedback](https://dl.acm.org/citation.cfm?id=2911489)
  - [Collaborative Filtering for Implicit Feedback Datasets](http://ieeexplore.ieee.org/document/4781121/)
  - [Factorization Meets the Item Embedding: Regularizing Matrix Factorization with Item Co-occurrence](https://dl.acm.org/citation.cfm?id=2959182)
- Factorization Machine
  - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
  - [Field-aware Factorization Machines for CTR Prediction](https://dl.acm.org/citation.cfm?id=2959134)
- Sparse LInear Method (SLIM)
  - [SLIM: Sparse Linear Methods for Top-N Recommender Systems](http://glaros.dtc.umn.edu/gkhome/node/774)
  - [Global and Local SLIM](http://glaros.dtc.umn.edu/gkhome/node/1192)
- Learning to Rank
  - [Wikipedia](https://en.wikipedia.org/wiki/Learning_to_rank)
  - [BPR: Bayesian personalized ranking from implicit feedback](https://dl.acm.org/citation.cfm?id=1795167)
  - [WSABIE: Scaling Up To Large Vocabulary Image Annotation](http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf)
  - [Top-1 Feedback](http://proceedings.mlr.press/v38/chaudhuri15.pdf)
  - [k-order statistic loss](http://www.ee.columbia.edu/~ronw/pubs/recsys2013-kaos.pdf)
  - [VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback](https://dl.acm.org/citation.cfm?id=3015834)
  - [The LambdaLoss Framework for Ranking Metric Optimization](https://dl.acm.org/citation.cfm?id=3271784)
- Cold-start
  - [Deep content-based music recommendation](https://papers.nips.cc/paper/5004-deep-content-based-music-recommendation)
  - [DropoutNet: Addressing Cold Start in Recommender Systems](https://papers.nips.cc/paper/7081-dropoutnet-addressing-cold-start-in-recommender-systems)
- Network Embedding
  - [awesome-network-embedding](https://github.com/chihming/awesome-network-embedding)
  - [Item2vec](https://arxiv.org/abs/1603.04259)
  - [entity2rec](https://dl.acm.org/citation.cfm?id=3109889)
- Translation Embedding
  - [Translation-based Recommendation](https://dl.acm.org/citation.cfm?id=3109882)
- Deep Learning
  - [Deep Neural Networks for YouTube Recommendations](https://ai.google/research/pubs/pub45530)
  - [Deep Learning based Recommender System: A Survey and New Perspectives](https://arxiv.org/abs/1707.07435)
  - [Neural Collaborative Filtering](https://dl.acm.org/citation.cfm?id=3052569)
  - [Collaborative Deep Learning for Recommender Systems](http://www.wanghao.in/CDL.htm)
  - [Collaborative Denoising Auto-Encoders for Top-N Recommender Systems](https://dl.acm.org/citation.cfm?id=2835837)
  - [Collaborative recurrent autoencoder: recommend while learning to fill in the blanks](https://dl.acm.org/citation.cfm?id=3157143)
  - [TensorFlow Wide & Deep Learning](https://www.tensorflow.org/tutorials/wide_and_deep)
  - [Deep Neural Networks for YouTube Recommendations](https://research.google.com/pubs/pub45530.html)
  - [Collaborative Memory Network for Recommendation Systems](https://arxiv.org/abs/1804.10862)


## Research Papers

### Relational Representation Learning
-   [On the Complexity of Exploration in Goal-driven Navigation](https://r2learning.github.io/assets/papers/hippo-r2l-camera-ready.pdf). Maruan Al-Shedivat, Lisa Lee, Ruslan Salakhutdinov, Eric Xing
-   [Deep Graph Infomax](http://petar-v.com/dgi_nips18_camera.pdf). Petar Veličković, Liam Fedus, William Hamilton, Pietro Liò, Yoshua Bengio, Devon Hjelm
-   [Image-Level Attentional Context Modeling Using Nested-Graph Neural Networks](https://arxiv.org/pdf/1811.03830.pdf). Guillaume Jaume, Behzad Bozorgtabar, Hazim Kemal Ekenel, Jean-Philippe Thiran, Maria Gabrani
-   [Compositional Language Understanding with Text-based Relational Reasoning](https://r2learning.github.io/assets/papers/CameraReadySubmission%208.pdf). Koustuv Sinha, Shagun Sodhani, William L Hamilton, Joelle Pineau
-   [Learning Graph Representation via Formal Concept Analysis](http://www.ar.sanken.osaka-u.ac.jp/pub/yoneda/NIPSWS2018_cr1113.pdf). Yuka Yoneda, Mahito Sugiyama, Takashi Washio
-   [A Simple Baseline Algorithm for Graph Classification](https://arxiv.org/pdf/1810.09155.pdf). Nathan De Lara, Edouard Pineau
-   [Pitfalls of Graph Neural Network Evaluation](https://arxiv.org/pdf/1811.05868.pdf). Oleksandr Shchur, Maximilian Mumme, Aleksandar Bojchevski, Stephan Günnemann
-   [TNE: A Latent Model for Representation Learning on Networks](https://abdcelikkanat.github.io/projects/TNE/TNE_R2L2018.pdf). Abdulkadir Celikkanat, Fragkiskos Malliaros
-   Using Ternary Rewards to Reason over Knowledge Graphs with Deep Reinforcement Learning. Frederic Godin, Anjishnu Kumar, Arpit Mittal
-   [A Case for Object Compositionality in GANs](https://drive.google.com/open?id=1HtZqg5mFXm3xeM-8Q8O1DlAd-2apGaik). Sjoerd van Steenkiste, Karol Kurach, Sylvain Gelly
-   [Learning DPPs by Sampling Inferred Negatives](https://zelda.lids.mit.edu/wp-content/uploads/sites/17/2018/11/nips_workshop.pdf). Zelda Mariet, Mike Gartrell, Suvrit Sra
-   [LanczosNet: Multi-Scale Deep Graph Convolutional Networks](http://www.cs.toronto.edu/~rjliao/papers/NIPS_R2L_lanczos_net.pdf). Renjie Liao, Zhizhen Zhao, Raquel Urtasun, Richard Zemel
-   [Chess2vec: Learning Vector Representations for Chess](http://www.berkkapicioglu.com/wp-content/uploads/2018/11/chess2vec_nips_2018_short.pdf). Berk Kapicioglu, Ramiz Iqbal
-   [Sparse Logistic Regression Learns All Discrete Pairwise Graphical Models](http://wushanshan.github.io/files/GraphicalModel_workshop.pdf). Shanshan Wu, Sujay Sanghavi, Alex Dimakis
-   [Towards Sparse Hierarchical Graph Classifiers](http://petar-v.com/spcls_nips18_camera.pdf). Catalina Cangea, Petar Veličković, Nikola Jovanović, Thomas Kipf, Pietro Liò
-   [GRevnet: Improving Graph Neural Nets with Reversible Computation](https://drive.google.com/file/d/1UYsTSnyKjl6MAox9vwGtV77wB_3vMavR/view?usp=sharing). Aviral Kumar, Jimmy Ba, Jamie Kiros, Kevin Swersky
-   [Detecting the Coarse Geometry of Networks](https://www.mis.mpg.de/preprints/2018/preprint2018_97.pdf). Melanie Weber, Emil Saucan, Jürgen Jost
-   [Modeling Attention Flow on Graphs](https://xiaoranxu.com/files/attflow_short_Xu.pdf). Xiaoran Xu
-   [Learning Generative Models across Incomparable Spaces](https://www.bunne.ch/paper/Bunne_2018_NeurIPS_R2L.pdf). Charlotte Bunne, David Alvarez-Melis, Andreas Krause , Stefanie Jegelka **Best Paper Award**
-   [Hierarchical Bipartite Graph Convolution Networks](https://drive.google.com/file/d/1cJabrT7Y_HN2DTuIkJz7kWiAFIlz9OOt/view?usp=sharing). Marcel Nassar
-   [Non-local RoI for Cross-Object Perception](https://drive.google.com/open?id=1idZrhvIL8n2rGWyz2x3fLOByrqeSsRz5). Shou-Yao Tseng, Hwann-Tzong Chen, Shao-Heng Tai, Tyng-Luh Liu
-   [Node Attribute Prediction: An Evaluation of Within- versus Across-Network Tasks](http://stanford.edu/~jugander/papers/neurips18w-withinacross.pdf). Kristen M. Altenburger, Johan Ugander
-   [Implicit Maximum Likelihood Estimation](https://people.eecs.berkeley.edu/~ke.li/papers/imle_nips18_r2l.pdf). Ke Li, Jitendra Malik
-   [Variational learning across domains with triplet information](https://arxiv.org/pdf/1806.08672.pdf). Rita Kuznetsova
-   [Fast k-Nearest Neighbour Search via Prioritized DCI](https://people.eecs.berkeley.edu/~ke.li/papers/pdci_nips18_r2l.pdf). Ke Li, Jitendra Malik
-   [Deep Determinantal Point Processes](https://arxiv.org/pdf/1811.07245.pdf). Mike Gartrell, Elvis Dohmatob
-   [Higher-Order Graph Convolutional Layer](http://sami.haija.org/papers/high-order-gc-layer.pdf). Sami A Abu-El-Haija, Bryan Perozzi, Amol Kapoor, Nazanin Alipourfard, Hrayr Harutyunyan
-   [Convolutional Set Matching for Graph Similarity](http://yunshengb.com/wp-content/uploads/2018/11/Convolutional_Set_Matching_for_Graph_Similarity.pdf). Yunsheng Bai, Hao Ding, Yizhou Sun, Wei Wang
-   [Improving Generalization for Abstract Reasoning Tasks Using Disentangled Feature Representations](https://arxiv.org/pdf/1811.04784.pdf). Xander Steenbrugge, Tim Verbelen, Bart Dhoedt, Sam Leroux
-   [From Node Embedding to Graph Embedding: Scalable Global Graph Kernel via Random Features](https://r2learning.github.io/assets/papers/RGE_NIPS18_RRL_Workshop.pdf). Lingfei Wu, Ian En-Hsu Yen, Kun Xu, Liang Zhao, Yinglong Xia, Michael Witbrock
-   [A Neural Framework for Learning DAG to DAG Translation](http://www.ccs.neu.edu/home/clara/resources/neural-framework-learning.pdf). M. Clara De Paolis Kaluza, Saeed Amizadeh, Rose Yu
-   [Semi-supervised learning for clusterable graph embeddings with NMF](https://priyeshv.github.io/R2L_SSNMF.pdf). Priyesh Vijayan, Anasua Mitra, Srinivasan Parthasarathy, Balaraman Ravindran
-   [Lifted Inference for Faster Training in end-to-end neural-CRF models](http://www.cse.iitd.ac.in/~mausam/papers/nipswork18.pdf). Yatin Nandwani, Ankit Anand, Mausam , Parag Singla
-   [Link Prediction in Dynamic Graphs for Recommendation](https://arxiv.org/pdf/1811.07174.pdf). Samuel G. Fadel, Ricardo Torres
-   Curvature and Representation Learning: Identifying Embedding Spaces for Relational Data. Melanie Weber, Maximillian Nickel
-   [Multi-Task Graph Autoencoders](https://arxiv.org/pdf/1811.02798.pdf). Phi Vu Tran
-   [Personalized Neural Embeddings for Collaborative Filtering with Text](https://r2learning.github.io/assets/papers/CameraReadySubmission%202.pdf). Guangneng Hu, Yu Zhang
-   [Symbolic Relation Networks for Reinforcement Learning](https://r2learning.github.io/assets/papers/CameraReadySubmission%203.pdf). Dhaval D Adjodah, Tim Klinger, Josh Joseph
-   [Extending the Capacity of CVAE for Face Sythesis and Modeling](https://r2learning.github.io/assets/papers/CameraReadySubmission%2011.pdf). Shengju Qian, Wayne Wu, Yangxiaokang Liu, Beier Zhu, Fumin Shen
-   [SARN: Relational Reasoning through Sequential Attention](https://r2learning.github.io/assets/papers/CameraReadySubmission%2049.pdf). Jinwon An, Seongwon Lyu, Sungzoon Cho
-   [Pairwise Relational Networks using Local Appearance Features for Face Recognition](https://arxiv.org/pdf/1811.06405.pdf). Bong-Nam Kang, YongHyun Kim, Daijin Kim
-   [Compositional Fairness Constraints for Graph Embeddings](https://r2learning.github.io/assets/papers/CameraReadySubmission%2035.pdf). Avishek Bose, William L Hamilton
-   [Improved Addressing in the Differentiable Neural Computer](http://people.idsia.ch/~csordas/nips2018.pdf). Róbert Csordás, Jürgen Schmidhuber
-   [Efficient Unsupervised Word Sense Induction, Disambiguation and Embedding](https://bigdata1.research.cs.dal.ca/behrouz/publication/nipsw2018/NIPSW2018_EfficientWordSenseDisambiguation.pdf). Behrouz Haji Soleimani, Habibeh Naderi, Stan Matwin
-   [Importance of object selection in Relational Reasoning tasks](https://r2learning.github.io/assets/papers/CameraReadySubmission%2019.pdf). Kshitij Dwivedi, Gemma Roig
-   [On Robust Learning of Ising Models](http://erikml.com/on_robust_learning_of_ising_models.pdf). Erik Lindgren, Vatsal Shah, Yanyao Shen, Alex Dimakis, Adam Klivans
-   [Feed-Forward Neural Networks need Inductive Bias to Learn Equality Relations](https://r2learning.github.io/assets/papers/CameraReadySubmission%2053.pdf). Tillman Weyde, Radha Manisha Kopparti
-   [Tensor Random Projection for Low Memory Dimension Reduction](https://r2learning.github.io/assets/papers/CameraReadySubmission%2041.pdf). Yang Guo, Yiming Sun, Madeleine Udell, Joel Tropp
-   [Leveraging Representation and Inference through Deep Relational Learning](https://r2learning.github.io/assets/papers/CameraReadySubmission%2042.pdf). Maria Leonor Pacheco, Ibrahim Dalal, Dan Goldwasser
-   [Learning Embeddings for Approximate Lifted Inference in MLNs](https://r2learning.github.io/assets/papers/CameraReadySubmission%2048.pdf). Maminur Islam, Somdeb Sarkhel, Deepak Venugopal

### Survey papers
- **Graph Neural Networks: A Review of Methods and Applications.**
*Jie Zhou, Ganqu Cui, Zhengyan Zhang, Cheng Yang, Zhiyuan Liu, Maosong Sun.* 2018. [paper](https://arxiv.org/pdf/1812.08434.pdf)

- **A Comprehensive Survey on Graph Neural Networks.**
*Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, Philip S. Yu.* 2019. [paper](https://arxiv.org/pdf/1901.00596.pdf)

- **Deep Learning on Graphs: A Survey.**
*Ziwei Zhang, Peng Cui, Wenwu Zhu.* 2018. [paper](https://arxiv.org/pdf/1812.04202.pdf)

- **Relational Inductive Biases, Deep Learning, and Graph Networks.**
*Battaglia, Peter W and Hamrick, Jessica B and Bapst, Victor and Sanchez-Gonzalez, Alvaro and Zambaldi, Vinicius and Malinowski, Mateusz and Tacchetti, Andrea and Raposo, David and Santoro, Adam and Faulkner, Ryan and others.* 2018. [paper](https://arxiv.org/pdf/1806.01261.pdf)

- **Geometric Deep Learning: Going beyond Euclidean data.**
*Bronstein, Michael M and Bruna, Joan and LeCun, Yann and Szlam, Arthur and Vandergheynst, Pierre.* IEEE SPM 2017. [paper](https://arxiv.org/pdf/1611.08097.pdf)

- **Computational Capabilities of Graph Neural Networks.**
*Scarselli, Franco and Gori, Marco and Tsoi, Ah Chung and Hagenbuchner, Markus and Monfardini, Gabriele.* IEEE TNN 2009. [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4703190)

- **Neural Message Passing for Quantum Chemistry.**
*Gilmer, Justin and Schoenholz, Samuel S and Riley, Patrick F and Vinyals, Oriol and Dahl, George E.* 2017. [paper](https://arxiv.org/pdf/1704.01212.pdf)

- **Non-local Neural Networks.**
*Wang, Xiaolong and Girshick, Ross and Gupta, Abhinav and He, Kaiming.* CVPR 2018. [paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf)

- **The Graph Neural Network Model.**
*Scarselli, Franco and Gori, Marco and Tsoi, Ah Chung and Hagenbuchner, Markus and Monfardini, Gabriele.* IEEE TNN 2009. [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4700287)


### Models

- **A new model for learning in graph domains.**
*Marco Gori, Gabriele Monfardini, Franco Scarselli.* IJCNN 2005. [paper](https://www.researchgate.net/profile/Franco_Scarselli/publication/4202380_A_new_model_for_earning_in_raph_domains/links/0c9605188cd580504f000000.pdf)

- **Graph Neural Networks for Ranking Web Pages.**
*Franco Scarselli, Sweah Liang Yong, Marco Gori, Markus Hagenbuchner, Ah Chung Tsoi, Marco Maggini.* WI 2005. [paper](https://www.researchgate.net/profile/Franco_Scarselli/publication/221158677_Graph_Neural_Networks_for_Ranking_Web_Pages/links/0c9605188cd5090ede000000/Graph-Neural-Networks-for-Ranking-Web-Pages.pdf)

- **Gated Graph Sequence Neural Networks.**
*Yujia Li, Daniel Tarlow, Marc Brockschmidt, Richard Zemel.* ICLR 2016. [paper](https://arxiv.org/pdf/1511.05493.pdf)

- **Geometric deep learning on graphs and manifolds using mixture model cnns.**
*Federico Monti, Davide Boscaini, Jonathan Masci, Emanuele Rodolà, Jan Svoboda, Michael M. Bronstein.* CVPR 2017. [paper](https://arxiv.org/pdf/1611.08402.pdf)

- **Spectral Networks and Locally Connected Networks on Graphs.**
*Joan Bruna, Wojciech Zaremba, Arthur Szlam, Yann LeCun.* ICLR 2014. [paper](https://arxiv.org/pdf/1312.6203.pdf)

- **Deep Convolutional Networks on Graph-Structured Data.**
*Mikael Henaff, Joan Bruna, Yann LeCun.* 2015. [paper](https://arxiv.org/pdf/1506.05163.pdf)

- **Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering.**
*Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst.* NIPS 2016. [paper](http://papers.nips.cc/paper/6081-convolutional-neural-networks-on-graphs-with-fast-localized-spectral-filtering.pdf)

- **Learning Convolutional Neural Networks for Graphs.**
*Mathias Niepert, Mohamed Ahmed, Konstantin Kutzkov.* ICML 2016. [paper](http://proceedings.mlr.press/v48/niepert16.pdf)

- **Semi-Supervised Classification with Graph Convolutional Networks.**
*Thomas N. Kipf, Max Welling.* ICLR 2017. [paper](https://arxiv.org/pdf/1609.02907.pdf)

- **Graph Attention Networks.**
*Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, Yoshua Bengio.* ICLR 2018. [paper](https://mila.quebec/wp-content/uploads/2018/07/d1ac95b60310f43bb5a0b8024522fbe08fb2a482.pdf)

- **Deep Sets.**
*Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Ruslan Salakhutdinov, Alexander Smola.* NIPS 2017. [paper](https://arxiv.org/pdf/1703.06114.pdf)

- **Graph Partition Neural Networks for Semi-Supervised Classification.**
*Renjie Liao, Marc Brockschmidt, Daniel Tarlow, Alexander L. Gaunt, Raquel Urtasun, Richard Zemel.* 2018. [paper](https://arxiv.org/pdf/1803.06272.pdf)

- **Covariant Compositional Networks For Learning Graphs.**
*Risi Kondor, Hy Truong Son, Horace Pan, Brandon Anderson, Shubhendu Trivedi.* 2018. [paper](https://arxiv.org/pdf/1801.02144.pdf)

- **Modeling Relational Data with Graph Convolutional Networks.**
*Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling.* ESWC 2018. [paper](https://arxiv.org/pdf/1703.06103.pdf)

- **Stochastic Training of Graph Convolutional Networks with Variance Reduction.**
*Jianfei Chen, Jun Zhu, Le Song.* ICML 2018. [paper](http://www.scipaper.net/uploadfile/2018/0716/20180716100330880.pdf)

- **Learning Steady-States of Iterative Algorithms over Graphs.**
*Hanjun Dai, Zornitsa Kozareva, Bo Dai, Alex Smola, Le Song.* ICML 2018. [paper](http://proceedings.mlr.press/v80/dai18a/dai18a.pdf)

- **Deriving Neural Architectures from Sequence and Graph Kernels.**
*Tao Lei, Wengong Jin, Regina Barzilay, Tommi Jaakkola.* ICML 2017. [paper](https://arxiv.org/pdf/1705.09037.pdf)

- **Adaptive Graph Convolutional Neural Networks.**
*Ruoyu Li, Sheng Wang, Feiyun Zhu, Junzhou Huang.* AAAI 2018. [paper](https://arxiv.org/pdf/1801.03226.pdf)

- **Graph-to-Sequence Learning using Gated Graph Neural Networks.**
*Daniel Beck, Gholamreza Haffari, Trevor Cohn.* ACL 2018. [paper](https://arxiv.org/pdf/1806.09835.pdf)

- **Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning.**
*Qimai Li, Zhichao Han, Xiao-Ming Wu.* AAAI 2018. [paper](https://arxiv.org/pdf/1801.07606.pdf)

- **Graphical-Based Learning Environments for Pattern Recognition.**
*Franco Scarselli, Ah Chung Tsoi, Marco Gori, Markus Hagenbuchner.* SSPR/SPR 2004. [paper](https://link.springer.com/content/pdf/10.1007%2F978-3-540-27868-9_4.pdf)

- **A Comparison between Recursive Neural Networks and Graph Neural Networks.**
*Vincenzo Di Massa, Gabriele Monfardini, Lorenzo Sarti, Franco Scarselli, Marco Maggini, Marco Gori.* IJCNN 2006. [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1716174)

- **Graph Neural Networks for Object Localization.**
*Gabriele Monfardini, Vincenzo Di Massa, Franco Scarselli, Marco Gori.* ECAI 2006. [paper](http://ebooks.iospress.nl/volumearticle/2775)

- **Knowledge-Guided Recurrent Neural Network Learning for Task-Oriented Action Prediction.**
*Liang Lin, Lili Huang, Tianshui Chen, Yukang Gan, Hui Cheng.* ICME 2017. [paper](https://arxiv.org/pdf/1707.04677.pdf)

- **Semantic Object Parsing with Graph LSTM.**
*Xiaodan LiangXiaohui ShenJiashi FengLiang Lin, Shuicheng Yan.* ECCV 2016. [paper](https://link.springer.com/content/pdf/10.1007%2F978-3-319-46448-0_8.pdf)

- **CelebrityNet: A Social Network Constructed from Large-Scale Online Celebrity Images.**
*Li-Jia Li, David A. Shamma, Xiangnan Kong, Sina Jafarpour, Roelof Van Zwol, Xuanhui Wang.* TOMM 2015. [paper](https://dl.acm.org/ft_gateway.cfm?id=2801125&ftid=1615097&dwn=1&CFID=38275959&CFTOKEN=6938a464cf972252-DF065FDC-9FED-EB68-3528017EA04F0D29)

- **Inductive Representation Learning on Large Graphs.**
*William L. Hamilton, Rex Ying, Jure Leskovec.* NIPS 2017. [paper](https://arxiv.org/pdf/1706.02216.pdf)

- **Graph Classification using Structural Attention.**
*John Boaz Lee, Ryan Rossi, Xiangnan Kong.* KDD 18. [paper](https://dl.acm.org/ft_gateway.cfm?id=3219980&ftid=1988883&dwn=1&CFID=38275959&CFTOKEN=6938a464cf972252-DF065FDC-9FED-EB68-3528017EA04F0D29)

- **Adversarial Attacks on Neural Networks for Graph Data.**
*Daniel Zügner, Amir Akbarnejad, Stephan Günnemann.* KDD 18. [paper](http://delivery.acm.org/10.1145/3230000/3220078/p2847-zugner.pdf?ip=101.5.139.169&id=3220078&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2E587F3204F5B62A59%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1545706391_e7484be677293ffb5f18b39ce84a0df9)

- **Large-Scale Learnable Graph Convolutional Networks.**
*Hongyang Gao, Zhengyang Wang, Shuiwang Ji.* KDD 18. [paper](http://delivery.acm.org/10.1145/3220000/3219947/p1416-gao.pdf?ip=101.5.139.169&id=3219947&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2E587F3204F5B62A59%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1545706457_bb20316c7ce038aefb97afcf4ef9668b)

- **Contextual Graph Markov Model: A Deep and Generative Approach to Graph Processing.**
*Davide Bacciu, Federico Errica, Alessio Micheli.* ICML 2018. [paper](https://arxiv.org/pdf/1805.10636.pdf)

- **Diffusion-Convolutional Neural Networks.**
*James Atwood, Don Towsley.* NIPS 2016. [paper](https://arxiv.org/pdf/1511.02136.pdf)

- **Neural networks for relational learning: an experimental comparison.**
*Werner Uwents, Gabriele Monfardini, Hendrik Blockeel, Marco Gori, Franco Scarselli.* Machine Learning 2011. [paper](https://link.springer.com/content/pdf/10.1007%2Fs10994-010-5196-5.pdf)

- **FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling.**
*Jie Chen, Tengfei Ma, Cao Xiao.* ICLR 2018. [paper](https://arxiv.org/pdf/1801.10247.pdf)

- **Adaptive Sampling Towards Fast Graph Representation Learning.**
*Wenbing Huang, Tong Zhang, Yu Rong, Junzhou Huang.* NeurIPS 2018. [paper](https://arxiv.org/pdf/1809.05343.pdf)

- **Structure-Aware Convolutional Neural Networks.**
*Jianlong Chang, Jie Gu, Lingfeng Wang, Gaofeng Meng, Shiming Xiang, Chunhong Pan.* NeurIPS 2018. [paper](http://papers.nips.cc/paper/7287-structure-aware-convolutional-neural-networks.pdf)

- **Bayesian Semi-supervised Learning with Graph Gaussian Processes.**
*Yin Cheng Ng, Nicolò Colombo, Ricardo Silva.* NeurIPS 2018. [paper](https://arxiv.org/pdf/1809.04379)

- **Mean-field theory of graph neural networks in graph partitioning.**
*Tatsuro Kawamoto, Masashi Tsubaki, Tomoyuki Obuchi.* NeurIPS 2018. [paper](http://papers.nips.cc/paper/7689-mean-field-theory-of-graph-neural-networks-in-graph-partitioning.pdf)

- **Hierarchical Graph Representation Learning with Differentiable Pooling.**
*Zhitao Ying, Jiaxuan You, Christopher Morris, Xiang Ren, Will Hamilton, Jure Leskovec.* NeurIPS 2018. [paper](https://papers.nips.cc/paper/7729-hierarchical-graph-representation-learning-with-differentiable-pooling.pdf)

- **How Powerful are Graph Neural Networks?**
*Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka.* ICLR 2019. [paper](https://openreview.net/pdf?id=ryGs6iA5Km)

- **Graph Capsule Convolutional Neural Networks.**
*Saurabh Verma, Zhi-Li Zhang.* ICML 2018 Workshop. [paper](https://arxiv.org/abs/1805.08090)

- **Capsule Graph Neural Network.**
*Zhang Xinyi, Lihui Chen.* ICLR 2019. [paper](https://openreview.net/pdf?id=Byl8BnRcYm)

### Applications

- **Discovering objects and their relations from entangled scene representations.**
*David Raposo, Adam Santoro, David Barrett, Razvan Pascanu, Timothy Lillicrap, Peter Battaglia.* ICLR Workshop 2017. [paper](https://arxiv.org/pdf/1702.05068.pdf)

- **A simple neural network module for relational reasoning.**
*Adam Santoro, David Raposo, David G.T. Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, Timothy Lillicrap.* NIPS 2017. [paper](https://arxiv.org/pdf/1706.01427.pdf)

- **Attend, Infer, Repeat: Fast Scene Understanding with Generative Models.**
*S. M. Ali Eslami, Nicolas Heess, Theophane Weber, Yuval Tassa, David Szepesvari, Koray Kavukcuoglu, Geoffrey E. Hinton.* NIPS 2016. [paper](https://arxiv.org/pdf/1603.08575.pdf)

- **Beyond Categories: The Visual Memex Model for Reasoning About Object Relationships.**
*Tomasz Malisiewicz, Alyosha Efros.* NIPS 2009. [paper](http://papers.nips.cc/paper/3647-beyond-categories-the-visual-memex-model-for-reasoning-about-object-relationships.pdf)

- **Understanding Kin Relationships in a Photo.**
*Siyu Xia, Ming Shao, Jiebo Luo, Yun Fu.* TMM 2012. [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6151163)

- **Graph-Structured Representations for Visual Question Answering.**
*Damien Teney, Lingqiao Liu, Anton van den Hengel.* CVPR 2017. [paper](https://arxiv.org/pdf/1609.05600.pdf)

- **Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition.**
*Sijie Yan, Yuanjun Xiong, Dahua Lin.* AAAI 2018. [paper](https://arxiv.org/pdf/1801.07455.pdf)

- **Few-Shot Learning with Graph Neural Networks.**
*Victor Garcia, Joan Bruna.* ICLR 2018. [paper](https://arxiv.org/pdf/1711.04043.pdf)

- **The More You Know: Using Knowledge Graphs for Image Classification.**
*Kenneth Marino, Ruslan Salakhutdinov, Abhinav Gupta.* CVPR 2017. [paper](https://arxiv.org/pdf/1612.04844.pdf)

- **Zero-shot Recognition via Semantic Embeddings and Knowledge Graphs.**
*Xiaolong Wang, Yufei Ye, Abhinav Gupta.* CVPR 2018. [paper](https://arxiv.org/pdf/1803.08035.pdf)

- **Rethinking Knowledge Graph Propagation for Zero-Shot Learning.**
*Michael Kampffmeyer, Yinbo Chen, Xiaodan Liang, Hao Wang, Yujia Zhang, Eric P. Xing.* 2018. [paper](https://arxiv.org/pdf/1805.11724.pdf)

- **Interaction Networks for Learning about Objects, Relations and Physics.**
*Peter Battaglia, Razvan Pascanu, Matthew Lai, Danilo Rezende, Koray Kavukcuoglu.* NIPS 2016. [paper](https://arxiv.org/pdf/1612.00222.pdf)

- **A Compositional Object-Based Approach to Learning Physical Dynamics.**
*Michael B. Chang, Tomer Ullman, Antonio Torralba, Joshua B. Tenenbaum.* ICLR 2017. [paper](https://arxiv.org/pdf/1612.00341.pdf)

- **Visual Interaction Networks: Learning a Physics Simulator from Vide.o** 
*Nicholas Watters, Andrea Tacchetti, Théophane Weber, Razvan Pascanu, Peter Battaglia, Daniel Zoran.* NIPS 2017. [paper](http://papers.nips.cc/paper/7040-visual-interaction-networks-learning-a-physics-simulator-from-video.pdf)

- **Relational neural expectation maximization: Unsupervised discovery of objects and their interactions.**
*Sjoerd van Steenkiste, Michael Chang, Klaus Greff, Jürgen Schmidhuber.* ICLR 2018. [paper](https://arxiv.org/pdf/1802.10353.pdf)

- **Graph networks as learnable physics engines for inference and control.**
*Alvaro Sanchez-Gonzalez, Nicolas Heess, Jost Tobias Springenberg, Josh Merel, Martin Riedmiller, Raia Hadsell, Peter Battaglia.* ICML 2018. [paper](https://arxiv.org/pdf/1806.01242.pdf)

- **Learning Multiagent Communication with Backpropagation.**
*Sainbayar Sukhbaatar, Arthur Szlam, Rob Fergus.* NIPS 2016. [paper](https://arxiv.org/pdf/1605.07736.pdf)

- **VAIN: Attentional Multi-agent Predictive Modeling.**
*Yedid Hoshen.* NIPS 2017 [paper](https://arxiv.org/pdf/1706.06122.pdf)

- **Neural Relational Inference for Interacting Systems.**
*Thomas Kipf, Ethan Fetaya, Kuan-Chieh Wang, Max Welling, Richard Zemel.* ICML 2018. [paper](https://arxiv.org/pdf/1802.04687.pdf)

- **Translating Embeddings for Modeling Multi-relational Data.**
*Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, Oksana Yakhnenko.* NIPS 2013. [paper](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)

- **Representation learning for visual-relational knowledge graphs.**
*Daniel Oñoro-Rubio, Mathias Niepert, Alberto García-Durán, Roberto González, Roberto J. López-Sastre.* 2017. [paper](https://arxiv.org/pdf/1709.02314.pdf)

- **Knowledge Transfer for Out-of-Knowledge-Base Entities : A Graph Neural Network Approach.**
*Takuo Hamaguchi, Hidekazu Oiwa, Masashi Shimbo, Yuji Matsumoto.* IJCAI 2017. [paper](https://arxiv.org/pdf/1706.05674.pdf)

- **Representation Learning on Graphs with Jumping Knowledge Networks.**
*Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, Stefanie Jegelka.* ICML 2018. [paper](https://arxiv.org/pdf/1806.03536.pdf)

- **Multi-Label Zero-Shot Learning with Structured Knowledge Graphs.**
*Chung-Wei Lee, Wei Fang, Chih-Kuan Yeh, Yu-Chiang Frank Wang.* CVPR 2018. [paper](https://arxiv.org/pdf/1711.06526.pdf)

- **Dynamic Graph Generation Network: Generating Relational Knowledge from Diagrams.**
*Daesik Kim, Youngjoon Yoo, Jeesoo Kim, Sangkuk Lee, Nojun Kwak.* CVPR 2018. [paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Kim_Dynamic_Graph_Generation_CVPR_2018_paper.pdf)

- **Deep Reasoning with Knowledge Graph for Social Relationship Understanding.**
*Zhouxia Wang, Tianshui Chen, Jimmy Ren, Weihao Yu, Hui Cheng, Liang Lin.* IJCAI 2018. [paper](https://arxiv.org/pdf/1807.00504.pdf)

- **Constructing Narrative Event Evolutionary Graph for Script Event Prediction.**
*Zhongyang Li, Xiao Ding, Ting Liu.* IJCAI 2018. [paper](https://arxiv.org/pdf/1805.05081.pdf)

- **Modeling Semantics with Gated Graph Neural Networks for Knowledge Base Question Answering.**
*Daniil Sorokin, Iryna Gurevych.* COLING 2018. [paper](https://arxiv.org/pdf/1808.04126.pdf)

- **Convolutional networks on graphs for learning molecular fingerprints.**
*David Duvenaud, Dougal Maclaurin, Jorge Aguilera-Iparraguirre, Rafael Gómez-Bombarelli, Timothy Hirzel, Alán Aspuru-Guzik, Ryan P. Adams.* NIPS 2015. [paper](https://arxiv.org/pdf/1509.09292.pdf)

- **Molecular Graph Convolutions: Moving Beyond Fingerprints.**
*Steven Kearnes, Kevin McCloskey, Marc Berndl, Vijay Pande, Patrick Riley.* Journal of computer-aided molecular design 2016. [paper](https://arxiv.org/pdf/1603.00856.pdf)

- **Protein Interface Prediction using Graph Convolutional Networks.**
*Alex Fout, Jonathon Byrd, Basir Shariat, Asa Ben-Hur.* NIPS 2017. [paper](http://papers.nips.cc/paper/7231-protein-interface-prediction-using-graph-convolutional-networks.pdf)

- **Traffic Graph Convolutional Recurrent Neural Network: A Deep Learning Framework for Network-Scale Traffic Learning and Forecasting.**
*Zhiyong Cui, Kristian Henrickson, Ruimin Ke, Yinhai Wang.* 2018. [paper](https://arxiv.org/pdf/1802.07007.pdf)

- **Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting.**
*Bing Yu, Haoteng Yin, Zhanxing Zhu.* IJCAI 2018. [paper](https://arxiv.org/pdf/1709.04875.pdf)

- **Semi-supervised User Geolocation via Graph Convolutional Networks.**
*Afshin Rahimi, Trevor Cohn, Timothy Baldwin.* ACL 2018. [paper](https://arxiv.org/pdf/1804.08049.pdf)

- **Dynamic Graph CNN for Learning on Point Clouds.**
*Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, Justin M. Solomon.* CVPR 2018. [paper](https://arxiv.org/pdf/1801.07829.pdf)

- **PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.**
*Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas.* CVPR 2018. [paper](https://arxiv.org/pdf/1612.00593.pdf)

- **3D Graph Neural Networks for RGBD Semantic Segmentation.**
*Xiaojuan Qi, Renjie Liao, Jiaya Jia, Sanja Fidler, Raquel Urtasun.* CVPR 2017. [paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Qi_3D_Graph_Neural_ICCV_2017_paper.pdf)

- **Iterative Visual Reasoning Beyond Convolutions.**
*Xinlei Chen, Li-Jia Li, Li Fei-Fei, Abhinav Gupta.* CVPR 2018. [paper](https://arxiv.org/pdf/1803.11189)

- **Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs.**
*Martin Simonovsky, Nikos Komodakis.* CVPR 2017. [paper](https://arxiv.org/pdf/1704.02901)

- **Situation Recognition with Graph Neural Networks.**
*Ruiyu Li, Makarand Tapaswi, Renjie Liao, Jiaya Jia, Raquel Urtasun, Sanja Fidler.* ICCV 2017. [paper](https://arxiv.org/pdf/1708.04320)

- **Conversation Modeling on Reddit using a Graph-Structured LSTM.**
*Vicky Zayats, Mari Ostendorf.* TACL 2018. [paper](https://arxiv.org/pdf/1704.02080)

- **Graph Convolutional Networks for Text Classification.**
*Liang Yao, Chengsheng Mao, Yuan Luo.* AAAI 2019. [paper](https://arxiv.org/pdf/1809.05679.pdf)

- **Attention Is All You Need.**
*Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.* NIPS 2017. [paper](https://arxiv.org/pdf/1706.03762)

- **Self-Attention with Relative Position Representations.**
*Peter Shaw, Jakob Uszkoreit, Ashish Vaswani.* NAACL 2018. [paper](https://arxiv.org/pdf/1803.02155)

- **Hyperbolic Attention Networks.**
*Caglar Gulcehre, Misha Denil, Mateusz Malinowski, Ali Razavi, Razvan Pascanu, Karl Moritz Hermann, Peter Battaglia, Victor Bapst, David Raposo, Adam Santoro, Nando de Freitas* 2018. [paper](https://arxiv.org/pdf/1805.09786)

