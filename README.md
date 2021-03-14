# Exercise(s) to Classify Genomic Sequences into Coding and Non-Coding
## Data
 To train, validate and test binary sequence (=time-series) classification methods we use DNA sequences that are either
  - completely coding or
  - completely non-coding
 These sequences are data from  
 Mertsch and Stanke, [End-to-end Learning of Evolutionary Models to Find Coding Regions in Genome Alignments](https://www.biorxiv.org/content/10.1101/2021.03.09.434414v1), *bioRxiv* 2021
 
## Models to Compare
 Let $k\in \{0,1,2,\dots\}$ be a model *order*. Let $x \in \{a,c,g,t\}^\ell$ be an input DNA sequence of length $\ell$.
 Let $y\in\{0,1\}$ be a *class*, here $y=1$ means *coding* (=positive) and $y=0$ (=negative) means non-coding.
 
   1. Two $k$-th order Markov chains, one for coding, one for non-coding, trained individually to maximize the likelihood of the respective data. 
   2. Like 1., but the positive model is 3-periodic.
   3. Two $k$-th order Markov chains, one for coding, one for non-coding. Then logistic regression to predict a probability of coding. Trained (discriminately) to miminize cross-entropy error (CEE).
   4. Like 3, but the positive model is 3-periodic.
   5. Like 4, but $M>2$ models are allowed and $M$ is optimized.
   6. $M$ HMMs with a fixed number of states ($n=3$) are trained jointly with logistic regression.
   
## Compare the Accuracy of above Models on the Test Data
