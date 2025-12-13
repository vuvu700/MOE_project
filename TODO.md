(TOTEST) -> indicate something to explore and talk about in the results

first benchmark the times of the different steps of the training

# for the first paper: 
use a linear combination of the outputs of the experts based on the coefs given by the gating network

 - use a custom loss func: (see 1.5)
    $ E^c = -log(\Sigma_{i=1}^{N_{experts}}p_i^c*exp(-0.5*||y-o_i^c||^2)) $ where $c$ is a sample, $p_i^c$ the (softmaxed) weigth for the expert $i$ and $o_i^c$ the output of the expert $i$.
    Because $ ||y-o_i^c||^2 $ is the formula for MSE, the adapted formula for CrossEntropy with logits and class index is $ E_j^c = -log(\Sigma_{i=1}^{N_{experts}}p_i^c*exp(-0.5*CE_j(o_i^c))) $
    (note: CE with logits (K classes), and $ y = $ the class index $j$ is: $
     CE_j(\hat{y}) = -log(\frac{\exp(\hat{y}_j)}{\sum_{k=1}^K \exp(\hat{y}_k)}) $ )
 - (TOTALK) sould speed up training
 - (TOTALK) if it enable better performances of the same training duration/epoches (-> will also show if it is slower)
 - (TOTALK) find about the % of use of the experts (relative to: each dataset if multiple used, the whole dataset)