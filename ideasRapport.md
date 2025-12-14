## Structure:

resumé:
1 - application de "Adaptive Mixtures of Local Experts" pour la classification d'images
  1.1 - objectif et metodologie
  1.2 - resume
  1.3 - premiere adaptation de la fonction de loss
    utilise "original + CE [+ balance]"
  1.4 - fonction de loss originale
  1.5 - utilisation de la cross entropie
  1.6 - conclusions
2 - application de "SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER" pour la classification d'images
  2.1 - objectif et metodologie
  2.2 - resume
  ...
  2.* - conclusions
3 - conclusions
4 - citations



(TOTEST) -> indicate something to explore and talk about in the results

first benchmark the times of the different steps of the training

## for the first paper: 
use a linear combination of the outputs of the experts based on the coefs given by the gating network

 - use a custom loss func: (see 1.5)
    $ E^c = -log(\Sigma_{i=1}^{N_{experts}}p_i^c*exp(-0.5*||y-o_i^c||^2)) $ where $c$ is a sample, $p_i^c$ the (softmaxed) weigth for the expert $i$ and $o_i^c$ the output of the expert $i$.
    Because $ ||y-o_i^c||^2 $ is the formula for MSE, the adapted formula for CrossEntropy with logits and class index is $ E_j^c = -log(\Sigma_{i=1}^{N_{experts}}p_i^c*exp(-0.5*CE_j(o_i^c))) $
    (note: CE with logits (K classes), and $ y = $ the class index $j$ is: $
     CE_j(\hat{y}) = -log(\frac{\exp(\hat{y}_j)}{\sum_{k=1}^K \exp(\hat{y}_k)}) $ )
 - (TOTALK) sould speed up training
 - (TOTALK) if it enable better performances of the same training duration/epoches (-> will also show if it is slower)
 - (TOTALK) find about the % of use of the experts (relative to: each dataset if multiple used, the whole dataset)

### my maths

first lets define:
 * $K$ the number of experts
 * $C$ the number of classes
 * $o_i$ the logits output of the $i$-th expert
 * $p_i$ the class probabilties output of the $i$-th expert
   * $p_i = {\exp(o_i)} / {\sum_{c=1}^K \exp(o_{i,c})} $
 * $g_i$ the gate for the $i$-th expert
 * $y$ the expected output
   * its a vector of the same shape as $o_i$ for the original paper
   * in my formula it will be the correct class to predict (so $ 0 \le y \lt C $)

original loss from the (Jacobs et al., 1991):
-> $ \mathcal{Loss} = -log(\Sigma_{i=1}^{K}g_i*exp(-0.5*||y-o_i||^2)) $  ([1.1](#originalLoss))
-> $ \frac{\partial \mathcal{Loss}}{\partial p_{j,k}} = - \frac{g_j * exp(-0.5*||y-p_i||^2)}{\Sigma_{i=1}^{K}{g_i*exp(-0.5*||y-p_i||^2)}} (y_{k} - p_{j,k}) $
-> $ \frac{\partial \mathcal{Loss}}{\partial g_j} = - \frac{-exp(0.5*||y-p_i||^2)}{\Sigma_{i=1}^{K}g_i*exp(-0.5*||y-p_i||^2)} $

my first aproche to adapte this to a classification probleme:
we can remarque that the equation for the mean squared error is
-> $ MSE(\hat{y}, y) = \left\| y - \hat{y} \right\|^2 $
so i rewrote the original loss to:
-> $ \mathcal{Loss} = -log(\Sigma_{i=1}^{K}g_i*exp(-0.5*MSE(o_i, y))) $ ([1.2](#originalLossRewrote))

Because i am working on a classification problem,
    the logits output of the experts can't be direct merged with the weigthed sum,
    is used softmax to obtain the probabilty distribution of the clases: $p_i$.
i replaced the MSE by the cross entropy loss of each expert $i$ :
-> $ CE_i(y) = -log(p_{i,y}) $
-> $ \mathcal{Loss} = -log(\Sigma_{i=1}^{K}g_i*exp(-0.5*CE_i(y))) $ ([1.3](#customLossCE))
-> $ \frac{\partial \mathcal{Loss}}{\partial p_j} = \frac{\partial f(p_j)}{\partial p_j}\frac{0.5 * g_j * exp(-0.5*f(p_j))}{\Sigma_{i=1}^{K}{g_i*exp(-0.5*f(p_i))}} $
-> $ \frac{\partial \mathcal{Loss}}{\partial g_j} = - \frac{-exp(0.5*f(p_j))}{\Sigma_{i=1}^{K}g_i*exp(-0.5*f(p_i))} $
This is the first loss that i used for my training.
... talk about the results of this methode

I later realized that the equation ([1.3](#customLossCE)) was equivalent to the following:
-> $ \mathcal{Loss} = -log(\Sigma_{i=1}^{K}g_i*exp(log[\sqrt{p_{i,y}}])) $
-> $ \mathcal{Loss} = -log(\Sigma_{i=1}^{K}g_i*\sqrt{p_{i,y}}) $
-> $ \frac{\partial \mathcal{Loss}}{\partial p_j} = -\frac{0.5*g_j}{\sqrt{p_j}*\Sigma_{i=1}^{K}{g[i]*\sqrt{p_i}}} $ 
-> $ \frac{\partial \mathcal{Loss}}{\partial g_j} = -\frac{\sqrt{p_j}}{\Sigma_{i=1}^{K}{g_i*\sqrt{p_i}}} $ 
This formula is the log likelihood of the sqare root of 
    the probabily distribution predicted by the experts.
if we remembered that the output of the network is:
-> $ \hat{y} = \Sigma_{i=1}^{K}g_i*p_i $
The simplified version of the equation ([1.3](#customLossCE))
    looks just like just appling the cross entropy loss to the 
    output of the model. 
-> $ \mathcal{Loss} = -log(\Sigma_{i=1}^{K}g_i*p_i) $ ([1.4](#baseLossCE))
-> $ \frac{\partial \mathcal{Loss}}{\partial p_i} = \frac{-g[j]}{\Sigma_{i=1}^{K}g_i*p_i} $ 
-> $ \frac{\partial \mathcal{Loss}}{\partial g_i} = \frac{-p[j]}{\Sigma_{i=1}^{K}g_i*p_i} $
... talk about the results of this methode

... put the table with all the methodes
