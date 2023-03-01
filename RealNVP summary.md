# RealNVP
- Architecture différente des modèles présentées précédemment (VAE)
- L'entrainement se fait par la maximisation de la vraisemblance (pas de loss), 
  en utilisant une prior simple (gaussienne) p_z que l'on compose avec des fonctions inversibles (g)
  p_x(x) = p_z(z) | det( jacob_g) |^-1

- Difficulté de cette architecture: le calcul du déterminant de la jacobienne est très couteux

# Architecture
- RealNVP propose une architecture inversible et stable qui fait un mapping entre la distribution des données et celle d'un espace latent par l'intermédiaire de 2 flows
  - Affine Coupling Layers:
    * S'attaquent au problèmes de difficultés de calcul de la jacobienne, en se basant sur l'idée que les 
      déterminants faciles à calculer correspondent aux matrices triangulaires. Tout en gardant une transformation complexe de la donnée en entrée 
      pour moduler la prior "simple"
    * L'architecture d'un coupling layer peut être quelconque, du moment que sa sortie correspond à une forme donnée dans l'article qui est imposée pour le calcul du déterminant
    * L'architecture choisie est celle d'une convolution
    * Un mask (stride) peut être introduit, ce qui produit une image en sortie qui donne toujours un déterminant facile à calculer et permet de réduire la complexité globale du modèle
    * On peut aussi stacker des layers de couplages grâces aux propriétéss sur les dérivées, les composés et le déterminant
    * On peut aussi mettre un paramètre à apprendre dans ce layer au niveau de la sortie (c'est le cas dans l'architecture proposée dans le papier)
  - Squeezing Layers:
    * Opération de reshape vers les channels (forme de filtrage)
  - Opération de batch normalization pour augmenter la robutesse du modèle, et qui rentre toujours de façon controlé dans la Jacobienne, donc facile à calculer
    
- L'ensemble formé par (couplage + squeezing + couplage) s'intègre dans un CNN ResNet (+ skip-connections)

# Avantages de RealNVP
- stable dans le training (comparé aux GANs)
- inférence exacte (pas d'ELBO, qui a tendance à limiter la capacité à apprendre les représentations en hautes dimensions), donc on a des résultats plus fins
- efficace computationnellement parlant (comparé aux modèles auto-régressifs, qui ne possèdent de plus pas de modélisation latentes)
- Globalement, c'est une approche qui fait le pont entre GAN, VAE, Auto-regresseurs 
