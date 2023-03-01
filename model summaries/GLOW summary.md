# Glow: Generative flow with 1x1 Convolution

Modèle génératif basé sur des flows.
Comme tous ces modèles de flows, on est sur une optimization de la log likelihood.
Cela a l'avantage d'avoir une traçabilité exact des variables latentes + parallélization.

L'architecture GLOW propose un flow particulier: Convolution 1x1 inversible.

Ce flow a montré une amélioration importante des résultats proposées auparavant selon le papier, avec une génération super réalistes et un entrainement efficace avec beaucoup de données.

# Cadre théorique

Objectif typique d'une architecture de flow:
- MAP optimization
Structure typique de flows normalisants:
- z ~ p
- x = g(z)  où g inversible: z = f(x)
Inversibilité permet l'utilisation de la formule de changement de variable, avec le challenge de calculer le déterminant.
Donc on cherche des bons flows qui rendent le calcul simple, i.e. matrices triangulaires.

# Architecture GLOW

L'architecture GLOW s'appuit sur le bloc de flow suivant (voir fig.1):
![image](https://user-images.githubusercontent.com/78101027/222218110-a452093d-4a03-4b2a-96c8-9a9e0cf222c1.png)
![image](https://user-images.githubusercontent.com/78101027/222218229-8669cbff-dccb-4e90-bae2-b12cce3bbe5b.png)

## Actnorm

Cette couche consiste en une "batch normalization" pour résoudre les classiques rencontrés dans les entrainements des modèles de deep learning.
Pour éviter l'introduction de bruits de normalization par GPU, la variante "Actnorm" utilise une transformation affine 
par channel, initialisée telle que le bruit d'activation soit nul en moyenne par channel avec une variance unitaire.

## Convolution 1x1 inversible

Cette couche consiste en une permutation de l'ordre des channels, paramétrisée par W, que le modèle apprend lors de l'entrainement.
Pour des raisons d'efficacité computationelle, on peut écrire W avec une décomposation de LU $W = P \times L \times (U + diag(s))$.

où $P$ est une matrice de permutation, $L$ in matrice triangulaire inférieure, $U$ une matrice triangulaire supérieure, $s$ un vecteur.

L'initiatilisation est faite en tirant une matrice de rotation W aléatoire, que l'on décompose ensuite selon LU.

## Couplage affine

Voir l'architecture RealNVP où cela était déjà détaillé.

## Architecture générale

L'architecture générale consiste en un ResNet convolutif dont les blocs sont données par la figure suivante (fig.2).

![image](https://user-images.githubusercontent.com/78101027/222222421-6850bd95-def3-420b-952c-add36241c13c.png)

# Gain de l'architecture GLOW

![image](https://user-images.githubusercontent.com/78101027/222222542-35d1473e-008a-4971-86b7-32afe040ac48.png)
