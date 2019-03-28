# Machine Learning / Deep Learning

## Problématique

- Calculer la fonction F qui associe à toute image l'unique objet représenté dessus.

Remarques:
- Une image peut être codée par une matrice à 3 dimensions (L=largeur, H=hauteur, B=bandes)
- Un ensemble fini de N objets peut être énuméré par N entiers 
- La fonction F se réduit à associer à chaque matrice (L,H,B) un entier dans [0,N[


Reformulation:
- On veut construire une fonction G qui approxime au mieux F: X --> Y
  - X : espace matriciel (L,H,B) = (128,128,3) 
  - Y : un réél dans [0,N[

## Questions
- Comment construire une fonction G qui aproxime F ?
- Comment vérifier que G approxime F, comment comparer deux fonctions candidates G ?
- Comment représenter la fonction F, comment définir notre cible F ?

# Réponses

- Comment représenter F :
  - Utiliser un ensemble de "samples" (x,y) tels que, par définition, y = F(x)
  - https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html

- Mesurer la précision de G :
  - Définir une distance (loss) entre deux valeurs dans Y
  - Etant donné une représentation R de F : R={ (x,y) | y = F(x) }
    - Pour chaque x, calculer la distance entre G(x) et F(x) : loss(F(x), G(x))
    - Calculer la somme / moyenne des distances calculées pour chaque x
  - Plus loss(G,F) est petit, plus G approxime bien F

- Approximer F, un problème d'optimisation ?
  - Imaginer F comme un récipient en fer (un moule) : les couples (x,y) du dataset 
  - Imaginer G_0 comme étant une boule d'argile maléable
    - un grand nombre de paramètre rééls : W
    - une fonction associant à tout (x,W) une valeur y' : un produit scalaire par exemple
  
  - Essayer de faire rentrer G dans le moule F : calculer loss(G,F)
  - Noter les differences entre G et F à la surface de F : gradient entre G(x) et F(x) par rapport à W
  - Modifier légèrement G pour réduire l'erreur (loss)
  - Si tout va bien, après un grand nombre d'itérations, G converge vers F
  
  - https://towardsdatascience.com/back-propagation-demystified-in-7-minutes-4294d71a04d7
 