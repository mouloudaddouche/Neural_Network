
Mecanisme d'installation: 

Ouvrir le terminal et exécuter les commandes suivantes:
	1. sudo apt-get install python2.7
	2. sudo apt-get install pip
	3. installer les packages neccesaires : pip install < keras, Tenserflow, numpy, pandas , matpolib , Pyside2>
Ou bien se placer dans le dossier "codes" et appuyer 2 fois sur l'executable generé nommé: Application

Manuel:

Au lancement de l’application, un menu principal est affiché à l’utilisateur avec les options : prétraitement texte, prétraitement image, apprentissage, test et quitter.
— En choisissant la tâche prétraitement texte, l’utilisateur peut prétraiter n’importe quel dataset alphanumérique en sélectionnant ses paramètres (normalisationet remplissage de valeurs manquantes).
 L’utilisateur se voit également afficher le da-taset avant et après le prétraitement sous forme de table, boxplot et histogrammes.Des informations statistiques s’accompagnent.
— En choisissant la tâche prétraitement image, l’utilisateur peut visualiser chaque image du dataset, les redimensionner et voir le résultat de leur normalisation après prétraitement.
— En sélectionnant l’option apprentissage, une fenêtre intermédiaire sera affichée qui permettra à l’utilisateur de charger un dataset, charger ou créer un réseau de neurones et lancer l’apprentissage.
— En décidant de créer un nouveau réseau l’utilisateur est emmené sur la fenêtre de création de réseau où il pourra graphiquement configurer son réseau suivant lesparamètres :Nombre d’epochs,Learning rate.Pourcentage de répartition du dataset,Type de répartition du dataset : régulier ou aléatoire.
— Une fois le dataset prétraité et la création du réseau achevée, il est possible de lancer l’apprentissage après avoir défini les paramètres de ce dernier :Nom du réseau,Type de couches : Input, Output, Convolutional, Flatten, Fully connected,Nombre d’output pour chaque couche,Fonctions d’activation : Sigmode, Tanh, Softmax, Rectified linear unit.
Les statistiques de l’apprentissage seront ensuite affichées à l’écran.
— Une fois le réseau entrainé, l’utilisateur a la possibilité de le tester, pour celail choisit l’option test du menu principal, une fenêtre lui sera donc affichée avec la possibilité de charger un réseau et un fichier à tester puis de lancer le test.
— Finalement l’utilisateur pourra cliquer sur le bouton "quitter" pour mettrefin à l’exécution de l’application .

