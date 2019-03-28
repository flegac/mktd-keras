# Configurer et tester son environnement Keras

## Miniconda

- Télécharger Miniconda pour votre OS - python 3.x : [miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Installer Miniconda (cf. [instructions](https://docs.conda.io/en/latest/miniconda.html))

## Environnement Conda
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#

**Mac-OS:** utilisez "bash", les autres shells ne sont pas tous compatibles avec conda.

- Créer un environnement à partir de keras_env.yml:
```bash
conda env create -f keras_env.yml
```
- Activez votre environnement :
```bash
conda activate keras_env
```
- Tester l'environnement
```bash
python keras_test.py
```
Vous devez obtenir la sortie suivante :
```
Using TensorFlow backend.
keras version 2.2.4
```

## Dépannage
https://docs.conda.io/projects/conda/en/latest/user-guide/troubleshooting.html

- En cas de problème de certificat SSL avec Conda [stackoverflow](https://stackoverflow.com/questions/33699577/conda-update-failed-ssl-error-ssl-certificate-verify-failed-certificate-ver)
```bash
conda config --set ssl_verify <pathToYourFile>.crt
```
