Все что нужно для запуска в lab4.py

ПРОСЬБА - не знаем почему, но гит сопротивляется добавить дирикторию с натренированными моделями для распознования объекта, поэтому перед запуском lab4.py, добавьте файл written_models с лабораторной работы 3 (https://github.com/StMichael99/Vision/tree/master/pattern_recognition/lab3)


Если хотим поменять видео, то нужно изменить
movie_name
Если хотим изменить обьект, то нужно изменить
image
original
replace
( будьте внимательны, классификатор узнает только hryvnia, stepler, karabin, trash )
Если хотим изменить классификтор, то нужно изменить
pred=logreg[desk].predict(temp)[0] на (например) pred=svm[desk].predict(temp)[0] 
( Внимание, доступны только logreg svm randomforest )
Если хотим изменить дескриптор ( возможные варианты brisk, sift orb ), то нужно поменять
desk
init_desk




Ссылка на протокол
https://docs.google.com/document/d/14GTElh3b5Vu4rQrsN-EFLw-Yiz6AUZ7-Igxa0A_iLwM/edit?usp=sharing
