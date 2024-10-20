<span style="color:red"> ATTENTION  </span>
* indice face commence à 1 pour OBJ 
* indice face commence à 0 pour OFF  

## Files

<p style="font-size:18px"><span style="color:cyan">get_critical_points</span></p> 

<span style="color:red">Attention : </span> Tous les modèles 3D sont transformés en nuage de pts avec N pts (cf fonction `setup_dataset()` dans `get_critical_points.py`). Peut importe N, à la fin il y aura tjrs 1024 valeurs qu'on obtient en appliquant un argmax : passage d'une matrice de taille Nx1024 --> 1x1024, et on récupère les indices. 

<span style="color:pink"> *outputs* : </span>  A partir d'un objet 3D, peut générer 4 fichiers obj contenant les nuages de pts représentant :

1. Les N pts de l'objet avec en rouge les pts critiques --> critical_pts_AND_'+file_name+'.obj
2. Les N pts de l'objet avec en rouge les pts critiques --> critical_pts_AND_'+file_name+'.obj
3. Les N pts de l'obejt après son passage dans PointNet --> all_pts_OF_'+file_name+'.obj
4. Les N pts de l'objet avec les critical pts en gradient, cad en fonction de leur importance/occurences des sommets (parmi les N init) après le argmax (cmap : Reds)

Et un fichier json avec comme information :

    point_cloud_info = {
        'input_path': input_path,
        'folder': Path(input_path).parent.name,
        'file_name': file_name,
        'category' : file_name.split('_')[0],
        'numero' : file_name.split('_')[1],
        'nb_critical_points': len(idx_critical_points),
        'shape': pcd.squeeze(0).numpy().shape,
        'critical_points_indices': list(idx_critical_points),        
        'pts_cloud': pcd.squeeze(0).tolist() ,
        'occurences': occur.tolist()    
    }

<span style="color:pink">  *inputs* </span>: 
* input_path: Path to the input .off file.
* output_path: Directory to save outputs.
* kind_of_outputs : dictionnaire indicant quels fichiers on souhaite générer parmi les 3, par défaut les 3 sont générés.

<span style="color:pink">  *Conda env* </span>: `PoinNet0_env`

<span style="color:pink">  *command line* </span>: 
 *  python3 get_critical_points.py /../odelNet10/chair/test/chair_0948.off outputs

OU

* python3 get_critical_points.py /../ModelNet10/chair/test/chair_0948.off outputs --kind_of_outputs '{"critical and non-critical points": true, "only critical points": false, "objet": false}'

---
<p style="font-size:18px"><span style="color:cyan">run_het_critical_points.ipynb</span></p> 
<span style="color:red">Attention : </span> Le notebook est à adapter en fonction de la structure du dossier initial et de celle du dossier de sortie.

<span style="color:pink"> *outputs* : </span>  Crée les fichiers des pts critiques pour chaque modèle 3d donné

<span style="color:pink">  *inputs* </span>: 
 * Chemin dossier avec les fichiers de modèles 3D
 * Où les stoker


<span style="color:pink">  *Conda env* </span>: `PoinNet0_env`






---
<p style="font-size:18px"><span style="color:cyan">xxx</span></p> 

<span style="color:pink"> *outputs* : </span>  ...

<span style="color:pink">  *inputs* </span>: 
...

<span style="color:pink">  *Conda env* </span>: `XXXX`

<span style="color:pink">  *command line* </span>: 
...


