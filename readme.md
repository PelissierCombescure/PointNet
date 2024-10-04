**get_critical_points** : A executer dans **PoinNet0_env**

*outputs* : A partir d'un objet 3D, peut générer 3 fichiers obj contenant les nuages de pts représentant :

1. Les 1024 de l'objet avec en rouge les pts critiques --> critical_pts_AND_'+file_name+'.obj
2. Les 1024 de l'objet avec en rouge les pts critiques --> critical_pts_AND_'+file_name+'.obj
3. Les 1024 points de l'obejt après son passage dans PointNet --> all_pts_OF_'+file_name+'.obj

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
        'num_points': pcd.squeeze(0).tolist()      
    }



*inputs* : 
* input_path: Path to the input .off file.
* output_path: Directory to save outputs.
* kind_of_outputs : dictionnaire indicant quels fichiers on souhaite générer parmi les 3, par défaut les 3 sont générés.

*command line* : 
 *  python3 get_critical_points.py ModelNet10/chair/test/chair_0948.off outputs

OU

* python3 get_critical_points.py ModelNet10/chair/test/chair_0948.off outputs --kind_of_outputs '{"critical and non-critical points": true, "only critical points": false, "objet": false}'
