For the results saved in this folder 3 different ways of calculating the 
volume of the bowl are used:
"all": where all segmented slices are multiplied with the distance between 
        two slices
"no_top": where all but the top slice where multiplied with the distance between
        two slices
"offset": where the top was multiplied slice was multiplied with the distance
          from the top slice to the the heighest slice in the bowl and the bottom
          slice was multiplied with the distance between two slices+the offset between
          the lowest point in the bowl and the bottom slice

The volumes are calculated for segmentation masks at the resolution of the
monkey MRI as well as at a higher resolution (I need to look up the specific resoultion)
Additionally for the segmentation masks at the MRI resolution the inner pixel of the
myocardium was added to the bp = incre_bp and the volumes where calculated
and the outer pixel of the blood pool was added to the myocardium = decre_pb and the
bp was calculated. 
Additionally the mesh was scaled by different scaling factors and for the same
z-distance as in the MRI machine the volume of the bowl where calculated = _scale.
