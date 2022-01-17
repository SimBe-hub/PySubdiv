from pysubdiv.main.data import files

# import a mesh to define volumes on
control_cage = files.read('anticline/anticline_cc_with_bb_2.obj')

# call the method .define_volumes(). The method takes only a mesh(self) as an input. And there is one prompt which takes
# the user input in Line 719. The code will interpret integer numbers and 'c' to cancel. The integers are the volumes
# indexes. In the program we can pick faces with two and the selection is marked with white edges.
# We can select multiple faces by picking more faces. Already selected faces will be deselected. I wrote an automatic
# picker which can be activated by pressing the purple button, when only one face is picked. Otherwise a message will
# be printed to console that only one face should be selected. The automatic selector will select the manifold part
# the first face is part of. Sometimes it will miss one or two faces. But this is not a problem they can be added then
# by manual picking (non-manifold parts cause problems here)
# by pressing the red button we can assign the selection a volume
# the darkblue button will make the selected faces invisible (good for selecting the manifold parts)
# green will clear the selection
# yellow will make the invisible faces reappear
# light blue undo the last face
# brown will print the volumes
control_cage.define_volumes()
# volumes will be saved to the property
print(control_cage.volumes)
# or in the data array
print(control_cage.data['volumes'])
# This function does not return a new mesh but will add data
# so we have to save the data array if we want to work with them.
control_cage.save_data('anticline/save_me_i_am_the_data')

# for the optimizer there is not change in the parameter i just fixed a bug so i think you just have to take the updated
# code and add your changes