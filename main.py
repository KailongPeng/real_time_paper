def zip_data():
    """
#!/bin/bash

# Set the path to the subjects folder
subjects_folder="/gpfs/milgram/scratch60/turk-browne/kp578/organizeDataForPublication/real_time_paper/data/subjects"
zippedFolder="/gpfs/milgram/scratch60/turk-browne/kp578/organizeDataForPublication/real_time_paper/data/zipped/"
mkdir "${zippedFolder}"
# Loop through each subxxx folder and create a zip file
for folder in "$subjects_folder"/*; do
    if [ -d "$folder" ]; then
        sub_id=$(basename "$folder")
        echo zip -r "${zippedFolder}/$sub_id.zip" "$folder"
        zip -r "${zippedFolder}/$sub_id.zip" "$folder"
    fi
done

    """


# subject's data can be found in xxx and store in the ./data folder


# data_preprocess and prepare data for plotting fig2c, fig4b and fig5
"""sbatch --array=1-1 data_preprocess/data_preprocess.sh"""


# adaptive threshold analysis code  # fig2b
"""fig2b/fig2b.py"""


# X Y co-activation analysis code  # fig2c
"""fig2c/fig2c.py"""


# behavioral categorical perception analysis code  # fig3c
"""fig3c/fig3c.py"""  # Done


# fig3d
"""fig3d/fig3d.py"""  # Done


# hippocampus subfield integration analysis code
"""fig3e/fig4b.py"""


# behavioral relevance of neural representational change. fig5
"""fig5/fig5.py"""


# data
    # subjects
        # recognition run
        # feedback run

        # behav
        # adaptiveThreshold


        # chosenMask
        # ROI mask for hippocalmpus subfield and visual areas







        # T1
        # T2
        # fmap


# deface
    # Nick: https://github.com/poldracklab/pydeface
    # bet is fine


# only focusing on the figure generating code, avoiding all other stuffs.

# feedback ROI (megaROI) selection and visualization code

# real-time processing code? discuss further
# code for megaROI selection and data collection? discuss further





# feedback ROI (megaROI) selection and visualization code
# adaptive threshold analysis code
# X Y co-activation analysis code
# categorical perception analysis code
# hippocampus subfield integration analysis code
# behavioral relevance of neural representational change
# real-time processing code?
