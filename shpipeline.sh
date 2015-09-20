#!/bin/bash
#./thisfilename cars planes (e.g.)
#included arguments are optional and can range from 1-all possible arguments

rm -r traininginput trainingoutput testinginput testingoutput $(pwd)/predictions #make sure the directories do not already exist before creating
rm runreport.txt

mkdir traininginput
mkdir trainingoutput
mkdir testinginput
mkdir testingoutput
mkdir $(pwd)/predictions

current_directory=$(pwd)
predictions_directory=$(pwd)/predictions

echo "Working in $current_directory. This is where training images, testing images and predicted optical flow images will be stored."


#load initial 
for image_directory in $@
do 
    num_files=$(ls -l $current_directory/dataset/$image_directory/sotam | grep "[0-9]*\.jpg" | wc -l)
    rand_perm=$(shuf -i 1-$num_files)
    test_set_size=$(($num_files/10))

    counter=0
    for index in $rand_perm
    do
	in_file=$current_directory/dataset/$image_directory/1/$index.jpg
	out_file=$current_directory/dataset/$image_directory/sotam/$index.jpg

	if [ -f $in_file ] && [ -f $out_file ] && [ $counter -lt $test_set_size ];
	then
	    cp $current_directory/dataset/$image_directory/1/$index.jpg testinginput
	    cp $current_directory/dataset/$image_directory/sotam/$index.jpg testingoutput
	elif [ -f $in_file ] && [ -f $out_file ] && [ $counter -ge $test_set_size ];
	then
	    cp $current_directory/dataset/$image_directory/1/$index.jpg traininginput
	    cp $current_directory/dataset/$image_directory/sotam/$index.jpg trainingoutput
	fi

	counter=$((counter+1))
    done
done

th cnn_flow1.2alua $current_directory $predictions_directory
