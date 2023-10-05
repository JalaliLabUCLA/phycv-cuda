total_iterations=0       # Initialize a counter for total iterations
total_detections=0      # Initialize a counter for total detections

for file in ~/PhyCV_CUDA/assets/input_images/train_images/*.png; do
    if [ "$total_iterations" -lt 50 ]; then
        ((total_iterations++))  # Increment the total iterations counter
        detection_output=$(./vevid -d -i "$file" -p 1280,720,10,0.1,4,2.5)  # Run your command and capture the output

        # Use 'grep' to count the number of detections and add it to the total detections counter
        detections=$(echo "$detection_output" | grep -c "detected obj")
        ((total_detections += detections))

        # Print the detection output for the current image
        echo "Detections in $file:"
        echo "$detection_output"
    else
        break  # Exit the loop when 200 files have been processed
    fi
done

echo "Total iterations: $total_iterations"
echo "Total detections in the first 200 images: $total_detections"

