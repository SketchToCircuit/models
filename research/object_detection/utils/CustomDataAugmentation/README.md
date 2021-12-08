## Modifications before proto compilation
1. Add proto messages:
    - Add <code>CustomAugmentation custom_augmentation = 40;</code> to <code>oneof preprocessing_step {...}</code>
    - Add new message: <code>message CustomAugmentation {}</code>

## Modifications before object_detection installation
1. Add custom_augmentation to $PYTHONPATH
    - Add <code>export PYTHONPATH=/mnt/hdd2/Sketch2Circuit/DataProcessing/ObjectDetection/CustomDataAugmentation:$PYTHONPATH</code> to ~/.bashrc
    - For autocomplete add path to VSCode python.autoComplete.extraPath

2. Add function to '/core/preprocessor.py':
    - Create function:

            def custom_augmentation(image, boxes):
                with tf.name_scope('CustomAugmentation', values=[image, boxes]):
                    return custaug.augment(image, boxes)

    - Add <code>custom_augmentation: (fields.InputDataFields.image, fields.InputDataFields.groundtruth_boxes)</code> to <code>prep_func_arg_map</code>

3. Add to '/builders/preprocessor_builer.py':
    - Add <code>'custom_augmentation': preprocessor.custom_augmentation</code> to <code>PREPROCESSING_FUNCTION_MAP</code>