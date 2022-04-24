label_to_num = {
    'any': 0,
    'epidural': 1,
    'subdural': 2,
    'subarachnoid': 3,
    'intraventricular': 4,
    'intraparenchymal': 5,
}
num_to_label = {v:k for k,v in label_to_num.items()}