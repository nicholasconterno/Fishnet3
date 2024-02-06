import csv


# hashmap format: {img_id: [((x_min, x_max), (y_min, y_max), label), ...]}
def process_data(file_path):
    data_map = {}

    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            img_id = row['img_id']
            x_min = row['x_min']
            x_max = row['x_max']
            y_min = row['y_min']
            y_max = row['y_max']
            label = row['label_l1']

            # Create the tuple for this row
            data_tuple = ((x_min, x_max), (y_min, y_max), label)

            # Check if the img_id is already a key in the dictionary
            if img_id in data_map:
                data_map[img_id].append(data_tuple)
            else:
                data_map[img_id] = [data_tuple]

    return data_map

# Use the function with the path to your CSV file
data = process_data('data/fishnet_labels.csv')
c=0
for i in data.keys():
    print(i, data[i])
    c+=1
    if c==5:
        break
print(len(data))