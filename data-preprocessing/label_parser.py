import csv

# hashmap format: {img_id: [is_fish, [((x_min, x_max), (y_min, y_max), label), ...]]}
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
            is_fish = row['is_fish'].lower()=='true' # Convert to boolean

            # Create the tuple for this row
            data_tuple = ((x_min, x_max), (y_min, y_max), label)

            # Check if the img_id is already a key in the dictionary
            if img_id in data_map:
                data_map[img_id][1].append(data_tuple)
                # Update is_fish if necessary
                if is_fish:
                    data_map[img_id][0] = True
            else:
                # Initialize the entry with is_fish and the list of data tuples
                data_map[img_id] = [is_fish, [data_tuple]]

    return data_map
