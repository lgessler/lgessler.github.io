import json

with open("spanish_dict_10.json") as f:
    small_dict = json.load(f)
with open("spanish_dict_100.json") as f:
    large_dict = json.load(f)

active_dict = large_dict

def linear_search(dictionary, target):
    steps = 0
    for entry in dictionary:
        steps += 1
        if entry["form"] == target:
            return entry["meaning"], steps
    return None, steps

meaning, steps = linear_search(active_dict, "perro")
print(f"Found meaning '{meaning}' in {steps} steps")
meaning, steps = linear_search(active_dict, "trabajar")
print(f"Found meaning '{meaning}' in {steps} steps")

def binary_search(dictionary, target):
    # First sort by Spanish word
    steps = 0
    left = 0
    right = len(dictionary) - 1
    
    while left <= right:
        steps += 1
        mid = (left + right) // 2
        if dictionary[mid]["form"] == target:
            return dictionary[mid]["meaning"], steps
        elif dictionary[mid]["form"] < target:
            left = mid + 1
        else:
            right = mid - 1
    return None, steps

sorted_dict = sorted(active_dict, key=lambda x: x["form"])
meaning, steps = binary_search(sorted_dict, "trabajar")
print(f"Found meaning '{meaning}' in {steps} steps")

def create_indexed_dictionary(dictionary):
    return {entry["form"]: entry for entry in dictionary}

def index_search(index, target):
    steps = 1  # just one step to look up in the index
    return index.get(target), steps

# Create the index once
indexed = create_indexed_dictionary(active_dict)

# Now we can look up words very quickly
entry, steps = index_search(indexed, "trabajar")
print(f"Found meaning '{entry['meaning']}' in {steps} steps")