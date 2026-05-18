import pickle

MEMORY_FILE = "face_memory.pkl"

# Load
with open(MEMORY_FILE, "rb") as f:
    known_faces = pickle.load(f)

print("\nStored names:\n")

for i, (name, _) in enumerate(known_faces):
    print(f"{i}: {name}")

# Choose entry to remove
index = int(input("\nEnter index to remove: "))

removed = known_faces.pop(index)

print(f"\nRemoved: {removed[0]}")

# Save back
with open(MEMORY_FILE, "wb") as f:
    pickle.dump(known_faces, f)

print("\nMemory updated!")