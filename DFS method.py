import matplotlib.pyplot as plt

class ImageSearchDFS:
    def __init__(self, image):
        self.image = image
        self.visited = [[False for _ in range(len(image[0]))] for _ in range(len(image))]
        self.components = []

    def is_valid(self, row, col):
        return 0 <= row < len(self.image) and 0 <= col < len(self.image[0]) and not self.visited[row][col]

    def dfs(self, row, col, component):
        # Define the possible moves (up, down, left, right)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Mark the current cell as visited
        self.visited[row][col] = True
        component.append((row, col))

        # Perform DFS for all valid neighbors
        for move in moves:
            new_row, new_col = row + move[0], col + move[1]
            if self.is_valid(new_row, new_col) and self.image[new_row][new_col] == 1:
                self.dfs(new_row, new_col, component)

    def find_connected_components(self):
        # Iterate through all cells in the image
        for row in range(len(self.image)):
            for col in range(len(self.image[0])):
                if not self.visited[row][col] and self.image[row][col] == 1:
                    # Found a new connected component
                    component = []
                    self.dfs(row, col, component)
                    self.components.append(component)

        return self.components

    def plot_image(self):
        plt.imshow(self.image, cmap='gray', interpolation='none')
        plt.title('Connected Components')
        plt.colorbar()
        plt.show()


# Example usage:
if __name__ == "__main__":
    # Example 2D grid representing the image
    image = [
        [1, 1, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 1, 1, 0]
    ]

    # Create an instance of the ImageSearchDFS class
    image_search = ImageSearchDFS(image)

    # Find the locations of connected components
    connected_components = image_search.find_connected_components()

    # Set the coordinates of the connected components to 1 and all others to 0
    result_image = [[2 if (i, j) in component in connected_components else 0 for j in range(len(image[0]))] for i in range(len(image))]

    # Plot the resulting image
    image_search.plot_image()
    plt.imshow(result_image, cmap='gray', interpolation='none')
    plt.title('Resulting Image')
    plt.colorbar()
    plt.show()

#%%

import numpy as np
data = np.loadtxt(r'C:\Users\lsf20\Downloads\image 1')
# Convert the list of lists to a NumPy array
image_array = np.array(data)


image_search = ImageSearchDFS(image_array)

# Find the locations of connected components
connected_components = image_search.find_connected_components()

# Set the coordinates of the connected components to 1 and all others to 0

# Plot the resulting image
image_search.plot_image()
plt.imshow(result_image, cmap='gray', interpolation='none')
plt.title('Resulting Image')
plt.colorbar()
plt.show()

    
# Set a threshold to convert the numerical values to binary (0 or 1)
threshold = 25000
binary_image = (image_array > threshold).astype(int)

# Plot the original image
plt.figure(figsize=(8, 6))
plt.imshow(image_array, cmap='hot', interpolation='none')
plt.title('Original Image')
plt.colorbar()
plt.show()

# Plot the binary image
plt.figure(figsize=(8, 6))
plt.imshow(binary_image, cmap='gray', interpolation='none')
plt.title('Binary Image (Thresholded)')
plt.colorbar()
plt.show()