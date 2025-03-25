import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import cv2

def generate_data():
    X1 = np.zeros((30, 3))
    X2 = np.zeros((30, 3))
    for k in range(9, 30):
        X1[k, np.random.randint(3)] = 1
        X2[k, np.random.randint(3)] = 1
    y = (np.any(X1 > 0.5, axis=1)).astype(int)
    return X1, X2, y

def cc_coeff(x, y):
    sxy = np.sum((x - x.mean()) * (y - y.mean())) / x.shape[0]
    return 2 * sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)

def compute_ccc(X1, X2):
    return np.array([cc_coeff(X1[:, i], X2[:, i]) for i in range(3)])

def compute_auc(X_train, X_test, y_train, y_test):
    model = LogisticRegression().fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


def visualize_data(X1, X2, ccc, auc):
    cell_size = 48
    padding = 400
    grid_width = 5
    margin = 100
    width = 6 * cell_size + padding + grid_width + 2 * margin
    height = 30 * cell_size + grid_width + 2 * margin
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    elephant = cv2.imread("./data/elephant.png")
    elephant = cv2.resize(elephant, (cell_size, cell_size))
    perm = np.random.permutation(30)
    X1 = X1[perm]
    X2 = X2[perm]
    start_x1 = margin
    start_x2 = margin + 3 * cell_size + padding + grid_width
    start_y = margin
    for i in range(30):
        for j in range(3):
            if X1[i, j] == 1:
                x = start_x1 + j * cell_size + grid_width // 2
                y = start_y + i * cell_size + grid_width // 2
                img[y:y+cell_size, x:x+cell_size] = elephant
            if X2[i, j] == 1:
                x = start_x2 + j * cell_size + grid_width // 2
                y = start_y + i * cell_size + grid_width // 2
                img[y:y+cell_size, x:x+cell_size] = elephant
    for i in range(31):
        cv2.line(img, (start_x1, start_y + i * cell_size), (start_x1 + 3 * cell_size, start_y + i * cell_size), (0, 0, 0), grid_width)
        cv2.line(img, (start_x2, start_y + i * cell_size), (start_x2 + 3 * cell_size, start_y + i * cell_size), (0, 0, 0), grid_width)
    for j in range(4):
        cv2.line(img, (start_x1 + j * cell_size, start_y), (start_x1 + j * cell_size, start_y + 30 * cell_size), (0, 0, 0), grid_width)
        cv2.line(img, (start_x2 + j * cell_size, start_y), (start_x2 + j * cell_size, start_y + 30 * cell_size), (0, 0, 0), grid_width)

    # Draw arrow and add text
    # arrow_start = (start_x1 + 3 * cell_size + grid_width, start_y + 15 * cell_size)
    # arrow_end = (start_x2, start_y + 15 * cell_size)
    # cv2.arrowedLine(img, arrow_start, arrow_end, (0, 0, 0), 2)

    # Add text with CCC and AUC
    ccc_text = f"CCC: {ccc[0]:.2f}%  {ccc[1]:.2f}%  {ccc[2]:.2f}%"
    auc_text = f"AUC: {auc:.2f}"

    # cv2.putText(img, ccc_text, (start_x1 + 3 * cell_size + padding, start_y + 14 * cell_size), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    # cv2.putText(img, auc_text, (start_x1 + 3 * cell_size + padding, start_y + 16 * cell_size), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite("./results/elephant_house.png", img)


if __name__ == '__main__':
    np.random.seed(42)
    X1, X2, y = generate_data()

    # Stratified split for train/test data
    X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, stratify=y, random_state=42)

    # Compute CCC and AUC
    ccc_values = compute_ccc(X1, X2)
    auc_value = compute_auc(X_train, X_test, y_train, y_test)

    print("CCC:", ccc_values)
    print("AUC:", auc_value)

    # Visualize data with the text and arrows
    visualize_data(X1, X2, ccc_values * 100, auc_value)
