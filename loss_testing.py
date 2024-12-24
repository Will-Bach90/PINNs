import matplotlib.pyplot as plt

epochs = []
losses = []
with open("loss.txt", "r") as f:
    for line in f:
        e, l = line.split()
        epochs.append(int(e))
        losses.append(float(l))

plt.plot(epochs, losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()


test_x = []
test_true = []
test_pred = []
with open("predictions_complex.txt", "r") as f:
    for line in f:
        if line.startswith("#"):
            continue
        xv, tv, pv = line.split()
        test_x.append(float(xv))
        test_true.append(float(tv))
        test_pred.append(float(pv))

plt.scatter(test_x, test_true, color='blue', label='True Values')
plt.scatter(test_x, test_pred, color='red', label='Predictions')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Test Set Predictions vs True (Sine + Noise)')
plt.legend()
plt.show()

# train_x = []
# train_true = []
# train_pred = []
# with open("training_complex.txt", "r") as f:
#     for line in f:
#         if line.startswith("#"):
#             continue
#         xv, tv, pv = line.split()
#         train_x.append(float(xv))
#         train_true.append(float(tv))
#         train_pred.append(float(pv))

# plt.scatter(train_x, train_true, color='blue', label='True Values')
# plt.scatter(train_x, train_pred, color='red', label='Predictions')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Train Set Predictions vs True (Sine + Noise)')
# plt.legend()
# plt.show()
