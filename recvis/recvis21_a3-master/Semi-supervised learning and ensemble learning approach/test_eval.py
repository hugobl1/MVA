model.eval()
correct = 0
for data, target in val_loader:
    if use_cuda:
        data, target = data.cuda(), target.cuda()
    output = model(data)
    pred = output.data.max(1, keepdim=True)[1]
    correctpred= pred.eq(target.data.view_as(pred)).cpu().numpy()

    break

for i in range (64):
    if (correctpred[i]==False):
        print(pred[i].cpu().numpy())



