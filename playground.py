

def train(train_data, batch_size,
          preprocessing_function, loss_function,
          learning_rate, n_epochs = 1, deviceid = -1):

    model = NN.SiameseNetwork()

    if deviceid > -1:
        model.cuda()

    optimzer = optim.Adam(model.parameters(), lr= learning_rate)

   updates = 0
   loss_history = []
   best_dev_pearson = -1.0
   best_epoch = -1

   for epoch in range(n_epochs):

       print(f"EPOCH {epoch}")
       loss = train_epoch( model, train_data,
                           loss_function, optimzer,
                           batch_size, epoch, deviceid)
       loss_history.append(loss)

       #EVAL HERE

def train_epoch( model, train_data,
                 loss_function, optimzer,
                 batch_size, epoch, deviceid):
    model.train()
    avg

    avg_loss = 0.0
    avg_acc = 0.0
    total = 0
    total_acc = 0.0
    total_loss = 0.0


    data_loader = train_data.get_loader(shuf= True, batch_size = batch_size)

    for iter, data in enumerate(data_loader):
        rec0, rec1, lbl = data

        if deviceid > -1:
            rec0 = autograd.(rec0.cud)
            rec1 = autograd.Variable(rec1.cuda())
        model.zero_grad()
        model.batch_size = len(train_labels)
        model.hidden = model.init_hidden()
        output = model(rec0.t()
