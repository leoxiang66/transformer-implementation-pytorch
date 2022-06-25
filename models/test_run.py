import torch
import torch.utils.data as data
from put_everthing_together import *
import torch.optim as optim

num_labels = 10

enc_input = torch.rand(2, 3, 768)
enc_att_mask = torch.tensor([[1,1,1],[1,1,0]])


dec_input = torch.rand(2, 5, 768)
dec_att_mask = torch.tensor([[1,1,0,0,0],[1,1,1,1,1]])

dec_output = torch.randint(low=0,high=num_labels,size=(2,5))


class MyDataSet(data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


loader = data.DataLoader(MyDataSet(enc_input, dec_input, dec_output), batch_size= 2, shuffle= True)

loss_fn = Loss()
trf = Transformer(6,768,num_labels)
optimizer = optim.SGD(trf.parameters(), lr=1e-4,momentum=0.99)
for epoch in range(1000):
    for enc_inputs, dec_inputs, labels in loader:
      '''
      enc_inputs: [batch_size, src_len,D]
      dec_inputs: [batch_size, tgt_len,D]
      labels: [batch_size, tgt_len]
      '''

      outputs = trf(enc_inputs, dec_inputs,enc_att_mask,dec_att_mask)
      optimizer.zero_grad()
      loss = loss_fn(outputs,labels)
      print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))


      loss.backward()
      optimizer.step()




