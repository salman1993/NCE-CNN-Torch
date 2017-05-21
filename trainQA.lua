--[[
  Training script for semantic relatedness prediction on the Twitter dataset.
  We Thank Kai Sheng Tai for providing the preprocessing/basis codes.
--]]

require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')

similarityMeasure = {}

include('util/read_data.lua')
include('util/Vocab.lua')
include('Conv.lua')
include('CsDis.lua')
include('metric.lua')
--include('PaddingReshape.lua')
printf = utils.printf

-- global paths (modify if desired)
similarityMeasure.data_dir        = 'data'
similarityMeasure.models_dir      = 'trained_models'
similarityMeasure.predictions_dir = 'predictions'

function header(s)
  print(string.rep('-', 80))
  print(s)
  print(string.rep('-', 80))
end

cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-dataset', 'TrecQA', 'dataset, can be TrecQA or WikiQA')
cmd:option('-version', 'raw', 'the version of TrecQA dataset, can be raw and clean')
cmd:text()
opt = cmd:parse(arg)

--read default arguments
local args = {
  model = 'conv', --convolutional neural network
  layers = 1, -- number of hidden layers in the fully-connected layer
  dim = 150, -- number of neurons in the hidden layer.
}

local model_name, model_class, model_structure
model_name = 'conv'
model_class = similarityMeasure.Conv
model_structure = model_name

--torch.seed()
torch.manualSeed(-3.0753778015266e+18)
print('<torch> using the automatic seed: ' .. torch.initialSeed())

if opt.dataset ~= 'TrecQA' and opt.dataset ~= 'WikiQA' and opt.dataset ~= 'kaggle' then
  print('Error dataset!')
  os.exit()
end
-- directory containing dataset files
local data_dir = 'data/' .. opt.dataset .. '/'

-- load vocab
local vocab = similarityMeasure.Vocab(data_dir .. 'vocab.txt')

-- load embeddings
print('loading word embeddings')

local emb_dir = 'data/glove/'
local emb_prefix = emb_dir .. 'glove.840B'
local emb_vocab, emb_vecs = similarityMeasure.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')

local emb_dim = emb_vecs:size(2)

-- use only vectors in vocabulary (not necessary, but gives faster training)
local num_unk = 0
local vecs = torch.Tensor(vocab.size, emb_dim)
for i = 1, vocab.size do
  local w = vocab:token(i)
  if emb_vocab:contains(w) then
    vecs[i] = emb_vecs[emb_vocab:index(w)]
  else
    num_unk = num_unk + 1
    vecs[i]:uniform(-0.05, 0.05)
  end
end
print('unk count = ' .. num_unk)
emb_vocab = nil
emb_vecs = nil
collectgarbage()
local taskD = 'qa'
-- load datasets
print('loading dataset: ' .. opt.dataset)
if opt.dataset == 'TrecQA' then
  train_dir = data_dir .. 'train-all/'
  dev_dir = data_dir .. opt.version .. '-dev/'
  test_dir = data_dir .. opt.version .. '-test/'
elseif opt.dataset == 'WikiQA' then
  train_dir = data_dir .. 'train/'
  dev_dir = data_dir .. 'dev/'
  test_dir = data_dir .. 'test/'
elseif opt.dataset == 'kaggle' then
  train_dir = data_dir .. 'train/'
  dev_dir = data_dir .. 'dev/'
  test_dir = data_dir .. 'test/'
end

-- adding unknown token to vocab
-- vocab:add_unk_token()

local train_dataset = similarityMeasure.read_relatedness_dataset(train_dir, vocab, taskD)
local dev_dataset = similarityMeasure.read_relatedness_dataset(dev_dir, vocab, taskD)
local test_dataset = similarityMeasure.read_relatedness_dataset(test_dir, vocab, taskD)
printf('train_dir: %s, num train = %d\n', train_dir, train_dataset.size)
printf('dev_dir: %s, num dev   = %d\n', dev_dir, dev_dataset.size)
printf('test_dir: %s, num test  = %d\n', test_dir, test_dataset.size)

-- initialize model
local model = model_class{
  emb_vecs   = vecs,
  structure  = model_structure,
  num_layers = args.layers,
  mem_dim    = args.dim,
  task       = taskD,
}

-- number of epochs to train
local num_epochs = 20

-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()


if lfs.attributes(similarityMeasure.predictions_dir) == nil then
  lfs.mkdir(similarityMeasure.predictions_dir)
end

-- train
local train_start = sys.clock()
local best_dev_score = -1.0
local best_dev_model = model

-- threads
--torch.setnumthreads(4)
--print('<torch> number of threads in used: ' .. torch.getnumthreads())

header('Training model')

local id = 2011
print("Id: " .. id)
for i = 1, num_epochs do
  local start = sys.clock()
  print('--------------- EPOCH ' .. i .. '--- -------------')
  model:trainCombineOnly(train_dataset)
  print('Finished epoch in ' .. ( sys.clock() - start) )

  local dev_predictions = model:predict_dataset(dev_dataset)
  local dev_log_loss = log_loss(dev_predictions, dev_dataset.labels)
  local dev_accuracy = accuracy(dev_predictions, dev_dataset.labels)
  printf('-- dev log loss: %.5f, accuracy: %.5f\n', dev_log_loss, dev_accuracy)

  -- keep track of least log loss and best dev model??

 -- evaluate test set and save predictions
    local test_predictions = model:predict_dataset(test_dataset)
    local predictions_save_path = string.format(
	        similarityMeasure.predictions_dir .. '/results-%s.%dl.%dd.epoch-%d.%.5f.%d.pred', args.model, args.layers, args.dim, i, dev_log_loss, id)
    local predictions_file = torch.DiskFile(predictions_save_path, 'w')
    print('writing predictions to ' .. predictions_save_path)
    for i = 1, test_predictions:size(1) do
      predictions_file:writeFloat(test_predictions[i])
    end
    predictions_file:close()
end
print('finished training in ' .. (sys.clock() - train_start))
