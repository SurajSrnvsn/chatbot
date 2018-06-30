#preprocesssing the data


import re
import time
import numpy as np
import tensorflow as tf

lines=open('movie_lines.txt',encoding='utf-8',errors='ignore').read().split('\n')
conversations=open('movie_conversations.txt',encoding='utf-8',errors='ignore').read().split('\n')

id2line={}
for line in lines:
    _line=line.split(' +++$+++ ')
    if len(_line)==5:
        id2line[_line[0]]=_line[4]

conv=[]
for conversation in conversations:
    _conv=conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conv.append(_conv.split(","))

questions=[]
answers=[]
for conversation in conv:
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])

def clean_text(text):
    text=text.lower()
    text=re.sub(r"i'm","i am",text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"/'ll", "will", text)
    text = re.sub(r"/'ve", "have", text)
    text = re.sub(r"/'re", "are", text)
    text = re.sub(r"/'d", "would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"[!@#$%^&*()_+={}|:<>?,.;']", "", text)
    return text

clean_ques=[]
for question in questions:
    clean_ques.append(clean_text(question))

clean_ans=[]
for answer in answers:
    clean_ans.append(clean_text(answer))

wordcountques={}
for question in clean_ques:
    for word in question.split():
        if word not in wordcountques:
            wordcountques[word]=1
        else:
            wordcountques[word]+=1

wordcountans={}
for answer in clean_ans:
    for word in answer.split():
        if word not in wordcountans:
            wordcountans[word]=1
        else:
            wordcountans[word]+=1

threshold=20
questionword2int={}
word_num=0
for word,count in wordcountques.items():
    if count >= threshold:
        questionword2int[word]=word_num
        word_num+=1

answerword2int={}
word_num=0
for word,count in wordcountans.items():
    if count >= threshold:
        answerword2int[word]=word_num
        word_num+=1

tokens=['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    questionword2int[token]=len(questionword2int)+1
for token in tokens:
    answerword2int[token]=len(answerword2int)+1
answerint2word={w_i:w for w,w_i in answerword2int.items()}
for i in range(len(clean_ans)):
    clean_ans[i]+=" <EOS>"

question_to_int=[]
for question in clean_ques:
    ints=[]
    for word in question.split():
        if word not in questionword2int:
            ints.append(questionword2int["<OUT>"])
        else:
            ints.append(questionword2int[word])
    question_to_int.append(ints)

ans_to_int=[]
for answer in clean_ans:
    ints=[]
    for word in answer.split():
        if word not in answerword2int:
            ints.append(answerword2int["<OUT>"])
        else:
            ints.append(answerword2int[word])
    ans_to_int.append(ints)

sorted_clean_ques=[]
sorted_clean_ans=[]
for length in range(1,26):
    for i in enumerate(question_to_int):
        if len(i[1])==length:
            sorted_clean_ques.append(question_to_int[i[0]])
            sorted_clean_ans.append(ans_to_int[i[0]])


#building the model


def model_input():
    inputs=tf.placeholder(tf.int32,[None, None],name='input')
    targets=tf.placeholder(tf.int32,[None,None],name='target')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32 , name='keep_prob')
    return inputs,targets,lr,keep_prob

def preprossing_targets(targets,word2int,batch_size):
    left_side=tf.fill([batch_size,1],word2int['<SOS>'])
    right_side=tf.strided_slice(targets,[0,0],[batch_size,-1],[1,1])
    preprossing_targets=tf.concat([left_side,right_side],1)
    return preprossing_targets

def encoder_rnn(rnn_inputs,rnn_size,num_layers,keep_prob,sequence_length):
    lstm=tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout=tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob=keep_prob)
    encoder_cell=tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
    _,encoder_state=tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,cell_bw=encoder_cell,inputs=rnn_inputs,sequence_length=sequence_length,dtype=tf.float32)
    return encoder_state

def decode_training_set(encoder_state,decoder_cell,decoder_embedded_input,sequence_length,decoding_scope,output_function,keep_prob,batch_size):
    attention_states=tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys,attention_values,attention_score_function,attention_construct_function=tf.contrib.seq2seq.prepare_attention(attention_states=attention_states,attention_option='bahdanau',num_units=decoder_cell.output_size)
    training_decoder_function=tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],attention_keys,attention_values,attention_score_function,attention_construct_function,name='attn_dec_train')
    decoder_output,decoder_final_state,decoder_final_context_state=tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,training_decoder_function,decoder_embedded_input,sequence_length,scope=decoding_scope)
    decoder_output_dropout=tf.nn.dropout(decoder_output,keep_prob)
    return output_function(decoder_output_dropout)

def decode_test_set(encoder_state,decoder_cell,decoder_embeddings_matrix,sos_id,eos_id,maximun_length,num_words,sequence_length,decoding_scope,output_function,keep_prob,batch_size):
    attention_states=tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys,attention_values,attention_score_function,attention_construct_function=tf.contrib.seq2seq.prepare_attention(attention_states,attention_option='bahdanau',num_units=decoder_cell.output_size)
    test_decoder_function=tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,encoder_state[0],attention_keys,attention_values,attention_score_function,attention_construct_function,decoder_embeddings_matrix,sos_id,eos_id,maximun_length,num_words,name='attn_dec_inf')
    test_predictions,decoder_final_state,decoder_final_context_state=tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,test_decoder_function,scope=decoding_scope)
    return test_predictions

def decoder_rnn(decoder_embedded_inputs,decoder_embedding_matrix,encoder_state,num_words,seq_length,rnn_size,num_layers,word2int,keep_prob,batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm=tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout=tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob=keep_prob)
        decoder_cell=tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
        weights=tf.truncated_normal_initializer(stddev=0.1)
        biases=tf.zeros_initializer()
        output_function=lambda x:tf.contrib.layers.fully_connected(x,num_words,None,scope=decoding_scope,weights_initializer=weights,biaser_initializer=biases)
        training_predictions=decode_training_set(encoder_state,decoder_cell,decoder_embedded_inputs,seq_length,decoding_scope,output_function,keep_prob,batch_size)
        decoding_scope.reuse_variables()
        test_preictions=decode_test_set(encoder_state,decoder_cell,decoder_embedding_matrix,word2int['<SOS>'],word2int['<EOS>'],seq_length-1,num_words,decoding_scope,output_function,keep_prob,batch_size)
        return training_predictions,test_preictions


def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answer_num_words, question_num_words,encoder_embedding_size, decoder_embeding_size, rnn_size, num_layers, questionword2int):
    encoder_embedded_input=tf.contrib.layers.embed_sequence(inputs,answer_num_words+1,encoder_embedding_size,initializer=tf.random_normal_initializer(0,1))
    encoder_state=encoder_rnn(encoder_embedded_input,rnn_size,num_layers,keep_prob,sequence_length)
    preprossed_targets=preprossing_targets(targets,questionword2int,batch_size)
    decoder_embeddings_matrix=tf.Variable(tf.random_uniform([question_num_words+1,decoder_embeding_size],0,1))
    decoder_embedded_input=tf.nn.embedding_lookup(decoder_embeddings_matrix,preprossed_targets)
    training_predictions,test_predictions=decoder_rnn(decoder_embedded_input,decoder_embeddings_matrix,encoder_state,question_num_words,sequence_length,rnn_size,num_layers,questionword2int,keep_prob,batch_size)
    return training_predictions,test_predictions


#training


epochs=100
batch_size=64
rnn_size=512
num_layers=3
encoding_embedings_size=512
decoding_embedings_size=512
learning_rate=0.01
learning_rate_decay=0.9
min_learning_rate=0.0001
keep_probability=0.5

tf.reset_default_graph()
session=tf.InteractiveSession()
inputs,targets,lr,keep_prob=model_input()
sequence_length=tf.placeholder_with_default(25,None,name='sequence_length')
input_shape=tf.shape(inputs)

training_predictions,test_predictions=seq2seq_model(tf.reverse(inputs,[-1]),targets,keep_prob,batch_size,sequence_length,len(answerword2int),len(questionword2int),encoding_embedings_size,decoding_embedings_size,rnn_size,num_layers,questionword2int)

with tf.name_scope("optimization"):
    loss_error=tf.contrib.seq2seq.sequence_loss(training_predictions,targets,tf.ones([input_shape[0],sequence_length]))
    optizimer=tf.train.AdamOptimizer(learning_rate)
    gradients=optizimer.compute_gradients(loss_error)
    clipped_gradiants=[(tf.clip_by_value(grad_tensor,-5.,5.),grad_variable)for grad_tensor,grad_variable in gradients if grad_tensor is not None]
    optizimer_gradiant_clipping=optizimer.apply_gradients(clipped_gradiants)

def apply_padding(batch_of_sequences,word2int):
    max_sequence_length=max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]

def split_into_batches(questions,answers,batch_size):
    for batch_index in range(0,len(questions)//batch_size):
        start_index=batch_index*batch_size
        questions_in_batch=questions[start_index:start_index+batch_size]
        answers_in_batch=answers[start_index:start_index+batch_size]
        padded_questions_in_batch=np.array(apply_padding(questions_in_batch,questionword2int))
        padded_answers_in_batch=np.array(apply_padding(answers_in_batch,answerword2int))
        yield padded_questions_in_batch,padded_answers_in_batch

training_validation_split=int(len(sorted_clean_ques)*0.15)
training_questions=sorted_clean_ques[training_validation_split:]
training_answers=sorted_clean_ans[training_validation_split:]
validation_questions=sorted_clean_ques[:training_validation_split]
validation_answers=sorted_clean_ans[:training_validation_split]

batch_index_check_training_loss=100
batch_index_check_validation_loss=((len(training_questions))//batch_size//2)-1
total_training_loss_error=0
list_validation_loss_error=[]
early_stepping_check=0
early_stepping_stop=1000
checkpoint="chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range (1,epochs+1):
    for batch_index,(padded_questions_in_batch,padded_answers_in_batch)in enumerate(split_into_batches(training_questions,training_answers,batch_size)):
        starting_time=time.time()
        _,batch_training_loss_error=session.run([optizimer_gradiant_clipping,loss_error],{inputs:padded_questions_in_batch,targets:padded_answers_in_batch,lr:learning_rate,sequence_length:padded_answers_in_batch.shape[1],keep_prob:keep_probability})
        total_training_loss_error+=batch_training_loss_error
        ending_time=time.time()
        batch_time=ending_time-starting_time
        if batch_index % batch_index_check_training_loss==0:
            print("Epoch:{:>3}/{},Batch:{:>4}/{},Training Loss Error:{:>6.3f},Training Time On 100 Batches:{:d} seconds".format(epoch,epochs,batch_index,len(training_questions//batch_size,total_training_loss_error//batch_index_check_training_loss,int(batch_time*batch_index_check_training_loss))))
            total_training_loss_error=0
        if batch_index % batch_index_check_validation_loss==0 and batch_index>0:
            total_validation_loss_error=0
            starting_time=time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions,validation_answers, batch_size)):
                _, batch_validation_loss_error = session.run(loss_error,{inputs: padded_questions_in_batch,targets: padded_answers_in_batch, lr: learning_rate,sequence_length: padded_answers_in_batch.shape[1],
                                                            keep_prob:1})
                total_validation_loss_error += batch_validation_loss_error
                ending_time = time.time()
                batch_time = ending_time - starting_time
                average_validation_loss_error=total_validation_loss_error/(len(validation_questions)/batch_size)
                print("The Validation Loss Error:{:>6.3f},Batch Validation Time:{:d}seconds".format(average_validation_loss_error,int(batch_time)))
                learning_rate*=learning_rate_decay
                if learning_rate<min_learning_rate:
                    learning_rate=min_learning_rate
                list_validation_loss_error.append(average_validation_loss_error)
                if average_validation_loss_error <= min(list_validation_loss_error):
                    print("I Speak Better Now!!!")
                    early_stepping_check=0
                    saver=tf.train.Saver()
                    saver.save(session,checkpoint)
                else:
                    print("Sorry i did not get better.")
                    early_stepping_check+=1
                    if early_stepping_check==early_stepping_stop:
                        break
        if early_stepping_check==early_stepping_stop:
            print("That is all that i can do :(")
            break
print("Training done")