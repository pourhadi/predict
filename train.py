
# coding: utf-8

# In[35]:


import turicreate as tc

train_data = tc.SFrame.read_json('/Users/danielpourhadi/saved.json')

test_data = tc.SFrame.read_json('/Users/danielpourhadi/test_saved.json')
# Make a train-test split
# train_data, test_data = data.random_split(0.9)

# Automatically picks the right model based on your data.
model = tc.linear_regression.create(train_data, target='Target', max_iterations=1000, validation_set=test_data)

model.save('model.mlmodel')

# Save predictions to an SArray
predictions = model.predict(test_data)

for x in range(0, 50):
    print(str(test_data['Target'][x]) + ': ' + str(predictions[x]) + " - " + str(predictions[x] - test_data['Target'][x]))

# print(test_data['Target'])
# print(predictions)

# Evaluate the model and save the results into a dictionary
results = model.evaluate(test_data)

print(results)

