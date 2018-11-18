import re
def clean_str(string):
	string = re.sub(r"\\", "", string)
	string = re.sub(r"\'", "", string)
	string = re.sub(r"\"", "", string)
	return string.strip().lower()

data_train = pd.read_csv('labeledTrainData.csv', sep='\t')
print(data_train.shape)

texts = []
labels = []

for idx in range(data_train.review.shape[0]):
	text = BeautifulSoup(data_train.review[idx], "lxml")
	texts.append(clean_str(text.get_text().encode('ascii','ignore').decode('utf-8')))
	labels.append(data_train.seniment[idx])
