
def evalutaion(actual_list,prediction_list):
	result = 0.0
	num = len(actual_list)
	for i in range(num):
		if prediction_list[i] + actual_list[i] != 0:
		    result += abs((prediction_list[i] - actual_list[i])/float(prediction_list[i] + actual_list[i]))
	return result / num
