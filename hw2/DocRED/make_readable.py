import json

def make_readable(result_file, rel_info_file):
	results = json.load(open(result_file, 'r'))
	rel_info = json.load(open(rel_info_file, 'r'))
	with open('{}_readable.txt'.format(result_file), 'w') as readable_f:
		for res in results:
			vertexSet = res['vertexSet']
			for sent in res['sents']:
				readable_f.write(' '.join(sent))
				readable_f.write('\n')
		
			for label in res['labels']:
				readable_f.write('{} {} {}\n'.format(vertexSet[label['h']][0]['name'], vertexSet[label['t']][0]['name'], rel_info[label['r']]))

			readable_f.write('\n')

if __name__ == '__main__':
	make_readable('fig_result/10_30_2020/dev_dev_bert_index_converted.json', 'fig_result/10_28_2020/rel_info.json')
	make_readable('fig_result/10_28_2020/dev.json', 'fig_result/10_28_2020/rel_info.json')
	make_readable('fig_result/10_28_2020/dev_dev_index_converted_no_natripe.json', 'fig_result/10_28_2020/rel_info.json')
	make_readable('fig_result/10_28_2020/dev_dev_gcn_gcn_index_converted.json', 'fig_result/10_28_2020/rel_info.json')
