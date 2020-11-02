class Graph(object):
	def __init__(self, entities, relations, scores, vocabs):
		"""
		Adapted from code for OneIE: http://blender.cs.illinois.edu/software/oneie/	
		:param entities (list): A list of	entity labels
		:param relations (list): A list of relations represented as tuples
		of (head index, tail index, relation label)
		:param vocabs (dict): Entity and relation type vocabularies 
		"""
		self.entities = entities
		self.relations = relations
		self.scores = scores
		self.vocabs = vocabs	
		self.entity_num	= len(self.entities)
		self.relation_num = len(self.relations)
		self.graph_local_score = 0.0		

	def add_relation(self, h_idx, t_idx, label, score):
		"""
		Add a relation edge to the graph
		:param h_idx: Index of the head node
		:param t_idx: Index of the tail node
		:param label: Index of the relation type label
		:param score (float): relation label score 
		"""
		if label:	
			self.relations.append((h_idx, t_idx, label))
			self.scores.append(score)
		self.relation_num = len(self.relations)
		self.graph_local_score += score

	@staticmethod
	def empty_graph(vocabs):
		"""
		Create an empty graph without nodes and edges
		:param vocabs (dict): Entity and relation type vocabularies
		"""
		return Graph([], [], [], vocabs)
	
	def to_label_idxs(self, max_relation_num):
		""" Generate label index lists to gather calculated scores
		:param: Max number of relations in the batch
		:return: Index and mask tensors
		"""	
		relation_idxs = [0]*max_relation_num
		mask_idxs = [0.0]*max_relation_num
		for i, (_, _, relation) in enumerate(self.relations):
			relation_idxs[i] = relation
			mask_idxs[i] = 1.0
		return relation_idxs, mask_idxs
