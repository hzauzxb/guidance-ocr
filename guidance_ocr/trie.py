class TrieNode:
    def __init__(self, char):
        self.children = {}
        self.char = char
        self.is_end_of_word = False

# 基于字典树维护一个可用节点的集合，表示当前状态能够输出的所有字
# 每过来一个token都将token转换为词，然后遍历这个词中的所有字，看当前可用的节点，是否存在可用的节点
# 若有可用节点，循环往复直到该词遍历完成，若无可用节点，返回失败，并退出
class Trie:
    def __init__(self):
        self.root = TrieNode(None)

    def insert(self, word):
        node = self.root
        for char_id, char in enumerate(word):
            if char not in node.children:
                node.children[char] = TrieNode(char)
            node = node.children[char]
        
        node.is_end_of_word = True

    def get_init_avinodes(self, root_node):
        if len(root_node.children) == 0:
            return [root_node]
        else:
            avi_nodes = [root_node]
            for children_char in root_node.children.keys():
                avi_nodes += self.get_init_avinodes(root_node.children[children_char])
            return avi_nodes

    def get_head_nodes(self):
        head_nodes = []
        for key in self.root.children.keys():
            head_nodes.append(self.root.children[key])
        return head_nodes