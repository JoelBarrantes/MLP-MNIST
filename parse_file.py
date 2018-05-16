
import re

def file_to_mat(filename):
    f = open(filename,"r")
    content = f.read()
    result = re.findall('(?<=Loss ).*\n.*', content)
    matrix_builder = "["
    i = 1
    for item in result:
        item = item.replace("\nTest set accuracy:  ", ",")
        matrix_builder += str(i)+","+item+";\n"
        i+=1
    matrix_builder+="]"
    return (matrix_builder)
print(file_to_mat("dropout_25.txt"))


