# import xml.etree.ElementTree as ET
import os
import glob
import xml.dom.minidom

path = r'D:\pythonProject\text_out\data_try\try1206\dataset_1206\box_label'

for xml_name in os.listdir(path):
    read_xml = os.path.join(path, xml_name)
    tree = xml.dom.minidom.parse(read_xml)
    root = tree.documentElement

    change1_folder = root.getElementsByTagName('folder')
    change2_filename = root.getElementsByTagName('filename')
    change3_path = root.getElementsByTagName('path')

    change1_folder[0].firstChild.data = path.split('\\')[-1]
    change3_path[0].firstChild.data = os.path.join(path, change2_filename[0].firstChild.data)
    print(change1_folder)
    print(change2_filename)
    print(change3_path)
    with open(read_xml, 'w') as fh:
        tree.writexml(fh)
        print('写入name/pose OK!')


#     for obj in root.iter('annotation'):
#         change1_folder = obj.find('folder').text
#         change2_filename = obj.find('filename').text
#         change3_path = obj.find('path').text
#         print(change1_folder)
#         print(change2_filename)
#         print(change3_path)
#
#         change1_folder = path.split('\\')[-1]
#         change3_path = os.path.join(path, obj.find('filename').text)
#
#         tree.write(os.path.join(path, xml_name))
#
#
# for xml_name in os.listdir(path):
#     read_xml = os.path.join(path, xml_name)
#     tree = ET.parse(read_xml)
#     root = tree.getroot()
#     for obj in root.iter('annotation'):
#         change1_folder = obj.find('folder').text
#         change2_filename = obj.find('filename').text
#         change3_path = obj.find('path').text
#         print(change1_folder)
#         print(change2_filename)
#         print(change3_path)