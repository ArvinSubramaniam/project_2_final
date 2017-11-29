import shutil

os.chdir ('test_set_images')

for i in range(1,50+1):
    shutil.copy('test_' + str(i) + '/test_' + str(i) + '.png', "satImage_%.3d.png" % i)