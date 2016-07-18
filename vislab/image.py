import skimage


def get_image_for_filename(image_filename):
    if image_filename is not None:
        image = skimage.io.imread(image_filename)
        return image
    else:
        return None


def image2jpg(src_path, dst_parh):
    import subprocess
    import shlex
    cmd = 'convert {0} {0}'.format(src_path, dst_parh)
    print cmd
    p = subprocess.Popen(
        shlex.split(cmd))
    p.wait()
