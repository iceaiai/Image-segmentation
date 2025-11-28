# Author: Mengkun Tian
# Script description:
# Convert the image from dm3,dm4, tif, png, jpg, jpeg to png (tif, jpg and jpeg are also allowed) with designated output format

import hyperspy.api as hs
import numpy as np
import tkinter as tk
from tkinter import filedialog,messagebox
import secrets
from PIL import Image
from pathlib import Path
import shutil

class image_converter:
    def __init__(self,set_inputdir=True,set_outputdir = True, output_format = 'png', image_resolution = (512,512)):
        """
        set_inputdir: True  -> user chooses input directory via dialog
                     False -> use current directory ('.') as input
        set_outputdir: True  -> user chooses output directory via dialog / overwrite prompt
                      False -> auto: <input_parent>/converted_images/<input_dir_name>
        """
        self.supported_inputformat = ('dm3','dm4','tif','png','jpg','jpeg')
        self.supported_outputformat = ('tif','png','jpg','jpeg')
        self.output_format = output_format.lower()
        if self.output_format not in self.supported_outputformat:
            raise ValueError(f"Output format {self.output_format} is not supported. Supported formats include 'tif','png','jpg','jpeg.'")
        self.image_resolution = image_resolution
        self.set_inputdir = set_inputdir
        self.set_outputdir = set_outputdir
        self.input_path: Path | None = None   # directory input
        self.output_path: Path | None = None  # directory output
        self.input_data= None # input_data and a flag. True if input_data is given. Output image will created at the same directory as the input data ending with '_resized'.


    def _set_input_dir(self) -> Path:
        root = tk.Tk()
        root.withdraw()
        dir = filedialog.askdirectory(title = 'Please choose the input directory.',initialdir = '.')
        root.destroy()
        # if no directory selected
        if not dir:
            raise RuntimeError('Please select the input directory.')
        return Path(dir).resolve()

    def _set_output_dir(self)-> Path:
        """
        Working directory is the input directory
        Default_dir is the working directory's parent / "converted_images"/ working directory
        If default directory not exist, system use the current directory's parent/ "converted_images" as output directory/current directory name.
        If default directory exist, ask user decide 1. ok: override the old existing directory; 2. no: choose another directory
        """
        if self.input_path is None:
            print('Input path is not selected. Automatically select the current directory as input.')
            self.input_path = Path('.')
        parent_dir = self.input_path.parent
        current_dirname = self.input_path.name
        default_dir = parent_dir / "converted_images"/current_dirname
        if default_dir.exists() and any(default_dir.iterdir()): 
            root =tk.Tk()
            root.withdraw()
            ans = messagebox.askyesno(
                title = "Output folder exists", 
                message = f"The folder:\n{default_dir}\n"
                f"already exists and is not empty.\n\n"
                f"Click 'Yes' to overwrite / reuse it,\n"
                f"or 'No' to choose a different output folder."
            )
            if ans:
                shutil.rmtree(default_dir)#https://docs.python.org/3/library/shutil.html
                default_dir.mkdir(parents=True,exist_ok = True)#https://docs.python.org/3/library/pathlib.html
                dir = default_dir
            else:
                dir = filedialog.askdirectory(initialdir='.',title = "Please choose another folder as output directory")
            root.destroy()
        else:
            default_dir.mkdir(parents=True,exist_ok = True)
            dir = default_dir
        return Path(dir).resolve()
    
    def image_loader(self):
        """
        Images can be loaded as single, multiple images in one folder, or in a directory or a direcoty with multiple subdirectories;
        User needs to select the input path;
        1. If the image(s) directly is(are) loaded by hyperspy, meaning input_data is not None, the output path will be directly selected to the current directory. 
        2. If a directory is selected, the output directory will create a folder called converted_images. 
        The file structures are given below at different sceinarios:
        1. input_data is not None:
            input dir: 
            dir_A:
               data1, data2, data3....dataN
            output dir:
            dir_A (same directory as input, no new directory is created):
               data1, data2, data3....dataN, data1_resized, data2_resized, data3_resized....dataN_resized
                where data1_resized, data2_resized, data3_resized....dataN_resized are the images after resized and transformed
        2. input_data is None:
            a. if a single directory is selected:
                input dir:
                dir_A(sub_folder)
                    data1, data2, data3....dataM
                output dir:
                dir_A(sub_folder)
                    data1, data2, data3....dataM
                converted_image:
                    dir_A(sub_folder)
                        data1, data2, data3....dataM

            b. if a directory with multiple subdirectory is selected:
                input dir:
                dir_A (parent,user selected) 
                    dir_B(sub_folder)
                        data1, data2, data3....dataM
                    dir_C (sub folders)
                        data1, data2, data3....dataN
                output dir:
                dir_A (parent) 
                    dir_B(sub_folder)
                        data1, data2, data3....dataM
                    dir_C (sub folders)
                        data1, data2, data3....dataN
                converted_image:
                    dir_A (parent) 
                        dir_B(sub_folder,resized and transformed)
                            data1, data2, data3....dataM
                        dir_C (sub_folder,resized and transformed)
                            data1, data2, data3....dataN
        """
        # image contain is a list that has the data structure (path,hs_image_object)
        image_container: list[tuple[Path,object]] = []

        if self.set_inputdir:
            self.input_path = self._set_input_dir()
        else:
            self.input_path = Path('.').resolve()
        
        if self.input_data is not None: #single or multple data are loaded, no output path need to be assigned
            s = hs.load(self.input_data)
            #handle if multiple images are loaded:
            if len(s) ==1:
                filename = Path(s.metadata.General.original_filename).stem
                p = self.input_path/ filename
                image_container.append((p,s))
            else:
                for i in range(len(s)):
                    s_sub = s[i]
                    filename = Path(s_sub.metadata.General.original_filename).stem
                    p = self.input_path/ filename
                    image_container.append((p,s_sub))
        else: # direcotry is selected, output path is designated
            #First, set the output directory;
            if self.set_outputdir:
                self.output_path = self._set_output_dir()
            else:
                parent_dir = self.input_path.parent
                current_dirname = self.input_path.name
                default_dir = parent_dir / "converted_images" / current_dirname
                default_dir.mkdir(parents=True, exist_ok=True)
                self.output_path = default_dir
            #Second, append the (file, signal)
            #for single directory with no subdirectory, use is_file to check
            for f in self.input_path.iterdir():
                if f.is_file() and f.suffix.lower().lstrip(".") in self.supported_inputformat: 
                    print('f = ',f)
                    s = hs.load(f)
                    image_container.append((f,s))
                elif f.is_dir():
                    for sub in f.iterdir():
                        if sub.is_file() and sub.suffix.lower().lstrip(".") in self.supported_inputformat: 
                            s = hs.load(sub)
                            image_container.append((sub,s))
        return image_container
    
    def convert_to_image(self,input_data=None):
        self.input_data = input_data
        image_container = self.image_loader()
        for i in image_container:
            path, signal = i
            data = np.array(signal.data)
            fields =data.dtype.fields
            #Handle the r,g,b cases
            if fields is not None:
                if all (c in fields for c in ('r','g','b')):
                    r = data['R'].astype(float)
                    g = data['G'].astype(float)
                    b = data['B'].astype(float)
                    data = 0.299 * r + 0.587 * g + 0.114 * b
            data =data.astype(float)
            #normalized data
            data_min = data.min()
            data_max = data.max()
            data -=data_min
            if data_max>0:
                data /=data_max
            data = (255*data).astype(np.uint8)
            #resize
            img = Image.fromarray(data)
            img = img.resize(size=self.image_resolution,resample=Image.BILINEAR)
            if self.input_data is not None:
                dest_dir = path.parent
                filename = path.stem
                name = f'{filename}_resized'
            else:
                #single directory case, path is the file, path.parent is the directory
                if path.parent == self.input_path:
                    dest_dir = self.output_path
                # directory with sub case, path is the file, path.parent is the directory
                else:
                    foldername = path.parent.name
                    dest_dir = self.output_path/foldername
                dest_dir.mkdir(parents=True,exist_ok=True)
                name = secrets.token_hex(5)
            out_path = dest_dir/f'{name}.{self.output_format}'
            img.save(out_path)
            print(f"[OK] {path} -> {out_path}")    

"""
Example use:
image = image_converter()
input_data = '*.jpg'
image.convert_to_image(input_data)
"""