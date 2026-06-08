import os

folder_path = "genre/people/genre_people_images"
image_extensions = (".jpg", ".jpeg")

images = os.listdir(folder_path)

for idx, filename in enumerate(images,start=1):
    ext = os.path.splitext(filename)[1]
    new_filename = f"{idx:004}{ext}"
    src = os.path.join(folder_path,filename)
    dst = os.path.join(folder_path,new_filename)
    os.rename(src,dst)

print("Renaming complete")




