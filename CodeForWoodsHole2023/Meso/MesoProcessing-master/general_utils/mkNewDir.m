function mkNewDir(pth)

if ~isfolder(pth)
    mkdir(pth);
end