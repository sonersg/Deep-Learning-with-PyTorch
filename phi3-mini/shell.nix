{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

pkgs.mkShell {
  packages = with pkgs; [
    python313
    python313Packages.pip
    python313Packages.torch # auto includes CUDA if available
    python313Packages.torchvision
    python313Packages.matplotlib
    python313Packages.pandas
    python313Packages.scikit-learn
    python313Packages.nltk
    python313Packages.numpy
    python313Packages.transformers
    python313Packages.accelerate
    python313Packages.huggingface-hub
    #cudaPackages.cudatoolkit
  ];
}
