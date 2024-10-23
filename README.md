# Col3D-MTL

Col3D-MTL is a CNN-based model that jointly estimates monocular depth and surface normal maps in colonoscopy frames. The framework consists of a shared encoder, θ, followed by an atrous spatial pyramid (ASPP) module to extract contextual information at different dilation rates. The decoding stage comprises a primary depth estimator decoder (bottom) and an auxiliary surface normal decoder (top). CBAM modules are introduced at the skip connections and after the convolutional layers of the depth decoder to enhance global context awareness. To explicitly enforce consistency among tasks, a depth-to-surface normal (D2SN) module receives the predicted depth and outputs a warped surface normal map. A2MIM is used to pre-train the encoder on phantom and patient colonoscopy data following a self-supervised learning approach based on masked image modelling.

![image](https://github.com/user-attachments/assets/481d77e4-ace5-4065-9309-4ca2acca5b48)

# Datasets

Download the [C3VD](https://durrlab.github.io/C3VD/), [CVC-ColonDB](https://polyp.grand-challenge.org/Databases/), and [PolypGen](https://www.synapse.org/Synapse:syn26376615/wiki/613312) and save it into the ./data folder.

# Evaluation

`
python eval.py --model_name Col3D-MTL --checkpoint_path ./logs/Col3D-MTL/model_best_d1 --data_path_eval ./data --gt_path_eval ./data 
         --filenames_file_eval ./data_splits/test.txt --multitask True --CL True
`


