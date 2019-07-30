Pytorch implementation 
This code is based on the [Exploit the Unknown Gradually: One-Shot Video-Based Person Re-Identification by Stepwise Learning](https://github.com/Yu-Wu/Exploit-Unknown-Gradually)

## Preparation
### Dependencies
- Python 3.6
- PyTorch (version >= 0.2.0)
- h5py, scikit-learn, metric-learn, tqdm

### Download datasets 
- MARS: [[Google Drive]](https://drive.google.com/open?id=1m6yLgtQdhb6pLCcb6_m7sj0LLBRvkDW0)   [[BaiduYun]](https://pan.baidu.com/s/1mByTdvXFsmobXOXBEkIWFw).
- Move the downloaded zip files to `./data/` and unzip here.


## Train

For the MARS datasaet:
```
python run.py --dataset mars --logs_dir logs/mars_EF_20/ --EF 20 --mode Dissimilarity --max_frames 256

```

## My Improvement
```
def get_Dissimilarity_result_rank(self):

        # extract feature
        u_feas = self.get_feature(self.u_data)
        l_feas = self.get_feature(self.l_data)
        print("u_features", u_feas.shape, "l_features", l_feas.shape)
        u_l_dist = np.dot(u_feas,np.transpose(l_feas))
        u_u_dist = np.dot(u_feas,np.transpose(u_feas))
        l_l_dist = np.dot(l_feas,np.transpose(l_feas))
        dist = re_ranking(u_l_dist,u_u_dist,l_l_dist) # distance rerank  (https://github.com/zhunzhong07/person-re-ranking)


        scores = np.zeros((u_feas.shape[0]))
        labels = np.zeros((u_feas.shape[0]))

        num_correct_pred = 0
        for idx, u_fea in enumerate(dist):
            index_min = np.argmin(u_fea)
            scores[idx] = - u_fea[index_min]  # "- dist" : more dist means less score
            labels[idx] = self.l_label[index_min]  # take the nearest labled neighbor as the prediction label

            # count the correct number of Nearest Neighbor prediction
            if self.u_label[idx] == labels[idx]:
                num_correct_pred += 1
        print("{} predictions on all the unlabeled data: {} of {} is correct, accuracy = {:0.3f}".format(
            self.mode, num_correct_pred, u_feas.shape[0], num_correct_pred/u_feas.shape[0]))
        return labels, scores

```

## origin

```
 def get_Dissimilarity_result(self):

        # extract feature 
        u_feas = self.get_feature(self.u_data)
        l_feas = self.get_feature(self.l_data)
        print("u_features", u_feas.shape, "l_features", l_feas.shape)

        scores = np.zeros((u_feas.shape[0]))
        labels = np.zeros((u_feas.shape[0]))

        num_correct_pred = 0
        for idx, u_fea in enumerate(u_feas):
            diffs = l_feas - u_fea
            dist = np.linalg.norm(diffs,axis=1)
            index_min = np.argmin(dist)
            scores[idx] = - dist[index_min]  # "- dist" : more dist means less score
            labels[idx] = self.l_label[index_min] # take the nearest labled neighbor as the prediction label

            # count the correct number of Nearest Neighbor prediction
            if self.u_label[idx] == labels[idx]:
                num_correct_pred +=1

        print("{} predictions on all the unlabeled data: {} of {} is correct, accuracy = {:0.3f}".format(
            self.mode, num_correct_pred, u_feas.shape[0], num_correct_pred/u_feas.shape[0]))
        return labels, scores

```


