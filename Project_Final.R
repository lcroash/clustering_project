# Customer Segmentation using The Instacart Online Grocery Shopping Dataset 2017

# The instacart dataset was developed for a kaggle competition to develop a predictive market basket analysis model.
# For the purposes of this project, I will be evaluating various clustering techniques to define a set of customer types.
# It is my hypothesis that a more fractured market basket analysis will provide association rules with more substantial
# support for inter-cluster groups that for the customer base as a whole.


### Loading the necessary libraries:

library(tidyverse)
library(tibble)
library(stringr)
library(cluster)
library(factoextra)
library(NbClust)
library(clustree)
library(FactoMineR)
library(corrplot)
library(kableExtra)
library(wordcloud)

### Begin by loading each of the supplied csv files
setwd("E:/College/Project/data")

aisles = read.csv("aisles.csv", header = T)
departments = read.csv("departments.csv", header = T)
order_prior = read.csv("order_products__prior.csv", header = T)
order_train = read.csv("order_products__train.csv", header = T)
orders = read.csv("orders.csv", header = T)
products = read.csv("products.csv", header = T)

### Aisle data
# aisle consists of 134 entries of aisle id and name.
# aisle could be important to differentiate between customer types:
# Possible features:
#   Top 5 aisles
  
head(aisles, 5)
glimpse(aisles)
summary(aisles)

### Department data
# departments consists of 21 entries of department id and name
# department could be important to defining customers similarly to aisle:
# Possible features:
#   Top 5 Departments
head(departments, 5)
glimpse(departments)
summary(departments)

### Prior order data
# - order_prior consists 32 millions entries corresponding the orders placed (at sku level) before the users most recent order:
#   - order_id: the id of the order
# - product_id: the id of the product
# - add_to_cart_order: the order that the product was added to the basket
# - reorder: whether the product has previously been ordered by this customer - 1 if ordered previously, 0 otherwise.
# 
# - add_to_cart_order could tell us about product affinity or importance of products to the customer.
# - do they shop staples first, is there a shopping path used by customers.
# - reorder: is the customer likely to try new products or do they consistently purchase the same items, if so, how regularly?
# Possible features:
#   New product preference - probability of buying products never purchased before 
head(order_prior, 5)
glimpse(order_prior)
summary(order_prior)
  
### Training order data
# Similar to order_prior
head(order_train, 5)
glimpse(order_train)
summary(order_train)

### Orders data
# - orders consists of 3.4M entries:
#   - order_id: the id of the order
# - user_id: the id of the user
# - eval_set: 3 possible values for data type: 
#   - prior: orders prior to the users most recent order
# - train: training data for kaggle competition
# - test: test data held back for kaggle competition
# - order_number: the order sequence number for this customer (1 = 1st, n = nth)
# - order_dow: order day of the week. This is not explicitly defined so we will make an assumption based on EDA.
# - order_hour_of_day: the hour of the day when this order was placed
# - days_since_prior_order: days since their last order
head(orders, 5)
glimpse(orders)
summary(orders)
  
### Product data
# - products consists of 50k product lines:
#   - product_id: the id of the product
# - product_name: descriptive name of the product
# - aisle_id: the aisle id
# - department_id: department id
# Possible features:
#   Dietary preferences - preference for organic / vegetarian / gluten free / dairy free
head(products, 5)
glimpse(products)
summary(products)

# ## Initial Intuition #########
# 
# - Product level information will not be useful to isolate macro level clusters of customers.
# - A more general high level approach should be adopted for transactional data i.e. aisle or department level.
# - Training and test sets should be drawn from the complete orders set rather than using the train, test, prior markers in the orders data since we are looking at customer data. Then we split the based on customer id.
# - The split was to allow for training of association rules which is irrelevant for this stage of development.

# ### Clustering can be performed using a variety of methods.
# Research on clustering tends to take the same approach, RFM to categorise but transaction behaviour
# then additionally with behavioural attribute such as department affinity and demographics.
# RFM: calculate the recency, monetary and frequency characteristics of customers then cluster using KMEANS
# - Department Affinity: calculate the frequency of orders within departments, average transaction values, customer value and department preferences in absolute terms.
# - Classify customers based on when they shop, weekday and time. 
# - Classify customers based on dietary preferences
 
# ### Build out the products data 
# Tag products with dietary preferences by specifying whether or not a product contain the text:
# - Organic
# - Gluten Free
# - Vegetarian
# - Dairy Free

wordcloud(products$product_name, min.freq=100, scale=c(5,1), 
          random.color=T,max.words = 30, random.order=F)


products = products%>%
  mutate(is_organic = ifelse(str_detect(str_to_lower(products$product_name),'organic'),1,0))

products = products%>%
  mutate(is_gluten_free = ifelse(str_detect(str_to_lower(products$product_name),'gluten'), 1,0))

products = products%>%
  mutate(is_vegetarian = ifelse(str_detect(str_to_lower(products$product_name),'vegetarian'), 1,0))

products = products%>%
  mutate(is_dairy_free = ifelse(str_detect(str_to_lower(products$product_name),'dairy free'), 1,0))

products = products%>%
  mutate(is_vegan = ifelse(str_detect(str_to_lower(products$product_name),'vegan'), 1,0))

glimpse(products)
summary(products)

### Build out the transaction data 

# Concatenate the prior and training order dataframes to have a full set of transaction details
# Before merging the dataframes, check of they have the same column structure

# Do the dataframes have the same column headers?
names(order_prior) == names(order_train)

# Column headers are the same.
# To avoid duplication of transaction data, check for order_id duplication in both dataframes
order_id_prior = order_prior %>%
  select(order_id) %>%
  unique()

order_id_train = order_train %>%
  select(order_id) %>%
  unique()

# If there is any duplication of order_id we will have to perform a dedupe of the data.
intersect(order_id_prior, order_id_train)

# Since there is no order_id duplication, we can safety perform a row-wise concatenation of the 2 dataframes
full.transactions = rbind(order_prior, order_train)

# Build a complete set of transactions, 
# To the SKU level transaction data, add the product information and the order information
full.transactions = full.transactions %>%
  left_join(products, by = "product_id")

# Add on order data
full.transactions = full.transactions %>%
  left_join(orders, by = "order_id")

# dim(full.transactions)
# summary(full.transactions)

### Build out the user data 
# 
# Examining the full_orders dataframe, we see that test orders do not appear in the transaction data,, they only appear in the orders dataframe. <br>
#   As we are concerned with complete transactions only to build out the user data, we can remove the test orders when developing the RFM matrix
full.orders = orders[orders$eval_set != "test",]

dim(orders)
dim(full.orders)
dim(orders[orders$eval_set == "test",])

### Cannot develop an RFM matrix due to the absence of spend data.
# We can however, use quantity of items as a proxy for testing under the assumption that larger numbers of items correspond to larger transaction values.

### Working out the "M" part of the RFM matrix
# Get the quantity of products in each order
quantity = full.transactions %>%
  group_by(order_id) %>%
  summarise(no_products = n())

# Add the order quantity to the order detail
full.orders = full.orders %>%
  left_join(quantity, by = "order_id")

# Sum the total items purchased by each user
quantity.items = full.orders %>%
  group_by(user_id) %>%
  summarise(total_products = sum(no_products))

# Summary to check for NA but OK!
summary(quantity.items)

### Working out the "R" part of the RFM matrix
# R in RFM terms refers to recency: how recently the customer / user has made a transaction
recent = full.orders %>%
  group_by(user_id) %>%
  summarise(average_days_between_transactions = mean(days_since_prior_order, na.rm = TRUE))

### Working out the "F" part of the RFM matrix 
# F in RFM terms refers to Frequency: how frequently does the customer / user make transactions
freq.transactions = full.orders %>%
  group_by(user_id) %>%
  summarise(no_transactions = n())

# Bringing the RFM data together
users.rfm = recent %>%
  left_join(freq.transactions, by = "user_id") %>%
  left_join(quantity.items, by = "user_id")

names(users.rfm) = c("user_id", "recency", "frequency", "quantity" )
head(users.rfm, 5)

# # Working out the scoring rank of users for each of the RFM (RFQ) scores
# RFM is scored on a scale from 1 - 10, where for:
#   - Recency: 1 = not recent, 10 = very recent
# - Frequency: 1 = not frequent, 10 = very frequent
# - Monetary (Quantity in this case): 1 = low quantity, 5 = high quantity
# 
# ### Scoring methodolgy
# First step is to split the range of values under each RFM heading in the number of groups: max_score <br>
#   Then scores are applied by bucketing users into each of their respective score band/quantiles.

max_score = 10

quant_score = seq(1, max_score)
quant_split = seq(0, 1, 1/max_score)
quants_recency = quantile(users.rfm$recency, quant_split)
quants_freq = quantile(users.rfm$frequency, quant_split)
quants_quantity = quantile(users.rfm$quantity, quant_split)

users.rfm$recency_score = as.numeric(cut(users.rfm$recency, quants_recency, quant_score))
users.rfm$freq_score = as.numeric(cut(users.rfm$frequency, quants_freq, quant_score))
users.rfm$quantity_score = as.numeric(cut(users.rfm$quantity, quants_quantity, quant_score))
users.rfm[is.na(users.rfm)] = 1
summary(users.rfm)

# Describe customers with special dietary needs
user.diet = full.transactions %>%
  group_by(user_id) %>%
  summarise(
    organic.pref = mean(is_organic),
    gluten.pref = mean(is_gluten_free),
    veg.pref = mean(is_vegetarian), 
    dairy.pref = mean(is_dairy_free), 
    vegan.pref = mean(is_vegan)
  )

# Users with a specific dietary need: gluten free / vegetarian / dairy free will be tagged as having a preferene (0 , 1)
# Users with a preference for organic will be tagged with the proportion of organic products they purchase.

user.diet$gluten.pref = (user.diet$gluten.pref > 0) * 1
user.diet$veg.pref = (user.diet$veg.pref > 0) * 1
user.diet$dairy.pref = (user.diet$dairy.pref > 0) * 1
user.diet$vegan.pref = (user.diet$vegan.pref > 0) * 1

# Bringing together RFM data with behvaioural data.
# We will join RFM with preferred shopping time, preferred shopping day and dietary preferences

users.rfm = users.rfm %>%
  left_join(user.diet)

### Building a usable Sample dataset

# The dataset is 33M+ observation, so we take a subset to develop our cluster input. <br>
#   1. Start by identifying a subset of customers and selecting their transactions
glimpse(users.rfm)

# There are 206,209 users, so we can take a sample of 5% to check on clusterability

set.seed(1)
sample_size = 0.05
mask = sample(1:nrow(users.rfm), round(nrow(users.rfm) * sample_size, 0))
sample.users = users.rfm[mask,]$user_id
sample.orders = full.orders[full.orders$user_id %in% sample.users,]$order_id
sample.data = full.transactions[full.transactions$order_id %in% sample.orders,]

### Breakdown by category
# Get a total number of products purchased per department per user
user.categories = sample.data %>%
  select(user_id,department_id) %>%
  left_join(departments, by = "department_id") %>%
  group_by(user_id, department) %>%
  summarise(
    count = n()
  ) 
summary(user.categories)

### Breakdown by sub-category
# Get a total number of products purchased per sub-category per user
# Breakdown by sub-category
user.subcategories = sample.data %>%
  select(department_id, aisle_id) %>%
  arrange(department_id, aisle_id) %>%
  left_join(aisles, by = "aisle_id") %>%
  left_join(departments, by = "department_id") %>%
  select(department_id, department, aisle_id, aisle) %>%
  group_by(department_id, department, aisle_id, aisle) %>%
  summarise(
    count = n()
  ) %>%
  arrange(desc(count))

sample.users = users.rfm[users.rfm$user_id %in% sample.users,]

# "One hot encode" the total customer values per department
ohe.users = spread(user.categories, key = department, value = count, fill = 0)
ohe.users = ohe.users %>% 
  drop_na()

# separate the user_id column from the user data.
user.data = ohe.users %>%
  left_join(user.diet)

##########################################################################################################################
### Due the large size of the data files, it may be necessary to remove these from working memory in order to maintain efficient processing<br>
# Sorting by object size and whether or not the objects are relevant from this point onwards, we can remove some larger objects from memory.
# rm(list = ls())
vars = ls()
size_of_objects_a = sapply(vars, function(x) format(object.size(get(x)), unit = 'auto'))
size_of_objects_b = sapply(vars, function(x) object.size(get(x)))
size_of_objects = data.frame(vars, size_of_objects_a, size_of_objects_b)
size_of_objects = size_of_objects %>%
  arrange(desc(size_of_objects_b)) %>%
  View()
sum(size_of_objects_b)
rm(list = c("full.transactions", "order_prior", "full.orders", "orders", "quantity", "order_id_prior", "order_train"))

##########################################################################################################################

### Hierarchical Clustering
### Investigate the clustering of user data based on the different linkage varieties to decide if one method is superior.
# All examples use scaled data, as some departments are inherantly more commonly used than others.

# Complete Linkage
h.clust_comp = hclust(dist(scale(user.data[-1])), method = "complete")
plot(h.clust_comp,main = "Complete Linkage - Scaled Data", xlab = "", ylab = "", sub = "", cex = 0.9)
scaled.user = cutree(h.clust_comp, 6)
table(scaled.user)

# # Average Linkage
# h.clust_avg = hclust(dist(scale(user.data)), method = "average")
# plot(h.clust_avg,main = "Average Linkage", xlab = "", ylab = "", sub = "", cex = 0.9)
# avg.cut = cutree(h.clust_avg, 5)
# table(avg.cut)
# 
# #Single Linkage
# h.clust_sing = hclust(dist(scale(user.data)), method = "single")
# plot(h.clust_sing, main = "Single Linkage", xlab = "", ylab = "", sub = "", cex = 0.9)
# sing.cut = cutree(h.clust_sing, 5)
# table(sing.cut)

### We can see that none of these clustering model are not very useful to identify a diverse set of users.
# We have other options with hierarchical clustering e.g. correlation based rather distance based clustering.
# This may be more practical in this environment where we want to identify customers with similar behaviour rather than similar traits.

# Cluster based on correlated data and complete linkage
samp_dist = as.dist(1-cor(t(user.data[-1])))
h.clust_cor = hclust(samp_dist, method = "complete")
plot(h.clust_cor, main = "Complete Linkage - Correlation", xlab = "", ylab = "", sub = "", cex = 0.9)
samp.cut = cutree(h.clust_cor, 6)
table(samp.cut)/sum(table(samp.cut))
corcomp.user = samp.cut

# We can see that correlation based approach ends up with a much more evenly spread cluster distribution. <br>
#   
#   Looking at a matrix of classification of euclidean vs correlation distance, we see that scaled euclidean distance puts 99.97% of users into cluster 1.<br>
#   Correlation distance has a more favourable distribution among the clusters, but a lower level cut would likely result in a more even distribution.

table(corcomp.user, scaled.user)
table(scaled.user)/sum(table(scaled.user))
table(corcomp.user)/sum(table(corcomp.user))


####################################################################################################


### K-Means Clustering
# Given that the first attempts using hieracrchical clustering did not give any insightful groupings using the transactional data, we can move to K-Means.<br>
#   K-Means being very popular and using numerical data, we should expect customers with similar buying patterns to be grouped closely in n-dimensional space. <br>
#   
#   1st step is to identify a useful value for k - the number of clusters that can be identified by the algorithm.<br>
#   Starting with 1 - 10, we will identify the number of clusters that minimises the within group sum of squares distance.


# Explore the sample data on users
dim(sample.users)[2]

sample.users[-c(1)] %>%
  gather(Attributes, value, 1:1) %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "lightblue2", color = "black") +
  facet_wrap(~Attributes, scales = "free_x") +
  labs(x = "Value", y = "Frequency")

# Explore correlated variables
corrplot(cor(scale(sample.users[-c(1, 2, 3, 4)])), type = "upper", method = "ellipse", tl.cex = 0.9)

# Perform PCA, exluding RFM values - when scaled they add no additional value to the quantiled scores.
res.pca = PCA(scale(sample.users[-c(1,2,3,4 )]), graph = FALSE)

# Screeplot shows 82.5% of variability can be explained with the first 5 principle components
# 72.9% with the first 4 principle components
fviz_screeplot(res.pca, addlabels = TRUE, ylim = c(0, 50))

# Get the contributions of each variable to the principle components
var <- get_pca_var(res.pca)

# PC1 is contributed to, mainly, by RFM scores.
fviz_contrib(res.pca, choice = "var", axes = 1 , top = 10)

# PC2 is contributed to, mainly, by vegan preference and dairy preference.
# Quantity does not effect PC2. moreso frequency of purchase and recency of purchase.
fviz_contrib(res.pca, choice = "var", axes = 2 , top = 10)

# PC3 is contributed to, mainly, by vegetarian preference and organic preference.
fviz_contrib(res.pca, choice = "var", axes = 3 , top = 10)

# Separation on the x-axis of PC1 vs PC2 is mostly effected by RFM, but y-axis is effected by dietary preference. 
fviz_pca_var(res.pca, col.var="contrib", axes = c(1, 2),
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE # Avoid text overlapping
             ) + theme_minimal() + ggtitle("Variables - PC1 vs PC2")

# Separation on the x-axis of Pc1 vs PC3 is mostly effected by RFM.
# Positive values effected by vegetarian and vegan preference
# Negative values effected by dairy, gluten free and organic preferences.
fviz_pca_var(res.pca, col.var="contrib", axes = c(1, 3),
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE # Avoid text overlapping
) + theme_minimal() + ggtitle("Variables - PC1 vs PC3")

# Separation on the x-axis of PC2 vs PC3 is mostly effected by RFM, but separation is not as pronouned as other PCA views.
# Y-Axis values are the same as above.
fviz_pca_var(res.pca, col.var="contrib", axes = c(2, 3),
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE # Avoid text overlapping
) + theme_minimal() + ggtitle("Variables - PC2 vs PC3")

set.seed(1)
# function to compute total within-cluster sum of squares
# Using the elbow method, the optimal number of clusters is between 4 and 6
fviz_nbclust(sample.users[-c(1, 2, 3, 4)], kmeans, method = "wss", k.max = 10) + theme_minimal() + ggtitle("the Elbow Method")


# Gap Statistic suggest that the optimal number of clusters is 6
gap.stat <- clusGap(sample.users[-c(1, 2, 3, 4)], FUN = kmeans, nstart = 30, K.max = 8, B = 50)
fviz_gap_stat(gap.stat) + theme_minimal() + ggtitle("fviz_gap_stat: Gap Statistic")


# Silhouette method suggests that the optimal number of clusters is 2
fviz_nbclust(sample.users[-c(1, 2, 3, 4)], kmeans, method = "silhouette", k.max = 10) + theme_minimal() + 
  ggtitle("The Silhouette Plot")


# Within Sum of Squares:
# Method aims to minimise the within-cluster sum of squares and maximise the between cluster sum of squares.
# After k = 4, the between sum of squares shows marginal increase for larger values of k.
# Even with a decreasing value of within sum of squares, we should use a smaller value of k.
# In this case, 4.

set.seed(1)
km1 = kmeans(sample.users[-c(1, 2, 3, 4)], 1, nstart = 25)
km2 = kmeans(sample.users[-c(1, 2, 3, 4)], 2, nstart = 25)
km3 = kmeans(sample.users[-c(1, 2, 3, 4)], 3, nstart = 25)
km4 = kmeans(sample.users[-c(1, 2, 3, 4)], 4, nstart = 25)
km5 = kmeans(sample.users[-c(1, 2, 3, 4)], 5, nstart = 25)
km6 = kmeans(sample.users[-c(1, 2, 3, 4)], 6, nstart = 25)
km7 = kmeans(sample.users[-c(1, 2, 3, 4)], 7, nstart = 25)
km8 = kmeans(sample.users[-c(1, 2, 3, 4)], 8, nstart = 25)
km9 = kmeans(sample.users[-c(1, 2, 3, 4)], 9, nstart = 25)
km10 = kmeans(sample.users[-c(1, 2, 3, 4)], 10, nstart = 25)

kms = data.frame(km1$cluster, km2$cluster, km3$cluster, km4$cluster, km5$cluster, km6$cluster, 
                   km7$cluster, km8$cluster)

# Dataframe to hold within and between distances
ssc <- data.frame(
  kmeans = c(2,3,4,5,6,7,8, 9, 10),
  within_ss = c(mean(km2$withinss), mean(km3$withinss), mean(km4$withinss), mean(km5$withinss), 
                mean(km6$withinss), mean(km7$withinss), mean(km8$withinss), mean(km9$withinss), mean(km10$withinss)),
  between_ss = c(km2$betweenss, km3$betweenss, km4$betweenss, km5$betweenss, 
                 km6$betweenss, km7$betweenss, km8$betweenss, km9$betweenss, km10$betweenss)
)

# Tabulate the within and between values.
ssc %<>% gather(., key = "measurement", value = value, -kmeans)

ssc %>% ggplot(., aes(x=kmeans, y=log10(value), fill = measurement)) + 
  geom_bar(stat = "identity", position = "dodge") + 
  ggtitle("Cluster Model Comparison") + xlab("Number of Clusters") + 
  ylab("Log10 Total Sum of Squares") + 
  scale_x_discrete(name = "Number of Clusters", limits = c("0","2", "3", "4", "5", "6", "7", "8", "9", "10"))


# # Using NbClust
# # For use on kaggle or databricks only
# res.nbclust <- NbClust(sample.users[-c(1, 2, 3, 4)], distance = "euclidean",
#                        min.nc = 2, max.nc = 6, 
#                        method = "kmeans", index ="all")
# factoextra::fviz_nbclust(res.nbclust) + theme_minimal() + ggtitle("NbClust's optimal number of clusters")

  # Using Clustree
  
  # add a prefix to the column names
  names(kms) <- seq(1:8)
  names(kms) <- paste0("k",colnames(kms))
  
  # get individual PCA values
  kms.pca <- prcomp(kms, center = TRUE, scale. = FALSE)
  ind.coord <- kms.pca$x
  ind.coord <- ind.coord[,1:2]
  
  # bind the dataframes 
  df <- bind_cols(as.data.frame(kms), as.data.frame(ind.coord))
  
  # Generate the cluster tree for PC1 & PC2
  clustree(df, prefix = "k")
  
  set.seed(1)
  km.res = eclust(sample.users[-c(1, 2, 3, 4)], "kmeans", k = 4, nstart = 25, graph = FALSE)
  # Better visuals for displaying clusters across PC1 - PC5
  fviz_cluster(km.res, axes = c(1, 2),geom = "point", ellipse.type = "norm", palette = "jco", ggtheme = theme_minimal(), main = "PC1 vs PC2")
  fviz_cluster(km.res, axes = c(1, 3),geom = "point", ellipse.type = "norm", palette = "jco", ggtheme = theme_minimal(), main = "PC1 vs PC3")
  fviz_cluster(km.res, axes = c(1, 4),geom = "point", ellipse.type = "norm", palette = "jco", ggtheme = theme_minimal(), main = "PC1 vs PC4")
  fviz_cluster(km.res, axes = c(1, 5),geom = "point", ellipse.type = "norm", palette = "jco", ggtheme = theme_minimal(), main = "PC1 vs PC5")
  
  # Generate a silhouette plot for K = 4
  fviz_silhouette(km.res, palette = "jco", gtheme = theme_classic())
  
  # Get average values of RFM and preferences per cluster
  users.behaviour = as.data.frame(sample.users[-1]) %>% 
    mutate(Cluster = km.res$cluster) %>% 
    group_by(Cluster) %>% summarise_all("mean") %>% 
    mutate_at(2:7, round, 0) %>%  
    mutate_at(8:12, round, 2)
  
  # Get average values of departments shopped from transaction data per cluster
  users.transactions = as.data.frame(user.data[-c(1, 23, 24, 25, 26, 27)]) %>% 
    mutate(Cluster = km.res$cluster) %>% 
    group_by(Cluster) %>% summarise_all("mean") %>% 
    mutate_at(2:22, round, 0)
  
  # Join the dataframes
  users.all = cbind(users.behaviour[-1], users.transactions[-c(1, 2)])   
  
  # Transpose for readability
  users.overall <- data.frame(t(users.all[-1]))
  colnames(users.overall) <- c(1, 2, 3, 4)
  
  # Print formatted table of behaviours.
  round(users.overall[1:4], 1) %>%
    kable() %>%
    kable_styling()
  
  ##########################################################################################################################

# Explore the sample transaction data

user.data[-1] %>%
  gather(Attributes, value, 1:26) %>%
  ggplot(aes(x = value)) +
  geom_histogram(fill = "lightblue2", color = "black") +
  facet_wrap(~Attributes, scales = "free_x") +
  labs(x = "Value", y = "Frequency")

glimpse(user.data)
dim(user.data)
temp.user.data = data.frame(user.data)
user.data = user.data[-c(1, 24, 25, 26, 27)]

# Highlight correlated variables
corrplot(cor(scale(user.data[-c(1)])), type = "upper", method = "ellipse", tl.cex = 0.9)

# Perform PCA, exluding user_id
res.pca.user = PCA(scale(user.data[-c(1)]), graph = FALSE)

# Screeplot shows 35.4% of variability can be explained with the first 2 principle components
# Additional components add little value
fviz_screeplot(res.pca.user, addlabels = TRUE, ylim = c(0, 50))

# Get the contributions of each variable to the principle components
var.user <- get_pca_var(res.pca.user)

# PC1 is contributed to, mainly, by purchases in Pantry and Produce.
fviz_contrib(res.pca.user, choice = "var", axes = 1 , top = 10)

# PC2 is contributed to, mainly, organic preference and household purchases.
fviz_contrib(res.pca.user, choice = "var", axes = 2 , top = 10)

# PC3 is contributed to, mainly, by dairy free and vegan preferences.
fviz_contrib(res.pca.user, choice = "var", axes = 3 , top = 10)

# Separation on the x-axis of PC1 vs PC2 is mostly effected by long life versus short life departments
# but y-axis is effected by dietary preference and preference for alcohol and personal care. 
fviz_pca_var(res.pca.user, col.var="contrib", axes = c(1, 2),
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE # Avoid text overlapping
) + theme_minimal() + ggtitle("Variables - PC1 vs PC2")

set.seed(1)
# function to compute total within-cluster sum of squares
# Using the elbow method, the optimal number of clusters is between 3
fviz_nbclust(user.data[-c(1)], kmeans, method = "wss", k.max = 10) + theme_minimal() + ggtitle("the Elbow Method")


# Gap Statistic suggest that the optimal number of clusters is 4
gap.stat.user <- clusGap(user.data[-c(1)], FUN = kmeans, nstart = 30, K.max = 6, B = 50)
fviz_gap_stat(gap.stat.user) + theme_minimal() + ggtitle("fviz_gap_stat: Gap Statistic")


# Silhouette method suggests that the optimal number of clusters is 2
fviz_nbclust(user.data[-c(1)], kmeans, method = "silhouette", k.max = 8) + theme_minimal() + ggtitle("The Silhouette Plot")

# Within Sum of Squares:
# Method aims to minimise the within-cluster sum of squares and maximise the between cluster sum of squares.
# After k = 4, the between sum of squares shows marginal increase for larger values of k.
# Even with a decreasing value of within sum of squares, we should use a smaller value of k.
# In this case, 4.

set.seed(1)
km1.user = kmeans(user.data[-c(1)], 1, nstart = 25)
km2.user = kmeans(user.data[-c(1)], 2, nstart = 25)
km3.user = kmeans(user.data[-c(1)], 3, nstart = 25)
km4.user = kmeans(user.data[-c(1)], 4, nstart = 25)
km5.user = kmeans(user.data[-c(1)], 5, nstart = 25)
km6.user = kmeans(user.data[-c(1)], 6, nstart = 25)
km7.user = kmeans(user.data[-c(1)], 7, nstart = 25)
km8.user = kmeans(user.data[-c(1)], 8, nstart = 25)
km9.user = kmeans(user.data[-c(1)], 9, nstart = 25)
km10.user = kmeans(user.data[-c(1)], 10, nstart = 25)

kms.user = data.frame(km1.user$cluster, km2.user$cluster, km3.user$cluster, km4.user$cluster, km5.user$cluster, km6.user$cluster, 
                 km7.user$cluster, km8.user$cluster)

ssc.user <- data.frame(
  kmeans = c(2, 3, 4, 5, 6, 7, 8, 9, 10),
  within_ss = c(mean(km2.user$withinss), mean(km3.user$withinss), mean(km4.user$withinss), mean(km5.user$withinss), 
                mean(km6.user$withinss), mean(km7.user$withinss), mean(km8.user$withinss), mean(km9.user$withinss), mean(km10.user$withinss)),
  between_ss = c(km2.user$betweenss, km3.user$betweenss, km4.user$betweenss, km5.user$betweenss, 
                 km6.user$betweenss, km7.user$betweenss, km8.user$betweenss, km9.user$betweenss, km10.user$betweenss)
)

# Comparing within and between sum of squares, the ideal value of K = 4
ssc.user %<>% gather(., key = "measurement", value = value, -kmeans)
#ssc$value <- log10(ssc$value)
ssc.user %>% ggplot(., aes(x=kmeans, y=log10(value), fill = measurement)) + 
  geom_bar(stat = "identity", position = "dodge") + 
  ggtitle("Cluster Model Comparison") + xlab("Number of Clusters") + 
  ylab("Log10 Total Sum of Squares") + 
  scale_x_discrete(name = "Number of Clusters", limits = c("0","2", "3", "4", "5", "6", "7", "8", "9", "10"))


# # Using NbClust
# # For use on kaggle or databricks only
# res.nbclust <- NbClust(sample.users[-c(1, 2, 3, 4)], distance = "euclidean",
#                        min.nc = 2, max.nc = 6, 
#                        method = "kmeans", index ="all")
# factoextra::fviz_nbclust(res.nbclust) + theme_minimal() + ggtitle("NbClust's optimal number of clusters")

# Using Clustree
# add a prefix to the column names
names(kms.user) <- seq(1:8)
names(kms.user) <- paste0("k",colnames(kms.user))

# get individual PCA
kms.pca.user <- prcomp(kms.user, center = TRUE, scale. = FALSE)
ind.coord.user <- kms.pca.user$x
ind.coord.user <- ind.coord.user[,1:2]

df.user <- bind_cols(as.data.frame(kms.user), as.data.frame(ind.coord.user))
clustree(df, prefix = "k")

set.seed(1)
km.res.user = eclust(user.data[-c(1)], "kmeans", k = 3, nstart = 25, graph = FALSE)

# Better visuals for displaying clusters across PC1 - PC5
fviz_cluster(km.res.user, axes = c(1, 2),geom = "point", ellipse.type = "norm", palette = "jco", ggtheme = theme_minimal(), main = "PC1 vs PC2")
fviz_cluster(km.res.user, axes = c(1, 3),geom = "point", ellipse.type = "norm", palette = "jco", ggtheme = theme_minimal(), main = "PC1 vs PC3")
fviz_cluster(km.res.user, axes = c(1, 4),geom = "point", ellipse.type = "norm", palette = "jco", ggtheme = theme_minimal(), main = "PC1 vs PC4")
fviz_cluster(km.res.user, axes = c(1, 5),geom = "point", ellipse.type = "norm", palette = "jco", ggtheme = theme_minimal(), main = "PC1 vs PC5")

fviz_silhouette(km.res.user, palette = "jco", gtheme = theme_classic())

as.data.frame(user.data[-c(1, 22, 23, 24, 25, 26)]) %>% 
  mutate(Cluster = km.res$cluster) %>% 
  group_by(Cluster) %>% summarise_all("mean") %>% 
  mutate_at(2:25, round, 0) %>%
  kable() %>% 
  kable_styling()


# Compare the clustering of RFM vs department transactions
table(km.res$cluster)
table(km.res.user$cluster)

table(km.res$cluster, km.res.user$cluster)