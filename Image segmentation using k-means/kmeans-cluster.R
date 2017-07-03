args = commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Pass two arguments <input-image-file-name> and <output-file-name>.\n", call.=FALSE)
} else {
  input = args[1]
  output = args[2]  
}

library("jpeg")
library("ggplot2")
inputImg = readJPEG(input)
imgDm = dim(inputImg)
imgRGB = data.frame(
  x = rep(1:imgDm[2], each = imgDm[1]),
  y = rep(imgDm[1]:1, imgDm[2]),
  R = as.vector(inputImg[,,1]),
  G = as.vector(inputImg[,,2]),
  B = as.vector(inputImg[,,3])
)
kClusters = 3
kMeans = kmeans(imgRGB[, c("R", "G", "B")], centers = kClusters)
kColours = rgb(kMeans$centers[kMeans$cluster,])

#Create the image
images = ggplot(data = imgRGB, aes(x = x, y = y)) + geom_point(colour = kColours) +
  labs(title = paste("Clustering of", kClusters, "Colours")) + xlab("x") + ylab("y") 
ggsave(output,plot = images)