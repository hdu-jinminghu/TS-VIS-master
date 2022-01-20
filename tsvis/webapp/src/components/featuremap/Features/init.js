/*
 * @Author: your name
 * @Date: 2021-12-08 15:22:41
 * @LastEditTime: 2021-12-23 09:43:47
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \webapp\src\components\features\Features\init.js
 */
import { imageData } from './test'

const featureMapType = {
    "gray-map":{"class":{"map-container":true,"turn-off":false,"gray-map":true,"empty":true}},
    "grad-cam-map":{"class":{"map-container":true,"turn-off":false,"grad-cam-map":true,"empty":true}},
    "pfv-map":{"class":{"map-container":true,"turn-off":false,"pfv-map":true,"empty":true}},
    "guide-backward-map":{"class":{"map-container":true,"turn-off":false,"guide-backward-map":true,"empty":true}},
}

const featuresMapData = {
    "gray-map":[],
    "grad-cam-map":[],
    "pfv-map":[],
    "guide-backward-map":{"layer1":imageData,"layer2":imageData,"layer3":imageData,"layer4":imageData},
}

const CETOSTYLEMAP = {"PFV":"pfv-map","GradCam":"grad-cam-map"}

const STYLEMAPTOCE = {"pfv-map":"PFV","grad-cam-map":"GradCam"}

const LAYERMAP = ["guide-backward-map"]

export {featureMapType, featuresMapData, CETOSTYLEMAP, STYLEMAPTOCE,LAYERMAP};