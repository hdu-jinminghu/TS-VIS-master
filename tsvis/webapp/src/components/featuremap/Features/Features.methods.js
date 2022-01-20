/*
 * @Author: your name
 * @Date: 2021-12-08 14:48:45
 * @LastEditTime: 2021-12-23 11:01:26
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \webapp\src\components\features\Features\Features.methods.js
 */
import * as d3 from 'd3'
import { CETOSTYLEMAP, STYLEMAPTOCE } from './init'
const methods = {
  // 界面初始化
  init() {
    // 初始化设置为空
    for (const type in this.featureMapType) {
      this.featureMapType[type]['class']['empty'] = true
    }
    // TODO
    this.featureMapType['guide-backward-map']['class']['empty'] = false

    if (this.getLog[this.userSelectRunFile]) {
      for (const feature of this.type) {
        if (this.getLog[this.userSelectRunFile].indexOf(feature) >= 0) {
          this.show(this.userSelectRunFile, feature, true)
        } else {
          this.show(this.userSelectRunFile, feature, false)
        }
      }
    }
  },
  // header颜色控制
  toggle: function(e) {
    const bubblePath = e.path
    for (const path of bubblePath) {
      try {
        let ele = d3.select(path)
        let classes = ele.attr('class')?ele.attr('class').split(' '):[]

        for (const itemClass of classes) {
          if (itemClass in this.featureMapType) {
            this.featureMapType[itemClass]['class']['turn-off'] = !this.featureMapType[itemClass]['class']['turn-off']
            return
          }
        }
        if (ele.property('id')) {
          let turn_off_class = ele.classed("turn-off")? false:true;
          ele.classed("turn-off",turn_off_class)
          d3.select(`#${ele.property('id')}_row`).classed("hidden",turn_off_class)
          return
        }
      } catch (err) {
        continue
      }
    }
  },
  // 初始化数据栏
  show: function(run, type, show) {
    Object.keys(CETOSTYLEMAP).includes(type) && this.featureMapType[CETOSTYLEMAP[type]] && (function(that){
      show && that.featureMapType[CETOSTYLEMAP[type]]['class']['empty'] && (function(){
        that.featuresMapData[CETOSTYLEMAP[type]].splice(0, that.featuresMapData[CETOSTYLEMAP[type]].length)
        that.fetchFeatures({ 'run': that.userSelectRunFile, 'tag': type, 'range': 0 })
      })()
      that.featureMapType[CETOSTYLEMAP[type]]['class']['empty'] = !show
    })(this)
  },
  // 加载键
  showMore: function(sType) {
    const iType = STYLEMAPTOCE[sType]
    const rangeStart = this.featuresMapData[sType].length
    this.fetchFeatures({ 'run': 'root', 'tag': iType, 'range': rangeStart })
  }
}

export default methods
