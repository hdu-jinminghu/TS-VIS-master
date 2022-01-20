<!--
 * @Author: your name
 * @Date: 2021-12-08 14:35:49
 * @LastEditTime: 2021-12-23 10:33:03
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \webapp\src\components\features\Features\Features.vue
-->
<template>
  <div>
    <div :class="['features-container']">
      <template v-for="featureMap in Object.keys(featureMapType)">
        <div :key="featureMap" :class="featureMapType[featureMap]['class']">
          <div :class="['header']" @click="toggle($event)">
            <div :class="['label-container']"><span>{{featureMap}}</span></div>
            <div :class="['circle-container']"><div :class="['circle']"></div></div>
            <div :class="['triangle-container']"><div :class="['triangle']"></div></div>
            <div :class="['tail-container']"><div :class="['tail']"></div></div>
          </div>
          <div :class="[featureMapType[featureMap]['class']['turn-off']?'hidden':'']">
            <template v-if="ONE_MAP.includes(featureMap)">
              <el-row>
                <template v-for="(mapData,index) in featuresMapData[featureMap]">
                  <el-col :span="3" :key="`${featureMap}_${index}`">
                    <div :class="['map-contrainer']">
                      <el-card>
                        <FeatureMap :url="mapData" ></FeatureMap>
                      </el-card>
                    </div>
                  </el-col>
                </template>
              </el-row>
              <div :class="['show-more']" @click="showMore(featureMap)"> 显示更多...</div>
            </template>
            <!-- 导向梯度 -->
            <template v-else>
              <template v-for="layer in Object.keys(featuresMapData[featureMap])">
                <div :key="layer">
                  <div :id="layer"  :class="['small-header']" @click="toggle($event)">
                    <div :class="['small-label-container']"><span>{{layer}}</span></div>
                    <div :class="['small-triangle-container']"><div :class="['small-triangle']"></div></div>
                    <div :class="['small-tail-container']"><div :class="['small-tail']"></div></div>
                  </div>
                </div>
                <el-row :key="`${layer}_row`" :id="`${layer}_row`">
                  <template v-for="(mapData,index) in featuresMapData[featureMap][layer]">
                    <el-col :span="3" :key="`${featureMap}_${layer}_${index}`">
                      <div :class="['map-contrainer']">
                        <el-card>
                          <FeatureMap :url="mapData" ></FeatureMap>
                        </el-card>
                      </div>
                    </el-col>
                  </template>
                </el-row>
                <div :class="['show-more']" @click="showMore(featureMap)" :key="`${layer}_show_more`"> 显示更多...</div>
              </template>
            </template>
            
          </div>
        </div>
      </template>
    </div>
  </div>
</template>

<script>
import methods from './Features.methods';
import * as d3 from 'd3';
import { featureMapType, featuresMapData} from './init'
import { FeatureMap } from './FeatureMap'
import { createNamespacedHelpers } from 'vuex'
const {
  mapGetters: mapFeaturemapGetters,
  mapActions: mapFeaturemapActions,
} = createNamespacedHelpers('featuremap')
const { mapState: mapLayoutStates } = createNamespacedHelpers('layout')

export default {
  data(){
    return {
      featureMapType:{},
      featuresMapData:{},
      gray_map:[],
      GradCam:[],
      PFV:[],
      guide_backward_map:[],
      type:["GradCam","PFV"],
      ONE_MAP:["grad-cam-map", "pfv-map"]
    }
  },
  components:{
    FeatureMap
  },
  created(){
    this.featuresMapData = featuresMapData;
    this.featureMapType = featureMapType;
  },
  mounted(){
    this.init();
  },
  watch:{
    getGradCam:{
      immediate:true,
      handler(val){
        this.featuresMapData["grad-cam-map"] && (function(that){
          that.featuresMapData["grad-cam-map"].splice(that.featuresMapData["grad-cam-map"].length,0,...val)
        })(this)
      }
    },
    getPFV:{
      immediate:true,
      handler(val){
        this.featuresMapData["pfv-map"] && (function(that){
          that.featuresMapData["pfv-map"].splice(that.featuresMapData["pfv-map"].length,0,...val)
        })(this)
      }
    },
    userSelectRunFile:{
      immediate:true,
      handler(val){
        if(this.getLog[val]){
          for(let feature of this.type){
            if(this.getLog[val].indexOf(feature) >= 0){
              this.show(val,feature, true);
            } else {
              this.show(val,feature, false);
            }
          }
        }
      }
    }
  },
  computed:{
    ...mapFeaturemapGetters(["getGray", "getGradCam", "getGuideBackward", "getPFV","getLog"]),
    ...mapLayoutStates(['userSelectRunFile']),
  },
  methods:{
    ...methods,
    ...mapFeaturemapActions(["fetchFeatures"]),
  },
}
</script>


<style lang="less" scoped>
@import 'Features.less';
</style>