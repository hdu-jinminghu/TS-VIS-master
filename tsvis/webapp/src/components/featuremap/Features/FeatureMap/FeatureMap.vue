<!--
 * @Author: your name
 * @Date: 2021-12-08 20:02:24
 * @LastEditTime: 2021-12-09 14:51:37
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \webapp\src\components\features\Features\FeatureMap\featureMap.vue
-->

<template>
  <div>
    <el-image
      :class="['image']"
      :src="url"
      :fit="'fill'"
      :preview-src-list="[url]">
    </el-image>
  </div>
</template>

<script>
import methods from './FeatureMap.methods';
import * as d3 from 'd3';
export default {
  props: {
      url:String,
  },
  data(){
    return {
    }
  },
  mounted(){
  },
  methods:{
    ...methods
  }

}
</script>


<style lang="less" scoped>
@import 'FeatureMap.less';
</style>