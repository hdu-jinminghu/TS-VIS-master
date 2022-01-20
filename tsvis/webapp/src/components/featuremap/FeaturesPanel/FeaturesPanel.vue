<!--
 * @Author: your name
 * @Date: 2021-12-08 14:35:49
 * @LastEditTime: 2021-12-08 15:03:28
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \webapp\src\components\features\FeaturesPanel\FeaturesPanel.vue
-->

<template>
  <div class="temp">
    Features paramenter panel
  </div>
</template>

<script>

import methods from './Feature.methods'

export default {
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
@import 'FeaturesPanel.less';
</style>
