/*
 * @Author: your name
 * @Date: 2021-12-08 14:45:44
 * @LastEditTime: 2021-12-14 17:29:12
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: \webapp\src\store\modules\features.js
 */
import http from '@/utils/request'
import port from '@/utils/api'

/* eslint-disable */

const state = {
  "gray_map":[],
  "GradCam":[],
  "guide_backward_map":[],
  "PFV":[],
  "log":{},

}

const getters = {
  getGray:(state) => state.gray_map,
  getGradCam:(state) => state.GradCam,
  getGuideBackward:(state) => state.guide_backward_map,
  getPFV:(state) => state.PFV,
  getLog:(state) => state.log
}

const actions = {
  // 设置类目信息
  async getSelfCategoryInfo(context,param){
    context.commit("setMapCategroy",param)
  },
  async fetchFeatures(context,param){
    await http.useGet(port.category.features, param).then((res)=>{
      if (Number(res.data.code) !== 200) {
        context.commit('setErrorMessage', param.run + ',' + param.tag + ',' + res.data.msg)
        return
      }
      context.commit("setFeaturesData",res.data.data)
    })
  }
}

const mutations = {
  setFeaturesData:(state,param)=>{
    let type = Object.keys(param)[0];
    state[type] = param[type][0]["value"];
  },
  setErrorMessage(state, param) {
    state.errorMessage = param
  },
  setMapCategroy(state, param) {
    for(let categoryInfoIndex=0;categoryInfoIndex<param[0].length;categoryInfoIndex+=1){
      state.log[param[0][categoryInfoIndex]] = param[1][categoryInfoIndex]
    }
    console.log(state.log)
  }
}

export default {
  namespaced: true,
  state,
  getters,
  actions,
  mutations
}
