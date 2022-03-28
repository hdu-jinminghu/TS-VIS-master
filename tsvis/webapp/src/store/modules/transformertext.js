import http from '@/utils/request'
import port from '@/utils/api'

/* eslint-disable */

const state = {
  'attention': {},
  'defaultFilter': 'all',
  "bidirectional": true,
  "displayMode": "light",
  "defaultLayer": 0,
  "defaultHead": 0
}

const getters = {
  getAttention: (state) => state.attention,
  getDefaultFilter: (state) => state.defaultFilter,
  getBidirctional: (state) => state.bidirectional,
  getDisplayMode: (state) => state.displayMode,
  getDefaultLayer: (state) => state.defaultLayer,
  getDefaultHead: (state) => state.defaultHead,
}

const actions = {
  // 设置类目信息
  async getSelfCategoryInfo (context, param) {
    context.commit("setTransformerTextCategroy", param)
  },
  async fetchTransformerTextData (context, param) {
    await http.useGet(port.category.transformertext, param).then((res) => {
      if (Number(res.data.code) !== 200) {
        context.commit('setErrorMessage', param.run + ',' + param.tag + ',' + res.data.msg)
        return
      }
      context.commit("setTransformerTextData", res.data.data)
    })
  }
}

const mutations = {
  setTransformerTextData: (state, param) => {

  },
  setErrorMessage (state, param) {
    state.errorMessage = param
  },
  setTransformerTextCategroy (state, param) {

  }
}

export default {
  namespaced: true,
  state,
  getters,
  actions,
  mutations
}
