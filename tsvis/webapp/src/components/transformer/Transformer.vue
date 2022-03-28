<template>
  <div class="transformer">
    <div class="visualization">
      <el-container>
        <el-header class="header">
          <el-row>
            <el-col>
              请选择要可视化注意力的语句：
              <el-select
                class="textSelect"
                v-model="getValue"
                @change="changeText"
              >
                <el-option
                  v-for="item in options"
                  :key="item.id"
                  :label="item.label"
                  :value="[item.id,item.label]"
                >
                </el-option>
              </el-select>
            </el-col>
          </el-row>
        </el-header>

        <el-main class="main">
          <text-attention-vis
            :testdata="data"
            :key="data"
          ></text-attention-vis>
        </el-main>
      </el-container>
    </div>
  </div>
</template>

<script>
import { createNamespacedHelpers } from 'vuex'
import TextAttentionVis from './text/TextAttentionVis'

import bert_test_data from '../../assets/bert_test_data.json'
import gpt2_test_data from '../../assets/gpt2_test_data.json'

export default {
  components: {
    TextAttentionVis
  },
  data () {
    return {
      getValue: 'The cat sat on the mat. The cat lay on the rug.',
      options: [{//选项数据：模拟从后端拿到的数据
        id: 1,
        label: 'The cat sat on the mat. The cat lay on the rug.'
      }, {
        id: 2,
        label: 'The quick brown fox jumps over the lazy dogs. It then quickly runs away.'
      }],
      data: { 0: bert_test_data, 1: gpt2_test_data }
    }
  },
  created () {
    this.data = bert_test_data
  },
  mounted () {

  },
  methods: {
    changeText (e) {
      let [id, label] = e
      if (id === 1) {
        this.data = bert_test_data
      } else if (id === 2) {
        this.data = gpt2_test_data
      }
    }
  }
}
</script>
<style scoped>
.transformer {
  width: 100%;
  height: 100%;
  background-color: white;
}

.visualization {
  height: 97.5%;
  margin: 1% 1% 0 1%;
  overflow-y: auto;
  background-color: white;
  border-radius: 5px 5px 0 0;
  box-shadow: rgba(0, 0, 0, 0.3) 0 0 10px;
}

.textSelect {
  margin-top: 3%;
  direction: ltr;
  width: 600px;
}
</style>