/*
 * @Description: router
 * @version: 1.0
 * @Author: xds
 * @Date: 2020-05-08 10:52:00
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2021-12-14 15:42:40
 */
import Vue from 'vue'
import Router from 'vue-router'
import Layout from '@/components/layout/Layout.vue'
import { Scalars, ScalarsPanel } from '@/components/scalars'
import { Medias, MediasPanel } from '@/components/medias'
import { Graphs, GraphsPanel } from '@/components/graphs'
import { Hyperparms, HyperparmsPanel } from '@/components/hyperparms'
import { Features, FeaturesPanel } from '@/components/featuremap'
import { Customs, CustomsPanel } from '@/components/customs'
import { Statistics, StatisticsPanel } from '@/components/statistics'
import { ROCs, ROCsPanel } from '@/components/rocs'
import { Embeddings, EmbeddingsPanel } from '@/components/embeddings'
import { Exception, ExceptionPanel } from '@/components/exception'
import { Transformer, TransformerPanel } from '@/components/transformer'

Vue.use(Router)

export default new Router({
  routes: [
    { path: '/', redirect: '/index' },
    {
      path: '/index',
      name: 'Layout',
      component: Layout,
      children: [
        {
          path: 'graph',
          components: {
            default: Graphs,
            right: GraphsPanel
          }
        },
        {
          path: 'scalar',
          components: {
            default: Scalars,
            right: ScalarsPanel
          }
        },
        {
          path: 'media',
          components: {
            default: Medias,
            right: MediasPanel
          }
        },
        {
          path: 'statistic',
          components: {
            default: Statistics,
            right: StatisticsPanel
          }
        },
        {
          path: 'embedding',
          components: {
            default: Embeddings,
            right: EmbeddingsPanel
          }
        },
        {
          path: 'featuremap',
          components: {
            default: Features,
            right: FeaturesPanel
          }
        },
        {
          path: 'roc',
          components: {
            default: ROCs,
            right: ROCsPanel
          }
        },
        {
          path: 'hyperparm',
          components: {
            default: Hyperparms,
            right: HyperparmsPanel
          }
        },
        {
          path: 'exception',
          components: {
            default: Exception,
            right: ExceptionPanel
          }
        },
        {
          path: 'transformer',
          components: {
            default: Transformer,
            right: TransformerPanel
          }
        },
        {
          path: 'custom',
          components: {
            default: Customs,
            right: CustomsPanel
          }
        }
      ]
    }
  ]
})
