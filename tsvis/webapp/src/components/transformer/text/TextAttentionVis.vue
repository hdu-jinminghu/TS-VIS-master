<template>
  <div>
    <el-container>
      <el-header>
      </el-header>
      <el-main>
        <div
          id="bertviz"
          style="padding: 8px"
          align="left"
        >
          <span style="user-select: none">
            <span class="dropdown-label">Layer: </span>
            <select id="layer"></select>
            <span class="dropdown-label">Head: </span>
            <select id="att_head"></select>
            <span
              class="dropdown-label"
              v-if="is_sentence_pair"
            >Attention:
            </span>
            <select
              id="filter"
              v-if="is_sentence_pair"
            >
              <option value="all">All</option>
              <option value="aa">Sentence A -> Sentence A</option>
              <option value="ab">Sentence A -> Sentence B</option>
              <option value="ba">Sentence B -> Sentence A</option>
              <option value="bb">Sentence B -> Sentence B</option>
            </select>
          </span>
          <div id="vis"></div>
        </div>
      </el-main>
    </el-container>
  </div>
</template>

<script>
import { createNamespacedHelpers } from 'vuex'
import * as d3 from 'd3'
import $ from 'jquery'

const HEADING_TEXT_SIZE = 16
const HEADING_HEIGHT = 42
const TEXT_SIZE = 15
const MATRIX_WIDTH = 200
const BOXWIDTH = TEXT_SIZE * 8
const BOXHEIGHT = 26
const HEIGHT_PADDING = 100
const PADDING_WIDTH = 25
const DOT_WIDTH = 70
const SOFTMAX_WIDTH = 70
const ATTENTION_WIDTH = 150
const PALETTE = {
  'dark': {
    'attn': '#2994de',
    'neg': '#ff6318',
    'pos': '#2090dd',
    'text': '#ccc',
    'selected_text': 'white',
    'heading_text': 'white',
    'text_highlight_left': "#1b86cd",
    'text_highlight_right': "#1b86cd",
    'vector_border': "#444",
    'connector': "#2994de",
    'background': 'black',
    'dropdown': 'white',
    'icon': 'white'
  },
  'light': {
    'attn': 'blue',
    'neg': '#ff6318',
    'pos': '#0c36d8',
    'text': '#202020',
    'selected_text': 'black',
    'heading_text': 'black',
    'text_highlight_left': "#e5e5e5",
    'text_highlight_right': "#478be8",
    'vector_border': "#EEE",
    'connector': "blue",
    'background': 'white',
    'dropdown': 'black',
    'icon': '#888'
  }
}
const MIN_CONNECTOR_OPACITY = 0

let config = {}

export default {
  components: {
  },
  props: { testdata: Object },
  data () {
    return {
    }
  },
  computed: {
    is_sentence_pair: function () {
      return Object.keys(this.testdata['attention']).length > 1 ? true : false
    }
  },
  created () {
    config.attention = this.testdata['attention']  // 后端数据中的attention数据
    // console.log('attention_data:', config.attention)
    config.filter = this.testdata['default_filter']  // 默认为all
    // console.log("config.filter:", config.filter)
    let attentionFilter = config.attention[config.filter]  // 提取默认过滤器（all）的数据
    // console.log("attentionFilter:", attentionFilter)
    config.nLayers = attentionFilter['attn'].length  // 模型注意力层数
    // console.log("config.nLayers:", config.nLayers)
    config.nHeads = attentionFilter['attn'][0].length  // 每层头个数 
    // console.log("config.nHeads:", config.nHeads)
    config.vectorSize = attentionFilter['queries'][0][0][0].length  // Query向量长度
    // console.log("config.vectorSize:", config.vectorSize)
    config.headVis = new Array(config.nHeads).fill(true)
    config.initialTextLength = attentionFilter.right_text.length  // 字符串长度（右侧）
    // console.log("config.initialTextLength:", config.initialTextLength)
    config.expanded = false  // 默认不展开
    // console.log("config.expanded:", config.expanded)
    config.bidirectional = this.testdata['bidirectional']  // 是否是双向注意力
    // console.log("config.bidirectional:", config.bidirectional)
    config.mode = this.testdata['display_mode']  // 显示模式，分为dark和light
    // console.log("config.mode:", config.mode)
    config.layer = (this.testdata['layer'] == null ? 0 : this.testdata['layer'])  // 默认显示层
    // console.log("config.layer:", config.layer)
    config.head = (this.testdata['head'] == null ? 0 : this.testdata['head'])  // 默认显示头
    // console.log("config.head:", config.head)



  },
  mounted () {
    this.$nextTick(function () {
      // let ref = this
      const layerSelect = $("#bertviz #layer")
      layerSelect.empty()
      for (let i = 0; i < config.nLayers; i++) {
        layerSelect.append($("<option />").val(i).text(i))
      }
      layerSelect.val(config.layer).change()
      layerSelect.on('change', (e) => {
        config.layer = +e.currentTarget.value
        this.render()
      })

      const headSelect = $("#bertviz #att_head")
      headSelect.empty()
      for (let i = 0; i < config.nHeads; i++) {
        headSelect.append($("<option />").val(i).text(i))
      }
      headSelect.val(config.head).change()
      headSelect.on('change', (e) => {
        config.head = +e.currentTarget.value
        this.render()
      })

      const filterSelect = $("#bertviz #filter")
      filterSelect.on('change', (e) => {
        config.filter = e.currentTarget.value
        this.render()
      })

      this.render()
    })
  },
  watch: {

  },
  methods: {
    render () {
      let attnData = config.attention[config.filter]
      let leftText = attnData.left_text
      let rightText = attnData.right_text
      let queries = attnData.queries[config.layer][config.head]
      let keys = attnData.keys[config.layer][config.head]
      let att = attnData.attn[config.layer][config.head]

      $("#bertviz #vis").empty()
      let height = config.initialTextLength * BOXHEIGHT + HEIGHT_PADDING
      let svg = d3.select("#bertviz #vis")
        .append('svg')
        .attr("width", "100%")
        .attr("height", height + "px")

      d3.select("#bertviz")
        .style("background-color", this.getColor('background'))
      d3.selectAll("#bertviz .dropdown-label")
        .style("color", this.getColor('dropdown'))

      this.renderVisExpanded(svg, leftText, rightText, queries, keys)
      this.renderVisCollapsed(svg, leftText, rightText, att)
      if (config.expanded == true) {
        this.showExpanded()
      } else {
        this.showCollapsed()
      }
    },
    renderVisCollapsed (svg, leftText, rightText) {

      let posLeftText = 0;
      let posAttention = posLeftText + BOXWIDTH;
      let posRightText = posAttention + ATTENTION_WIDTH + PADDING_WIDTH;

      svg = svg.append("g")
        .attr("id", "collapsed")
        .attr("visibility", "hidden");

      this.renderText(svg, leftText, "leftText", posLeftText, false);
      this.renderAttn(svg, posAttention, posRightText, false);
      this.renderText(svg, rightText, "rightText", posRightText, false);
    },
    renderVisExpanded (svg, leftText, rightText, queries, keys) {

      let posLeftText = 0;
      let posQueries = posLeftText + BOXWIDTH + PADDING_WIDTH;
      let posKeys = posQueries + MATRIX_WIDTH + PADDING_WIDTH * 1.5;
      let posProduct = posKeys + MATRIX_WIDTH + PADDING_WIDTH;
      let posDotProduct = posProduct + MATRIX_WIDTH + PADDING_WIDTH;
      let posRightText = posDotProduct + BOXHEIGHT + PADDING_WIDTH;

      svg = svg.append("g")
        .attr("id", "expanded")
        .attr("visibility", "hidden");

      this.renderHeadingsExpanded(svg, posQueries, posKeys, posProduct, posDotProduct, posRightText);
      this.renderText(svg, leftText, "leftText", posLeftText, true);
      this.renderTextQueryLines(svg, posQueries - PADDING_WIDTH, posQueries - 2);
      this.renderVectors(svg, "keys", keys, posKeys);
      this.renderQueryKeyLines(svg, posQueries + MATRIX_WIDTH + 1, posKeys - 3);
      this.renderVectors(svg, "queries", queries, posQueries);
      this.renderHorizLines(svg, "hlines1", posProduct - PADDING_WIDTH + 1, posProduct - 1);
      this.renderVectors(svg, "product", keys, posProduct);
      this.renderHorizLines(svg, "hlines2", posDotProduct - PADDING_WIDTH + 2, posDotProduct);
      let dotProducts = new Array(rightText.length).fill(0);
      this.renderDotProducts(svg, dotProducts, posDotProduct);
      this.renderText(svg, rightText, "rightText", posRightText, true);
      this.renderHorizLines(svg, "hlines3", posRightText - PADDING_WIDTH - 2, posRightText);
      this.renderVectorHighlights(svg, "key-vector-highlights", posKeys);
      this.renderVectorHighlights(svg, "product-vector-highlights", posProduct)
    },
    renderHeadingsExpanded (svg, posQueries, posKeys, posProduct, posDotProduct, posSoftmax) {
      let headingContainer = svg.append("svg:g")
        .attr("id", "heading")

      let queryHeadingContainer = headingContainer.append("text")
        .attr("x", posQueries + 68)
        .attr("y", HEADING_HEIGHT - 12)
        .attr("height", BOXHEIGHT)
        .attr("width", MATRIX_WIDTH)
        .style('fill', this.getColor('heading_text'));

      queryHeadingContainer.append('tspan')
        .text('Query ')
        .attr("y", HEADING_HEIGHT - 12)
        .attr("font-size", HEADING_TEXT_SIZE + "px");

      queryHeadingContainer.append('tspan')
        .text('q')
        .attr("y", HEADING_HEIGHT - 12)
        .attr("font-size", HEADING_TEXT_SIZE + "px");

      let keyHeadingContainer = headingContainer.append("text")
        .attr("x", posKeys + 73)
        .attr("y", HEADING_HEIGHT - 12)
        .attr("height", BOXHEIGHT)
        .attr("width", MATRIX_WIDTH)
        .attr("font-size", HEADING_TEXT_SIZE + "px")
        .style('fill', this.getColor('heading_text'));

      keyHeadingContainer.append('tspan')
        .text('Key ')
        .style('font-size', HEADING_TEXT_SIZE + "px")
        .attr("y", HEADING_HEIGHT - 12);

      keyHeadingContainer.append('tspan')
        .text('k ')
        .style('font-size', HEADING_TEXT_SIZE + "px")
        .attr("y", HEADING_HEIGHT - 12);

      let productHeadingContainer = headingContainer.append("text")
        .attr("x", posProduct + 28)
        .attr("y", HEADING_HEIGHT - 12)
        .attr("height", BOXHEIGHT)
        .attr("width", MATRIX_WIDTH)
        .attr("font-size", HEADING_TEXT_SIZE + "px")
        .style('fill', this.getColor('heading_text'));

      productHeadingContainer.append('tspan')
        .text('q \u00D7 k (elementwise)')
        .style('font-size', HEADING_TEXT_SIZE + "px")
        .attr("y", HEADING_HEIGHT - 12);

      let dotProductHeadingContainer = headingContainer.append("text")
        .attr("x", posDotProduct - 6)
        .attr("y", HEADING_HEIGHT - 12)
        .attr("height", BOXHEIGHT)
        .attr("width", MATRIX_WIDTH)
        .attr("font-size", HEADING_TEXT_SIZE + "px")
        .style('fill', this.getColor('heading_text'));

      dotProductHeadingContainer.append('tspan')
        .text('q')
        .style('font-size', HEADING_TEXT_SIZE + "px")
        .attr("y", HEADING_HEIGHT - 12);

      dotProductHeadingContainer.append('tspan')
        .text(' \u2219 k')
        .style('font-size', HEADING_TEXT_SIZE + "px")
        .attr("y", HEADING_HEIGHT - 12);

      headingContainer.append("text")
        .attr("x", posSoftmax + 9)
        .attr("y", HEADING_HEIGHT - 12)
        .attr("height", BOXHEIGHT)
        .attr("width", SOFTMAX_WIDTH)
        .attr("font-size", HEADING_TEXT_SIZE + "px")
        .style("text-anchor", "start")
        .style('fill', this.getColor('heading_text'))
        .text("Softmax");

      headingContainer.append("text")
        .attr("id", "placeholder")
        .attr("x", posProduct + 55)
        .attr("y", HEADING_HEIGHT + 55)
        .attr("height", BOXHEIGHT)
        .attr("width", SOFTMAX_WIDTH + MATRIX_WIDTH + DOT_WIDTH)
        .attr("font-size", 20 + "px")
        .text("No token selected")
        .attr("fill", this.getColor('text_highlighted'));
    },
    renderTextQueryLines (svg, start_pos, end_pos) {
      let attnData = config.attention[config.filter];
      let leftText = attnData.left_text; // Use for shape not values
      let linesContainer = svg.append("svg:g");
      linesContainer.selectAll("line")
        .data(leftText)
        .enter()
        .append("line") // Add line
        .classed('text-query-line', true)
        .style("opacity", 0)
        .attr("x1", start_pos)
        .attr("y1", function (d, i) {
          return i * BOXHEIGHT + HEADING_HEIGHT + BOXHEIGHT / 2;
        })
        .attr("x2", end_pos)
        .attr("y2", function (d, i) {
          return i * BOXHEIGHT + HEADING_HEIGHT + BOXHEIGHT / 2;
        })
        .attr("stroke-width", 2)
        .attr("stroke", this.getColor('connector'))
    },
    renderVectors (svg, id, vectors, leftPos) {
      let vectorContainer = svg.append("svg:g")
        .attr("id", id);

      if (id == "product") {
        vectorContainer.style("opacity", 0);
      }

      let vector = vectorContainer.append("g") //.classed("attention_boxes", true) // Add outer group
        .selectAll("g")
        .data(vectors) // Loop over query/key vectors, one for each token
        .enter()
        .append("g") // Add (sub) group for each token
        .classed('vector', true)
        .attr("data-index", function (d, i) {
          return i;
        }) // make parent index available from DOM

      if (id == "queries") {
        vector.append("rect")
          .classed("vectorborder", true)
          .attr("x", leftPos - 1)
          .attr("y", (d, i) => {
            return i * BOXHEIGHT + HEADING_HEIGHT;
          })
          .attr("width", MATRIX_WIDTH + 2)
          .attr("height", BOXHEIGHT - 5)
          .style("fill-opacity", 0)
          .style("stroke-width", 1)
          .style("stroke", this.getColor('vector_border'))
          .attr("rx", 1)
          .attr("ry", 1)
          .style("stroke-opacity", 1)
      } else if (id == "keys") {
        vector.append("rect")
          .classed("vectorborder", true)
          .attr("x", leftPos - 1)
          .attr("y", function (d, i) {
            return i * BOXHEIGHT + HEADING_HEIGHT;
          })
          .attr("width", MATRIX_WIDTH + 2)
          .attr("height", BOXHEIGHT - 6)
          .style("fill-opacity", 0)
          .style("stroke-width", 1)
          .style("stroke", this.getColor('vector_border'))
          .attr("rx", 1)
          .attr("ry", 1)
          .style("stroke-opacity", 1)
      } else {
        vector.append("rect")
          .classed("vectorborder", true)
          .attr("x", leftPos - 1)
          .attr("y", function (d, i) {
            return i * BOXHEIGHT + HEADING_HEIGHT;
          })
          .attr("width", MATRIX_WIDTH + 2)
          .attr("height", BOXHEIGHT - 6)
          .style("fill-opacity", 0)
          .style("stroke-width", 1)
          .style("stroke", this.getColor('vector_border'))
          .attr("rx", 1)
          .attr("ry", 1)
          .style("stroke-opacity", 1)
      }

      vector.selectAll(".element")
        .data(function (d) {
          return d;
        }) // loop over elements within each query vector
        .enter() // When entering
        .append("rect") // Add rect element for each token index (j), vector index (i)
        .classed('element', true)
        .attr("x", function (d, i) { // i is vector index, j is index of token
          return leftPos + i * MATRIX_WIDTH / config.vectorSize;
        })
        .attr("y", function (d, i) {
          let j = +this.parentNode.getAttribute("data-index");
          return j * BOXHEIGHT + HEADING_HEIGHT;
        })
        .attr("width", MATRIX_WIDTH / config.vectorSize)
        .attr("height", BOXHEIGHT - 6)
        .attr("rx", .7)
        .attr("ry", .7)
        .attr("data-value", function (d) {
          return d
        })
        .style("fill", (d) => {
          if (d >= 0) {
            return this.getColor('pos');
          } else {
            return this.getColor('neg')
          }
        })
        .style("opacity", function (d) {
          return Math.tanh(Math.abs(d) / 4);
        })
    },
    renderQueryKeyLines (svg, start_pos, end_pos) {
      let attnMatrix = config.attention[config.filter].attn[config.layer][config.head];
      let linesContainer = svg.append("svg:g");
      let lineFunction = d3.line()
        .x(function (d) {
          return d.x;
        })
        .y(function (d) {
          return d.y;
        });

      linesContainer.selectAll("g")
        .data(attnMatrix)
        .enter()
        .append("g") // Add group for each source token
        .classed('qk-line-group', true)
        .style("opacity", 0)
        .attr("source-index", function (d, i) { // Save index of source token
          return i;
        })
        .selectAll("path")
        .data(function (d) { // Loop over all target tokens
          return d;
        })
        .enter() // When entering
        .append("path")
        .attr("d", function (d, targetIndex) {
          let sourceIndex = +this.parentNode.getAttribute("source-index");
          let y1 = sourceIndex * BOXHEIGHT + HEADING_HEIGHT + BOXHEIGHT / 2;
          let y2 = targetIndex * BOXHEIGHT + HEADING_HEIGHT + BOXHEIGHT / 2;
          let x1 = start_pos;
          let x2 = (start_pos + end_pos) / 2 + 1;
          let x3 = end_pos;

          return lineFunction([
            { 'x': x1, 'y': y1 },
            { 'x': x2, 'y': y1 },
            { 'x': x2, 'y': y2 },
            { 'x': x3, 'y': y2 },

          ])
        })
        .attr("fill", "none")
        .attr("stroke-width", 2)
        .attr("stroke", this.getColor('connector'))
        .attr("stroke-opacity", function (d) {
          if (d == 0) {
            return 0;
          } else {
            return Math.max(MIN_CONNECTOR_OPACITY, Math.tanh(Math.abs(1.8 * d)));
          }
        });
    },
    renderHorizLines (svg, id, start_pos, end_pos) {
      let attnMatrix = config.attention[config.filter].attn[config.layer][config.head];
      let linesContainer = svg.append("svg:g")
        .attr("id", id);
      linesContainer.selectAll("g")
        .data(attnMatrix)
        .enter()
        .append("g") // Add group for each source token
        .classed('horiz-line-group', true)
        .style("opacity", 0)
        .attr("source-index", function (d, i) { // Save index of source token
          return i;
        })
        .selectAll("line")
        .data(function (d) { // Loop over all target tokens
          return d;
        })
        .enter() // When entering
        .append("line")
        .attr("x1", start_pos)
        .attr("y1", function (d, targetIndex) {
          return targetIndex * BOXHEIGHT + HEADING_HEIGHT + BOXHEIGHT / 2;
        })
        .attr("x2", end_pos)
        .attr("y2", function (d, targetIndex) {
          return targetIndex * BOXHEIGHT + HEADING_HEIGHT + BOXHEIGHT / 2;
        })
        .attr("stroke-width", 2)
        .attr("stroke", this.getColor('connector'))
        .attr("stroke-opacity", function (d) {
          if (d == 0) {
            return 0;
          } else {
            return Math.max(MIN_CONNECTOR_OPACITY, Math.tanh(Math.abs(1.8 * d)));
          }
        });
    },
    renderDotProducts (svg, dotProducts, leftPos) {
      svg.append("svg:g")
        .attr("id", "dotproducts")
        .style("opacity", 0)
        .selectAll("rect")
        .data(dotProducts)
        .enter()
        .append("rect")
        .classed('dotproduct', true)
        .attr("x", leftPos + 1)
        .attr("y", function (d, i) {
          return i * BOXHEIGHT + HEADING_HEIGHT;
        })
        .attr("height", BOXHEIGHT - 4)
        .attr("width", BOXHEIGHT - 4)
        .style("stroke-width", 1.2)
        .style("stroke", this.getColor('vector_border'))
        .style("stroke-opacity", 1)
        .style("fill-opacity", 0)
        .attr("rx", 2)
        .attr("ry", 2)
    },
    renderVectorHighlights (svg, id, start_pos) {
      let attnMatrix = config.attention[config.filter].attn[config.layer][config.head];
      let vectorHighlightsContainer = svg.append("svg:g")
        .attr("id", id);
      vectorHighlightsContainer.selectAll("g")
        .data(attnMatrix)
        .enter()
        .append("g") // Add group for each source token
        .classed('vector-highlight-group', true)
        .style("opacity", 0)
        .attr("source-index", function (d, i) { // Save index of source token
          return i;
        })
        .selectAll("rect")
        .data(function (d) { // Loop over all target tokens
          return d;
        })
        .enter() // When entering
        .append("rect")
        .attr("x", start_pos - 1)
        .attr("y", function (d, targetIndex) {
          return targetIndex * BOXHEIGHT + HEADING_HEIGHT;
        })
        .attr("height", BOXHEIGHT - 5)
        .attr("width", MATRIX_WIDTH + 3)
        .style("fill-opacity", 0)
        .attr("stroke-width", 2)
        .attr("stroke", this.getColor('connector'))
        .attr("stroke-opacity", function (d) {
          return Math.tanh(Math.abs(1.8 * d));
        });
    },
    renderText (svg, text, id, leftPos, expanded) {

      let tokenContainer = svg.append("svg:g")
        .attr("id", id)
        .selectAll("g")
        .data(text)
        .enter()
        .append("g");
      if (id == "leftText" || id == "rightText") {
        let fillColor;
        if (id == "rightText") {
          fillColor = this.getColor('text_highlight_right');
        }
        if (id == "leftText") {
          fillColor = this.getColor('text_highlight_left');
        }

        tokenContainer.append("rect")
          .classed("highlight", true)
          .attr("fill", fillColor)
          .style("opacity", 0.0)
          .attr("height", BOXHEIGHT)
          .attr("width", BOXWIDTH)
          .attr("x", leftPos)
          .attr("y", function (d, i) {
            return i * BOXHEIGHT + HEADING_HEIGHT - 1;
          });
      }

      let offset;
      if (id == "leftText") {
        offset = -8;
      } else {
        offset = 8;
      }

      let textContainer = tokenContainer.append("text")
        .classed("token", true)
        .text(function (d) {
          return d;
        })
        .attr("font-size", TEXT_SIZE + "px")
        .style("fill", this.getColor('text'))
        .style("cursor", "default")
        .style("-webkit-user-select", "none")
        .attr("x", leftPos + offset)
        .attr("y", function (d, i) {
          return i * BOXHEIGHT + HEADING_HEIGHT;
        })
        .attr("height", BOXHEIGHT)
        .attr("width", BOXWIDTH)
        .attr("dy", TEXT_SIZE);

      if (id == "leftText") {
        textContainer.style("text-anchor", "end")
          .attr("dx", BOXWIDTH - 2);
        // let ref = this
        tokenContainer.on("mouseover", (d, index) => {
          config.index = index;
          this.highlightSelection(svg, index);
          this.showComputation(svg, index);
        });
        tokenContainer.on("mouseleave", () => {
          config.index = null;
          this.unhighlightSelection(svg);
          this.hideComputation(svg)
        });

        if (expanded) {
          tokenContainer.append('path')
            .attr("d", "M256 8C119 8 8 119 8 256s111 248 248 248 248-111 248-248S393 8 256 8zM124 296c-6.6 0-12-5.4-12-12v-56c0-6.6 5.4-12 12-12h264c6.6 0 12 5.4 12 12v56c0 6.6-5.4 12-12 12H124z")
            .classed("minus-sign", true)
            .attr("fill", this.getColor('icon'))
            .style('font-size', "17px")
            .style('font-weight', 900)
            .style('opacity', 0)
            .attr("dy", 17)
            .attr("transform", function (d, i) {
              let x = leftPos + 5;
              let y = i * BOXHEIGHT + HEADING_HEIGHT + 4;
              return "translate(" + x + " " + y + ")" +
                "scale(0.03 0.03) "
            });
          tokenContainer.append('rect')
            .attr("x", leftPos + 5)
            .attr("y", function (d, i) {
              return i * BOXHEIGHT + HEADING_HEIGHT + 4;
            })
            .style('opacity', 0)
            .attr("dy", 17)
            .attr("height", 16)
            .attr("width", 16)
            .on("click", (d, i) => {
              config.expanded = false;
              this.showCollapsed();
            })
            .on("mouseover", function (d, i) {
              d3.select(this).style("cursor", "pointer");
            })
            .on("mouseout", function (d, i) {
              d3.select(this).style("cursor", "default");
            })

        } else {
          tokenContainer.append('path')
            .attr("d", "M256 8C119 8 8 119 8 256s111 248 248 248 248-111 248-248S393 8 256 8zm144 276c0 6.6-5.4 12-12 12h-92v92c0 6.6-5.4 12-12 12h-56c-6.6 0-12-5.4-12-12v-92h-92c-6.6 0-12-5.4-12-12v-56c0-6.6 5.4-12 12-12h92v-92c0-6.6 5.4-12 12-12h56c6.6 0 12 5.4 12 12v92h92c6.6 0 12 5.4 12 12v56z")
            .classed("plus-sign", true)
            .attr("fill", this.getColor('icon'))
            .style('font-size', "17px")
            .style('font-weight', 900)
            .style('opacity', 0)
            .attr("dy", 17)
            .attr("transform", function (d, i) {
              let x = leftPos + 5;
              let y = i * BOXHEIGHT + HEADING_HEIGHT + 4;
              return "translate(" + x + " " + y + ")" +
                "scale(0.03 0.03) "
            });
          tokenContainer.append('rect')
            .attr("x", leftPos + 5)
            .attr("y", function (d, i) {
              return i * BOXHEIGHT + HEADING_HEIGHT + 4;
            })
            .style('opacity', 0)
            .attr("dy", 17)
            .attr("height", 16)
            .attr("width", 16)
            .on("click", (d, i) => {
              config.expanded = true;
              this.showExpanded();
            })
            .on("mouseover", function (d, i) {
              d3.select(this).style("cursor", "pointer");
            })
            .on("mouseout", function (d, i) {
              d3.select(this).style("cursor", "default");
            })
        }
      }
    },
    renderAttn (svg, start_pos, end_pos, expanded) {
      let attnMatrix = config.attention[config.filter].attn[config.layer][config.head];
      let attnContainer = svg.append("svg:g");
      attnContainer.selectAll("g")
        .data(attnMatrix)
        .enter()
        .append("g") // Add group for each source token
        .classed('attn-line-group', true)
        .attr("source-index", function (d, i) { // Save index of source token
          return i;
        })
        .selectAll("line")
        .data(function (d) { // Loop over all target tokens
          return d;
        })
        .enter() // When entering
        .append("line")
        .attr("x1", start_pos)
        .attr("y1", function (d) {
          let sourceIndex = +this.parentNode.getAttribute("source-index");
          return sourceIndex * BOXHEIGHT + HEADING_HEIGHT + BOXHEIGHT / 2;
        })
        .attr("x2", end_pos)
        .attr("y2", function (d, targetIndex) {
          return targetIndex * BOXHEIGHT + HEADING_HEIGHT + BOXHEIGHT / 2;
        })
        .attr("stroke-width", 2)
        .attr("stroke", this.getColor('attn'))
        .attr("stroke-opacity", function (d) {
          return d;
        });
    },
    highlightSelection (svg, index) {
      // let ref = this
      svg.select("#queries")
        .selectAll(".vector")
        .style("opacity", 1);
      svg.select("#queries")
        .selectAll(".vectorborder")
        .style("stroke", (d, i) => {
          return i == index ? this.getColor('connector') : this.getColor('vector_border');
        })
        .style("stroke-width", function (d, i) {
          return i == index ? 2 : 1;
        })
        ;
      svg.select("#queries")
        .select(".matrixborder")
        .style("stroke-opacity", 0);
      svg.select("#leftText")
        .selectAll(".highlight")
        .style("opacity", function (d, i) {
          return i == index ? 1.0 : 0.0;
        });
      svg.select("#leftText")
        .selectAll(".token")
        .style("fill", (d, i) => {
          return i == index ? this.getColor('selected_text') : this.getColor('text');
        });
      if (config.expanded) {
        svg.select("#leftText")
          .selectAll(".minus-sign")
          .style("opacity", function (d, i) {
            return i == index ? 1.0 : 0.0;
          })
      } else {
        svg.select("#leftText")
          .selectAll(".plus-sign")
          .style("opacity", function (d, i) {
            return i == index ? 1.0 : 0.0;
          })
      }
      svg.selectAll(".i-index")
        .text(index);
      svg.selectAll(".attn-line-group")
        .style("opacity", function (d, i) {
          return i == index ? 1.0 : 0.0;
        });
      svg.selectAll(".qk-line-group")
        .style("opacity", function (d, i) {
          return i == index ? 1.0 : 0.0;
        });
      if (config.bidirectional) {
        svg.select("#keys")
          .selectAll(".vectorborder")
          .style("stroke-opacity", 1);
      } else {
        svg.select("#product")
          .selectAll(".vector")
          .style("opacity", function (d, i) {
            return i <= index ? 1.0 : 0.0;
          });
        svg.select("#dotproducts")
          .selectAll("rect")
          .style("opacity", function (d, i) {
            return i <= index ? 1.0 : 0.0;
          });
      }
      svg.select('#hlines1')
        .selectAll(".horiz-line-group")
        .style("opacity", function (d, i) {
          return i == index ? 1.0 : 0.0;
        });
      svg.select('#hlines2')
        .selectAll(".horiz-line-group")
        .style("opacity", function (d, i) {
          return i == index ? 1.0 : 0.0;
        });
      svg.select('#hlines3')
        .selectAll(".horiz-line-group")
        .style("opacity", function (d, i) {
          return i == index ? 1.0 : 0.0;
        });
      svg.select('#key-vector-highlights')
        .selectAll(".vector-highlight-group")
        .style("opacity", function (d, i) {
          return i == index ? 1.0 : 0.0;
        });
      svg.select('#product-vector-highlights')
        .selectAll(".vector-highlight-group")
        .style("opacity", function (d, i) {
          return i == index ? 1.0 : 0.0;
        });
      svg.selectAll(".text-query-line")
        .style("opacity", function (d, i) {
          return i == index ? 1.0 : 0.0;
        })
    },
    unhighlightSelection (svg) {
      svg.select("#queries")
        .selectAll(".vector")
        .style("opacity", 1.0);
      svg.select("#queries")
        .selectAll(".vectorborder")
        .style("stroke", this.getColor('vector_border'))
        .style("stroke-width", 1);
      svg.select("#queries")
        .select(".matrixborder")
        .style("stroke-opacity", 1);
      svg.select("#leftText")
        .selectAll(".highlight")
        .style("opacity", 0.0);
      svg.select("#leftText")
        .selectAll(".token")
        .style("fill", this.getColor('text'));
      svg.select("#leftText")
        .selectAll(".minus-sign")
        .style("opacity", 0);
      svg.select("#leftText")
        .selectAll(".plus-sign")
        .style("opacity", 0);
      svg.selectAll(".i-index")
        .text("i");
      if (!config.expanded) {
        svg.selectAll(".attn-line-group")
          .style("opacity", 1)
      }
      svg.selectAll(".qk-line-group")
        .style("opacity", 0);
      svg.selectAll(".horiz-line-group")
        .style("opacity", 0);
      svg.selectAll(".vector-highlight-group")
        .style("opacity", 0);
      svg.selectAll(".text-query-line")
        .style("opacity", 0);

      if (!config.bidirectional) {
        svg.select("#keys")
          .selectAll(".vector")
          .style("opacity", 1);
        svg.select("#right_text")
          .selectAll("text")
          .style("opacity", 1);
      }
    },
    showComputation (svg, query_index) {
      let attnData = config.attention[config.filter];
      let query_vector = attnData.queries[config.layer][config.head][query_index];
      let keys = attnData.keys[config.layer][config.head];
      let att = attnData.attn[config.layer][config.head][query_index];

      let seq_len = keys.length;
      let productVectors = [];
      let dotProducts = [];
      for (let i = 0; i < seq_len; i++) {
        let key_vector = keys[i];
        let productVector = [];
        let dotProduct = 0;
        for (let j = 0; j < config.vectorSize; j++) {
          let product = query_vector[j] * key_vector[j];
          productVector.push(product); // Normalize to be on similar scale as query/key
          dotProduct += product;
        }
        productVectors.push(productVector);
        let scaledDotProduct = dotProduct / Math.sqrt(config.vectorSize)
        dotProducts.push(scaledDotProduct);
      }
      this.updateVectors(svg, 'product', productVectors);
      this.updateDotProducts(svg, dotProducts);
      this.updateSoftmax(svg, att);
      this.updateTextAttention(svg, att);
      svg.select("#placeholder").style("opacity", 0);

    },
    hideComputation (svg) {
      svg.select("#product").style("opacity", 0);
      svg.select("#dotproducts").style("opacity", 0);
      svg.select("#softmaxes").style("opacity", 0);
      svg.select('#rightText').selectAll("rect").style("opacity", 0);
      svg.select("#placeholder").style("opacity", 1);
    },
    updateVectors (svg, id, data) {
      let vectorContainer = svg.select('#' + id).style("opacity", 1);
      let vectors = vectorContainer.selectAll(".vector");
      vectors
        .data(data)
        .selectAll(".element") // loop over elements rects within each vector
        .data(function (d) {
          return d;
        }) // Bind them to array of elements from parent array
        .style("fill", (d) => {

          if (d >= 0) {
            return this.getColor('pos');
          } else {
            return this.getColor('neg');
          }
        })
        .attr("data-value", function (d) {
          return d
        })
        .style("opacity", function (d) {
          return Math.tanh(Math.abs(d) / 4);
        });
    },
    updateDotProducts (svg, dotProducts) {
      let vectorContainer = svg.select('#dotproducts').style("opacity", 1);
      vectorContainer.selectAll(".dotproduct")
        .data(dotProducts)
        .style("fill", (d) => {
          if (d >= 0) {
            return this.getColor('pos');
          } else {
            return this.getColor('neg');
          }
        })
        .style("fill-opacity", function (d) {
          return Math.tanh(Math.abs(d) / 4);
        })
        .style("stroke", (d) => {
          if (d >= 0) {
            return this.getColor('pos');
          } else {
            return this.getColor('neg');
          }
        })
        .style("stroke-opacity", function (d) {
          // Set border to slightly higher opacity
          return Math.max(Math.tanh(Math.abs(d) / 2), .35);
        })
        .attr("data-value", function (d) {
          return d
        })
    },
    updateSoftmax (svg, softmax) {
      let vectorContainer = svg.select('#softmaxes').style("opacity", 1);
      vectorContainer.selectAll(".softmax")
        .data(softmax)
        .attr("width", function (d) {
          return Math.max(d * SOFTMAX_WIDTH, 1);
        })
        .attr("data-value", function (d) {
          return d
        })

    },
    updateTextAttention (svg, attn) {
      let container = svg.select('#rightText');
      container.selectAll(".highlight")
        .data(attn)
        .style("opacity", function (d) {
          return d;
        })
    },
    showCollapsed () {
      if (config.index != null) {
        let svg = d3.select("#bertviz #vis");
        this.highlightSelection(svg, config.index);
      }
      d3.select("#bertviz #expanded").attr("visibility", "hidden");
      d3.select("#bertviz #collapsed").attr("visibility", "visible");
    },
    showExpanded () {
      if (config.index != null) {
        let svg = d3.select("#vis");
        this.highlightSelection(svg, config.index);
        this.showComputation(svg, config.index);
      }
      d3.select("#bertviz #expanded").attr("visibility", "visible");
      d3.select("#bertviz #collapsed").attr("visibility", "hidden")
    },
    getColor (name) {
      return PALETTE[config.mode][name]
    }
  }
}
</script>
<style lang="less" scoped>
.input_text {
  display: flex;
  flex-direction: row;
}

.message_choose_button {
  margin-left: 1%;
}

.dropdown-label {
  font-size: 14px;
  color: black;
}
</style>