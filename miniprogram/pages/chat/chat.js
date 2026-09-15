// pages/chat/chat.js
const app = getApp()

Page({
    /**
     * 页面的初始数据
     */
    data: {
      messages: [],         // 消息列表 {role, content, time}
      inputText: '',        // 用户输入的内容
      isStreaming: false,   // 是否正在等待AI回复
      scrollIntoView: '',   // 滚动定位到的元素id
    },

    /**
     * 输入框内容改变时，更新inputText
     */
    onInput(e) {
      this.setData({ inputText: e.detail.value })
    },
    
    /**
     * 将js的时间戳转换为自定义格式的时间字符串(xx小时:xx分)
     */
    getTime() {
      const now = new Date()
      // const pre = `${now.getFullYear()}-${(now.getMonth() + 1).toString().padStart(2, '0')}-${(now.getDay() + 1).toString().padStart(2, '0')}`
      return `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`
    },
    
    /**
     * 获得页面的输入(用户的问题)
     */
    sendMessage() {
      const text = this.data.inputText.trim()
      // 如果输入为空或者当前正处于问答状态，则不发送
      if (!text || this.data.isStreaming) return
      // 将用户的问题封装为消息对象
      const userMsg = {
        role: 'user',
        content: text,
        time: this.getTime()
      }
      // 保存用户的问题到messages数组中, 同时将isStreaming设置为true
      this.setData({
        messages: [...this.data.messages, userMsg],
        inputText: '',
        isStreaming: true
      })

      // 滚动到底部
      this.scrollBottom()

      // ai回答部分
      const aiMsg = {
        role: 'ai',
        content: '',
        time: this.getTime()
      }

      let timeoutId = setTimeout(() => {
        if (this.data.isStreaming) {
          // 思考超时,停止交互
          this.setData({ isStreaming: false })
          // 显示提示(持续3秒)
          wx.showToast({ title: '思考超时', icon: 'none', duration: 3000 })
        }
      }, (180 * 1000))

      const requestTask = wx.request({
        url: `${app.globalData.apiBase}/api/chat/stream`,
        method: 'POST',
        data: {
          message: text,
          temperature:0.7,
          max_tokens:512
        },
        header: {
          'Content-Type': 'application/json'
        },
        // 流式交互,需要设置chunked编码,还需要设置responseType为arraybuffer(缓冲区)
        enableChunked: true,
        responseType: 'arraybuffer',
        success: (res) => {
          // 因为流式交互会多次交互,所以不能再这里交互成功的处理逻辑
        },
        fail: (err) => {
          // 清空计时器
          clearTimeout(timeoutId)
          // 交互状态清空
          this.setData({ isStreaming: false })
          wx.showToast({ title: '交互失败', icon: 'none' })
          console.error('交互失败:', err)
        }
      })

      // 定义缓冲区变量
      let buffer = ''
      // 流式交互,每次接收到的数据都是一个chunk,需要拼接起来,然后解析
      requestTask.onChunkReceived((res) => {
        // 将接收到的数据转换为字符串
        const uint8Array = new Uint8Array(res.data)
        const str = this.uft8ArrayToString(uint8Array)
      })
    },

    uft8ArrayToString(uint8Array) {
      let out = ''
      let i = 0
      let len = uint8Array.length
      let c, char2, char3

      while (i < len) {
        c = uint8Array[i++]
        switch (c >> 4) {
          case 0:
          case 1:
          case 2:
          case 3:
          case 4:
          case 5:
          case 6:
          case 7:
            // ascii->char
            out += String.fromCharCode(c)
            break
          case 12:
          case 13:
            char2 = uint8Array[i++]
            out += String.fromCharCode((c & 0x1F) << 6 | (char2 & 0x3F))
            break
          case 14:
            char2 = uint8Array[i++]
            char3 = uint8Array[i++]
            out += String.fromCharCode((c & 0x0F) << 12 | (char2 & 0x3F) << 6 | (char3 & 0x3F))
            break
          default:
            break
        }
      }
      return out
    },
    
    /**
     * 滚动到底部(让滑块和底部的view对齐)
     */
    scrollBottom() {
      this.setData({ scrollIntoView: 'bottom' })
    },
})
