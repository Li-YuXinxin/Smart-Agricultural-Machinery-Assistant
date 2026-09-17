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
      // 常见问题
      hotQuestions: [
        '绿萝叶子发黄怎么办？',
        '多肉多久浇一次水？',
        '室内养花适合什么光照？',
        '植物长虫了怎么处理？'
      ],
      // 打字动画
      cursorVisible: true,
    },

    /**
     * 长按复制消息
     */
    onCopyMessage(e) {
      const content = e.currentTarget.dataset.content
      if (!content) return
      wx.setClipboardData({
        data: content,
        success: () => {
          wx.showToast({ title: '已复制', icon: 'success' })
        }
      })
    },

    /**
     * 点击常见问题
     */
    onHotQuestion(e) {
      const question = e.currentTarget.dataset.q
      this.setData({ inputText: question })
      this.sendMessage()
    },

    /**
     * 清空对话
     */
    clearChat() {
      if (this.data.messages.length === 0) return
      wx.showModal({
        title: '清空对话',
        content: '确定要清空所有对话记录吗？',
        success: (res) => {
          if (res.confirm) {
            this.setData({ messages: [] })
          }
        }
      })
    },

    /**
     * 启动光标闪烁
     */
    _cursorTimer: null,
    startCursorBlink() {
      this._cursorTimer = setInterval(() => {
        this.setData({ cursorVisible: !this.data.cursorVisible })
      }, 500)
    },
    stopCursorBlink() {
      if (this._cursorTimer) {
        clearInterval(this._cursorTimer)
        this._cursorTimer = null
      }
    },

    /**
     * Markdown → HTML（轻量解析，支持常用格式）
     */
    mdToHtml(md) {
      if (!md) return ''
      let html = md
      // 转义HTML特殊字符
      html = html.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      // 代码块 ```...```
      html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, '<pre style="background:#f6f8fa;padding:16rpx;border-radius:8rpx;overflow-x:auto;font-size:24rpx;line-height:1.5;"><code>$2</code></pre>')
      // 标题 ### / ## / #
      html = html.replace(/^### (.+)$/gm, '<h3 style="font-size:30rpx;font-weight:700;margin:16rpx 0 8rpx;">$1</h3>')
      html = html.replace(/^## (.+)$/gm, '<h2 style="font-size:32rpx;font-weight:700;margin:20rpx 0 10rpx;">$1</h2>')
      html = html.replace(/^# (.+)$/gm, '<h1 style="font-size:36rpx;font-weight:700;margin:24rpx 0 12rpx;">$1</h1>')
      // 行内代码
      html = html.replace(/`([^`]+)`/g, '<code style="background:#f0f0f0;padding:2rpx 8rpx;border-radius:4rpx;font-size:24rpx;">$1</code>')
      // 粗体 + 斜体
      html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
      // 粗体
      html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      // 斜体
      html = html.replace(/\*(.+?)\*/g, '<em>$1</em>')
      // 有序列表（兼容 1. / 1、 / 1）以及前导空格）
      html = html.replace(/^\s*(\d+)\s*[\.、\)）]\s*(.+)$/gm, '<p style="margin:4rpx 0;padding-left:16rpx;">$1. $2</p>')
      // 无序列表（兼容 - / * / • 以及前导空格）
      html = html.replace(/^\s*[\-\*•]\s+(.+)$/gm, '<p style="margin:4rpx 0;padding-left:16rpx;">• $1</p>')
      // 分割线
      html = html.replace(/^---+$/gm, '<hr style="border:none;border-top:1rpx solid #eee;margin:16rpx 0;"/>')
      // 换行
      html = html.replace(/\n/g, '<br/>')
      return html
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
        content: '思考中……',
        htmlContent: '<span style="color:#999;">思考中……</span>',
        time: this.getTime()
      }

      // 保存ai回答到messages数组中，防止用户等待最终的回答时间过长
      // 同时确保即使超过token限制，也能显示前半部分的答案
      this.setData({messages:[...this.data.messages,aiMsg]})

      this.startCursorBlink()
      let timeoutId = this.checkChunkTimeout(null)
      // let timeoutId = setTimeout(() => {
      //   if (this.data.isStreaming) {
      //     // 思考超时,停止交互
      //     this.setData({ isStreaming: false })
      //     // 显示提示(持续3秒)
      //     wx.showToast({ title: '思考超时', icon: 'none', duration: 3000 })
      //     // 如果第一次接受的消息就超时
      //     const msgs = this.data.messages
      //     if (msgs.length && msgs[msgs.length-1].role === 'ai' && msgs[msgs.length-1].content === '') {
      //         // 删除最后一个ai消息,因为ai消息是空的
      //         msgs.pop()
      //         this.setData({ messages: msgs })
      //     }
      //   }
      // }, (180 * 1000))

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
        timeout: 200000,
        success: (res) => {
          // 因为流式交互会多次交互,所以不能再这里交互成功的处理逻辑
        },
        fail: (err) => {
          // 清空计时器
          clearTimeout(timeoutId)
          this.stopCursorBlink()
          // 交互状态清空
          this.setData({ isStreaming: false })
          // 删除空的ai占位消息
          const msgs = this.data.messages
          if (msgs.length && msgs[msgs.length-1].role === 'ai' && msgs[msgs.length-1].content === '思考中……') {
            msgs.pop()
            this.setData({ messages: msgs })
          }
          wx.showToast({ title: '请求失败，请重试', icon: 'none' })
          console.error('请求失败:', err)
        }
      })

      // 定义缓冲区变量
      let buffer = ''
      // 流式交互,每次接收到的数据都是一个chunk,需要拼接起来,然后解析
      requestTask.onChunkReceived((res) => {
        // 将接收到的数据转换为字符串
        const uint8Array = new Uint8Array(res.data)
        const str = this.uft8ArrayToString(uint8Array)
        // 拼凑来自后端的答案（片段）
        buffer += str
        const lines = buffer.split('\n')
        // 如果最后一行没有形成完整的行(\n结尾),保存该行,拼接下一次收到的信息,但如果最后一行是完整的行,则直接解析该行,并清空缓存
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data:')) {
            // 真正的信息在data:后面
            const jsonStr = line.substring(5).trim()
            if (!jsonStr) continue
            try {
              const data = JSON.parse(jsonStr)
              if (data.done) {
                // 后端发送完毕
                this.onDone(timeoutId)
              } else if (data.chunk !== undefined) {
                // 后续还有信息
                timeoutId = this.onChunk(data.chunk, timeoutId)
              } else if (data.error) {
                // 如果发生错误
                this.onError(data.error, timeoutId)
              }
            } catch(e) {
              console.error('解析JSON失败:', e)
            }
          }
        }
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

    /**
     * 后端发送完毕,停止交互
     */
    onDone(timeoutId){
      // 清空计时器
      clearTimeout(timeoutId)
      // 拆分来源文档信息
      const msgs = this.data.messages
      const last = msgs[msgs.length - 1]
      if (last && last.role === 'ai') {
        const idx = last.content.indexOf('\n来源文档:')
        if (idx !== -1) {
          last.source = last.content.substring(idx + 1)
          last.content = last.content.substring(0, idx)
        }
        // 最终渲染Markdown为HTML
        last.htmlContent = this.mdToHtml(last.content)
      }
      this.stopCursorBlink()
      // 交互状态清空
      this.setData({isStreaming:false, messages: msgs})
      // wx.showToast({
      //   title:'交互完成',
      //   icon:'none'
      // })
    },

    /**
     * 每次片段交互
     */
    onChunk(chunk, timeoutId){
      // 每次片段交互

      timeoutId = this.checkChunkTimeout(timeoutId)

      const msgs = this.data.messages
      const last = msgs[msgs.length-1]
      if (last.role === 'ai') {
        // ai消息,拼接内容
        if (last.content === '思考中……') {
          last.content = chunk
        } else {
          last.content += chunk
        }
        // 实时转换Markdown为HTML
        last.htmlContent = this.mdToHtml(last.content)
        this.setData({messages:msgs})
        // 调试：打印AI原始回复内容
        if (last.content.length < 200) console.log('AI原始内容:', last.content)
        // 滚动到底部
        this.scrollBottom()
      }

      // 为下一次片段计时
      return timeoutId
    },

    /**
     * 交互失败
     */
    onError(err, timeoutId){
      if (timeoutId)
        clearTimeout(timeoutId)

      this.stopCursorBlink()
      this.setData({isStreaming:false})
      wx.showToast({
        title:'交互失败',
        icon:'none'
      })
      console.error('交互失败:',err)
      // 如果第一次接受的消息就出错
      const msgs = this.data.messages
      if (msgs.length && msgs[msgs.length-1].role === 'ai' && msgs[msgs.length-1].content === '思考中……') {
          // 删除最后一个ai消息,因为ai消息还没有收到任何内容
          msgs.pop()
          this.setData({messages:msgs})
      }
    },

    /**
     * 预览来源文档
     */
    previewSource(e) {
      const source = e.currentTarget.dataset.source
      // 来源格式: "来源文档: E:\...\uuid_filename.pdf" 或 "来源文档: xxx.pdf,yyy.docx"
      const raw = source.replace('来源文档:', '').trim()
      const paths = raw.split(',').map(s => s.trim()).filter(Boolean)
      if (paths.length === 0) return
      // 提取文件名（去掉路径部分，只保留文件名）
      let fileName = paths[0]
      if (fileName.includes('\\')) fileName = fileName.split('\\').pop()
      if (fileName.includes('/')) fileName = fileName.split('/').pop()
      if (!fileName) return
      const downloadUrl = `${app.globalData.apiBase}/api/knowledge/download?name=${encodeURIComponent(fileName)}`
      wx.showLoading({ title: '加载中...' })
      wx.downloadFile({
        url: downloadUrl,
        success: (res) => {
          wx.hideLoading()
          if (res.statusCode === 200) {
            wx.openDocument({
              filePath: res.tempFilePath,
              showMenu: true,
              fail: () => { wx.showToast({ title: '无法预览此文件', icon: 'none' }) }
            })
          } else {
            wx.showToast({ title: '文件不存在或已被删除', icon: 'none' })
          }
        },
        fail: () => {
          wx.hideLoading()
          wx.showToast({ title: '下载失败', icon: 'none' })
        }
      })
    },

    checkChunkTimeout(timeoutId) {
      // 清空计时器
      if(timeoutId)
        clearTimeout(timeoutId)

      return setTimeout(() => {
        if (this.data.isStreaming) {
          // 思考超时,停止交互
          this.stopCursorBlink()
          this.setData({ isStreaming: false })
          // 显示提示(持续3秒)
          wx.showToast({ title: '思考超时', icon: 'none', duration: 3000 })
          // 如果第一次接受的消息就超时
          const msgs = this.data.messages
          if (msgs.length && msgs[msgs.length-1].role === 'ai' && msgs[msgs.length-1].content === '思考中……') {
              // 删除最后一个ai消息,因为ai消息还没有收到任何内容
              msgs.pop()
              this.setData({ messages: msgs })
          }
        }
      }, (180 * 1000))
    }
})
