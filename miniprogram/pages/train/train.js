// pages/train/train.js
const app = getApp()
Page({

    /**
     * 页面的初始数据
     */
    data: {
      imageList:[], // 每张图片至少3个键值对 {path, name, size}
      label:'',     // 手动输入的分类，默认为空
      clearOld: false,  // 是否清除之前的数据
      isTraining: false, // 是否正在微调
      fullLoading: false, // 按钮状态
      status:{status:'idle', result:null},
      ws:null,
      canStart:false
    },

    /**
     * 生命周期函数--监听页面加载
     */
    onLoad(options) {
      this.initWebSocket()  // 初始化 WebSocket
      this.updateCanStart()
    },

    /**
     * 初始化 ws
     */
    initWebSocket() {
      try {
        // 从全局配置中获取 API 基础地址，并去除协议前缀
        const base = app.globalData.apiBase.replace('http://', '').replace('https://', '')
        // ws 的 api 路径
        const wsUrl = `ws://${base}/api/train/ws`

        // 创建 WebSocket 连接
        const ws = wx.connectSocket({
          url: wsUrl,
          success:()=>{console.log('ws连接成功！')}
        })

        // 配置 ws 的回调函数，当服务器推送训练状态数据时触发
        ws.onMessage((res)=>{
          try {
              const get_data = JSON.parse(res.data) // 解析接收到的 JSON 数据
              console.log('ws接收到数据：', get_data)  // ← 看这里
              console.log('当前状态:', get_data.status)
              this.setData ({status:get_data})      // 更新页面数据中的训练状态
              console.log('ws接收到数据：', get_data)
          }catch(e){
              console.error('解析失败：',e)
          }
        })

        // 配置 WebSocket 连接关闭回调，当连接意外断开时，自动进行重连
        ws.onClose(()=>{
          console.log('ws 连接关闭, 3 秒后重连')
          setTimeout(()=>{
            this.initWebSocket()
          }, 3000)
        })

        // 配置 WebSocket 错误回调，记录连接过程中的错误信息
        ws.onError((res)=>{
          console.log('ws 连接错误', res)
        })

        // 将 WebSocket 实例保存到页面数据中
        this.setData({ws:ws})
      } catch(e) {
        console.error('ws 初始化失败', e)
      }
    },

    /**
     * 更新 canStart 的状态, 其结果会决定"开始训练"按钮是否可点击
     * 1. 图片列表不为空（已上传训练图片）
     * 2. 标签名称不为空（已填写训练标签）
     * 3. 当前不在训练中（防止重复提交）
     */
    updateCanStart(){
      const can = this.data.imageList.length > 0 && this.data.label.trim() !== '' && !this.data.isTraining
      this.setData({ canStart: can })
    },

    /**
     * 生命周期函数--监听页面显示
     */
    onShow() {
      this.updateCanStart()
    },

    /**
     * 生命周期函数--监听页面卸载
     */
    onUnload() {
      // 手动关闭并释放WebSocket连接，释放资源
      if (this.data.ws) {
        this.data.ws.close()
      }
    },

    /**
     * 选择图片
     */
    chooseImages(){
      // 计算还可以选择多少张图片（最多支持9张）
      const remaining = 9 - this.data.imageList.length

      // 如果已选图片已达到上限（9张），提示用户并终止操作
      if (remaining <= 0) {
        wx.showToast({
          title: '最多只能选择9张图片',
          icon: 'none'
        })
        return
      }

      // 微信小程序规定每次最多只能选择9张图片
      wx.chooseMedia({
        count: Math.min(9, remaining),
        mediaType: ['image'],                    // 只允许选择图片
        sourceType: ['album', 'camera'],         // 可从相册选择或拍照
        success: (res) => {
          // 获得用户选择的图片，转换为统一格式
          const newFiles = res.tempFiles.map(f => ({
            path: f.tempFilePath,                // 本地临时文件路径
            name: f.tempFilePath.split('/').pop() || 'image.jpg',  // 提取文件名
            size: f.size                         // 文件大小
          }))

          // 获取当前已选图片列表
          const list = this.data.imageList

          // 截取新加入的图片（最多不超过剩余容量）
          const added = newFiles.slice(0, remaining)

          // 如果选择的图片超出剩余容量，提示用户
          if (newFiles.length > remaining) {
            wx.showToast({
              title: `仅能添加前${remaining}张`,
              icon: 'none'
            })
          }

          // 更新图片列表，合并已有图片和新添加的图片，并同步更新按钮状态
          this.setData({
            imageList: [...list, ...added]
          }, this.updateCanStart)
        }
      })
    },

    /**
     * 清除所有已选图片
     * 将图片列表置为空数组，并同步更新"开始训练"按钮状态
     */
    clearImages(){
      this.setData({ imageList: [] }, this.updateCanStart)
    },

    /**
     * 删除指定索引的图片
     * 
     */
    removeImage(e) {
      // 从事件对象中获取被点击图片的索引
      const index = e.currentTarget.dataset.index
      // 获取当前图片列表
      const list = this.data.imageList
      // 根据索引删除对应的图片（原地修改数组）
      list.splice(index, 1)
      // 更新图片列表，并同步更新"开始训练"按钮状态
      this.setData({ imageList: list }, this.updateCanStart)
    },

    /**
     * 标签输入事件处理
     * 监听用户输入分类标签，实时保存到data中
     */
    onLabelInput(e) {
      // 将用户输入的分类label名称保存到data，并同步更新"开始训练"按钮状态
      this.setData({ label: e.detail.value }, this.updateCanStart)
    },

    /**
     * 重新训练复选框状态变化事件处理
     */
    onClearOldChange(e) {
      this.setData({ clearOld: e.detail.value > 0 }, this.updateCanStart)
    },

    /**
     * 开始训练
     */
    startFull(){
      // 如果当前不能训练，则直接返回（防止用户重复点击或条件不满足）
      if (!this.data.canStart) return

      // 设置加载状态和训练状态
      this.setData({
        fullLoading: true,
        isTraining: true
      })

      // 模态窗口显示载入信息
      wx.showLoading({
        title: '准备图片中...',
        mask: true
      })

      // 文件系统管理器，用于读取图片文件
      const fs = wx.getFileSystemManager()

      // 遍历图片列表，创建图片读取任务
      const tasks = this.data.imageList.map((item) => {
        return new Promise((resolve, reject) => {
          // 读取图片文件为base64格式
          fs.readFile({
            filePath: item.path,
            success: (res) => {
              const base64 = wx.arrayBufferToBase64(res.data)
              const ext = item.path.split('.').pop() || 'jpg'
              // 统一图片扩展名
              const mineType = `image/${ext === 'jpg' ? 'jpeg' : ext}`
              // 拼接并返回base64的数据
              resolve(`data:${mineType};base64,${base64}`)
            },
            fail: reject
          })
        })
      })

      Promise.all(tasks).then((images)=>{
        // 关闭模态窗口
        wx.hideLoading()
        // 重新打开一个模态窗口
        wx.showLoading({
          title: '开始训练……',
          mask:true
        })

        // 发起HTTP POST请求，开始模型训练
        wx.request({
          url: `${app.globalData.apiBase}/api/train/finetune`,
          method: 'POST',
          data: {
            label: this.data.label,
            images: images,
            clear_old: this.data.clearOld
          },
          success: (res) => {
            // 请求成功，判断状态码
            if (res.statusCode === 200) {
              wx.showToast({ title: '训练成功', icon: 'success' })
              // 清空数据，为下一次训练做准备
              this.setData({imageList:[],label:'',clearOld:false,isTraining:false},this.updateCanStart)
            } else {
              // 解析错误信息并展示
              const get_data = res.data
              wx.showToast({ title: get_data.detail || '训练失败', icon: 'none' })
            }
          },
          fail: () => {
            // 网络请求失败
            wx.hideLoading()
            wx.showToast({ title: '网络错误', icon: 'none' })
            this.setData({isTraining:false})
          }
        })
      }).catch((err)=>{
        wx.hideLoading()
        wx.showToast({
          title: '图片读取/转换失败',
          icon:'none'
        })
        this.setData({isTraining:false})    
      })
    },

    /**
     * 停止训练
     */
    stopTraining(){
      // 如果当前状态不是运行中（running），则不执行停止操作
      if (this.data.status.status != 'running') return

      // 发起HTTP POST请求，请求停止训练
      wx.request({
        url: `${app.globalData.apiBase}/api/train/stop`,
        method: 'POST',
        success: () => {
          wx.showToast({ title: '停止训练成功', icon: 'none' })
          this.setData({ 
            isTraining: false, 
            fullLoading: false 
          }, this.updateCanStart)
        },
        fail: () => {
          wx.showToast({ title: '停止训练失败', icon: 'none' })
        }
      })
    }
})