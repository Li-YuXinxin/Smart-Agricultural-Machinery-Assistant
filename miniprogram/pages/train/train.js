// pages/train/train.js
Page({

    /**
     * 页面的初始数据
     */
    data: {
      imageList:[], // 每张图片至少3个键值对 {path, name, size}
      label:'',     // 手动输入的分类，默认为空
      clearOld: false,  // 是否清除之前的数据
      isTrainning: false, // 是否正在微调
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
        const wsUrl = 'ws://${base}/api/train/ws'

        // 创建 WebSocket 连接
        const ws = wx.connectSocket({
          url: 'wsUrl',
          success:()=>{console.log('ws连接成功！')}
        })

        // 配置 ws 的回调函数，当服务器推送训练状态数据时触发
        ws.onMessage((res)=>{
          try {
              const get_data = JSON.parse(res.data) // 解析接收到的 JSON 数据
              this.setData ({status:get_data})      // 更新页面数据中的训练状态
              console.log('ws接收到数据：',data)
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
        count: Math.min(9, remaining)
      })

      // TODO！！！
    },

    /**
     * 清除图片
     */
    clearImages(){

    },

    /**
     * 开始训练
     */
    startFull(){

    },

    /**
     * 停止训练
     */
    stopTraining(){

    }
})