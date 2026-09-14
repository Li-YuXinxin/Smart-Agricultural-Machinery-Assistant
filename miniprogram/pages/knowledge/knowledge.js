// pages/knowledge/knowledge.js
const app = getApp()
Page({

    /**
     * 页面的初始数据
     */
    data: {
      loading:false,
      result:null
    },

    /**
     * 选择文件
     */
    chooseFile(){
      wx.chooseMessageFile({
        count:1,
        type:'file',
        extension:['txt', 'doc','docx','pdf'],
        success:(res)=>{
          const file = res.tempFiles[0]
          wx.showLoading({title:'上传中...'})
          wx.uploadFile({
            url:`${app.globalData.apiBase}/api/knowledge/upload`,
            filePath:file.path,
            name: 'file',
            success:(res)=>{
              try {
                const result = JSON.parse(res.data)
                //保存结果
                this.setData({result:result})
                wx.hideLoading()
                wx.showToast({
                  title:'上传成功',
                  icon:'success'
                })
              } catch(e){
                this.setData({result:null})
                wx.hideLoading()
                wx.showToast({
                  title:'上传失败',
                  icon:'none'
                })
              }
            },
            fail:(res)=>{
              wx.showToast({
                title: '无法与服务器通信',
                icon: 'none'
              })
            }
          })
        }
      })
    }
})