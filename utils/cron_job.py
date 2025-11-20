


class CronJob():

    def __init__(self, url):
        self.url = url

    def fetch(self):
        # 1. Nếu data chưa có trong cache thì fetch
        # 2. Nếu là tháng hiện tại chưa có trong cache -> fetch lại csv của tháng hiện tại
        # 3. Nếu đã có data của các tháng trong quá khứ -> trong cache đã có rồi không cần fetch lại
        # 4. Nếu có data của tháng hiện tại trong cache -> cần fetch lại vì url này 1 tiếng cập nhập 1 lần
        NotImplementedError
