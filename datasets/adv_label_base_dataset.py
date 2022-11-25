from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.data.data_manager import DatasetWrapper
from dassl.utils import check_isfile, read_image


class AdvDatum(Datum):

    def __init__(self, impath='', label=0, domain=-1, classname='', adv_cn=''):
        super().__init__(impath, label, domain, classname)
        assert isinstance(adv_cn, str)
        self._adv_cn = adv_cn

    @property
    def adv_cn(self):
        return self._adv_cn


class AdvDatasetWrapper(DatasetWrapper):

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            'label': item.label,
            'domain': item.domain,
            'impath': item.impath,
            'adv_cn': item.adv_cn,
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = 'img'
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output['img'] = img

        if self.return_img0:
            output['img0'] = self.to_tensor(img0)

        return output
