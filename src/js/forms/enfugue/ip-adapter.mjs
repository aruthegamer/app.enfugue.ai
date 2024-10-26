/** @module forms/enfugue/ip-adapter */
import { FormView } from "../base.mjs";
import {
    SelectInputView,
    CheckboxInputView
} from "../input.mjs";

/**
 * The form class containing the strength slider
 */
class IPAdapterFormView extends FormView {
    /**
     * @var object All field sets and their config
     */
    static fieldSets = {
        "IP Adapter": {
            "ipAdapterModel": {
                "label": "Model",
                "class": SelectInputView,
                "config": {
                    "value": "default",
                    "options": {
                        "default": "Default",
                        "plus": "Plus",
                        "composition": "Composition",
                        "plus-face": "Plus Face",
                        "full-face": "Full Face",
                        "face-id": "Face ID",
                        "face-id-plus": "Face ID Plus"
                    },
                    "tooltip": "Which IP adapter model to use. 'Plus' will in general find more detail in the source image while considerably adjusting the impact of your prompt, and 'Plus Face' will ignore much of the image except for facial features. 'Composition' will ignore the subject and colors of the image, and instead only focus on how the image is composed. 'Full Face' is similar to 'Plus Face' but extracts more features.<br /><br />'Face ID' will extract features only from faces using the InsightFace model. 'Face ID Plus' will do this as well as the standard CLIP-based feature extraction."
                }
            },
            "ipAdapterPositional": {
                "label": "Use Positional IP Adapter",
                "class": CheckboxInputView,
                "config": {
                    "value": false,
                    "tooltip": "Whether to use a positional IP adapter. This will use tiled positional embeddings to help the model understand the spatial relationships between different parts of the image, but can be undesired in some cases. This is important for models like 'Composition' which uses the general shape of the image, and less important for models like 'FaceID' which isolate to faces."
                }
            }
        }
    };

    /**
     * @var bool Hide submit
     */
    static autoSubmit = true;
};

export { IPAdapterFormView };
