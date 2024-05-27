var data2018 = ee.FeatureCollection('users/derricksaw123/data2018'),
    data2023 = ee.FeatureCollection('users/derricksaw123/data2023');

// Get Region of Interest (ROI), whichi is Perak boundary from Malaysia shp files
var roi = ee
    .FeatureCollection('FAO/GAUL/2015/level1')
    .filter(ee.Filter.eq('ADM1_NAME', 'Perak'));

// Visualize on Perak
Map.setCenter(101.07, 4.59, 10);
Map.addLayer(
    roi.style({
        color: 'red',
        fillColor: '00000000',
        width: 4,
    }),
    {},
    'Perak Boundary'
);

var visualization = {
    bands: ['B4', 'B3', 'B2'],
    min: 0.0,
    max: 3000,
    width: 4,
    color: 'red',
};

var yearsOfInterest = [2018, 2023];

var roiImages = yearsOfInterest.map(getROIImageWithNDRE);

var visualization = {
    bands: ['B4', 'B3', 'B2'],
    min: 0.0,
    max: 3000,
};

function getROIImageWithNDRE(year) {
    var startOfYear = ee.Date(year + '-01-01');
    var endOfYear = startOfYear.advance(1, 'year');

    // Cloud masking
    var s2 = ee
        .ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(startOfYear, endOfYear)
        .filterBounds(roi);

    var csPlus = ee
        .ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
        .filterDate(startOfYear, endOfYear)
        .filterBounds(roi);

    var QA_BAND = 'cs';
    var CLEAR_THRESHOLD = 0.5;

    var s2MaskedImageCollection = s2
        .filterDate(startOfYear, endOfYear)
        .filterBounds(roi)
        .linkCollection(csPlus, [QA_BAND])
        .map(function (img) {
            return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD));
        })
        .map(fillNull);

    // Fill null introduced by masking using focal mean
    function fillNull(image) {
        var mean = image.focalMean(1.5, 'square', 'pixels', 8);
        return image.unmask(mean);
    }

    // Create composite image across temporal and clip Perak out
    var image = s2MaskedImageCollection.median().clip(roi);

    var resampledImage = image
        .setDefaultProjection(
            s2MaskedImageCollection.select(1).first().projection()
        )
        .resample('bicubic');

    Map.addLayer(
        resampledImage,
        visualization,
        'Resampled Sentinel-2 Perak Image in ' + year
    );

    return resampledImage;
}

var dataset = [
    {
        year: 2018,
        image: roiImages[0],
        classData: data2018,
    },
    {
        year: 2023,
        image: roiImages[1],
        classData: data2023,
    },
];

// Bands for trainings
var bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'];

// Classification visualize args
var palette = [
    '#d63000',
    '#a604ff',
    '#1488ff',
    '#00d308',
    '#f3f706',
    '#bbbbbf',
];

var viz = {
    min: 0,
    max: 5,
    palette: palette,
};

// Mapping for class name
var classmap = ee.Dictionary({
    0: 'Residential',
    1: 'Industry',
    2: 'Water bodies',
    3: 'Forest',
    4: 'Vegetation',
    5: 'Primary Transport Network',
});

function model(args) {
    print('--------------Perak ' + args.year + '--------------');

    var data = args.image
        .sampleRegions({
            collection: args.classData,
            properties: ['class'],
            scale: 30,
            tileScale: 8,
        })
        .randomColumn({
            seed: 3033,
        });

    var training = data.filter(ee.Filter.lt('random', 0.7));
    var testing = data.filter(ee.Filter.gte('random', 0.7));
    var results = [];

    // Build CART model
    var cartClf = ee.Classifier.smileCart().train(training, 'class', bands);
    results.push(evaluateModel(args.image, testing, cartClf, 'CART', null));

    // Build RF model
    var rfClf = ee.Classifier.smileRandomForest({
        numberOfTrees: 50,
        seed: 3033,
    }).train(training, 'class', bands);
    results.push(
        evaluateModel(args.image, testing, rfClf, 'Random Forest', null)
    );

    // Build GB model
    var gbClf = ee.Classifier.smileGradientTreeBoost({
        numberOfTrees: 15,
        seed: 3033,
    }).train(training, 'class', bands);
    results.push(
        evaluateModel(args.image, testing, gbClf, 'Gradient Boosting', null)
    );

    var bestModel = ee
        .FeatureCollection(
            ee.List(results).map(function (dict) {
                dict = ee.Dictionary(dict);
                return ee.Feature(null, {
                    model: dict.get('model'),
                    accuracy: dict.getNumber('accuracy'),
                });
            })
        )
        .sort('accuracy', false)
        .first();

    print('Best Model:', bestModel.get('model'));

    var tunedModel = modelTuning(
        args.year,
        args.image,
        bestModel.model,
        training,
        testing
    );

    Map.addLayer(
        tunedModel.result,
        viz,
        'Land Use Classification of Perak ' + args.year + ' by RF'
    );

    return calculateArea(tunedModel.result, args.year);
}

// Function for performing classification, visualization, and evaluation
function evaluateModel(img, testing, clf, name, layerName) {
    var classificationResult = img.select(bands).classify(clf);

    // Model testing
    var testClassified = testing.classify(clf);
    var confusionMatrix = testClassified.errorMatrix('class', 'classification');
    var accuracy = confusionMatrix.accuracy();
    var producersAccuracy = confusionMatrix.producersAccuracy();
    var usersAccuracy = confusionMatrix.consumersAccuracy();
    var kappa = confusionMatrix.kappa();

    // Print accuracy metrics
    print('----------' + name);
    print('Overall Accuracy:', accuracy);
    print("Producer's Accuracy:", producersAccuracy);
    print("User's Accuracy:", usersAccuracy);
    print('Kappa Coefficient:', kappa);

    return {
        model: clf,
        result: classificationResult,
        accuracy: accuracy,
    };
}

// Since both best model are random forest
// Parameter tuning for best model (RF)
function modelTuning(year, image, model, training, testing) {
    print('Parameter tuning RF model');
    var numTreesList = ee.List.sequence(50, 250, 50); // Example range for number of trees
    var accuracies = numTreesList.map(function (numTrees) {
        var classifier = ee.Classifier.smileRandomForest({
            numberOfTrees: numTrees,
            seed: 3033,
        }).train({
            features: training,
            classProperty: 'class',
            inputProperties: bands,
        });

        // Classify the testing data
        var accuracy = testing
            .classify(classifier)
            .errorMatrix('class', 'classification')
            .accuracy();

        return ee.Feature(null, {
            accuracy: accuracy,
            numberOfTrees: numTrees,
        });
    });

    var accuracyFeatures = ee.FeatureCollection(accuracies);

    // Create chart
    var chart = ui.Chart.feature
        .byFeature(accuracyFeatures, 'numberOfTrees', 'accuracy')
        .setChartType('LineChart')
        .setOptions({
            title: 'Hyperparameter Tuning for the numberOfTrees Parameter',
            vAxis: {
                title: 'Validation Accuracy',
            },
            hAxis: {
                title: 'Number of Trees',
                gridlines: {
                    count: 15,
                },
            },
            lineSize: 3,
            pointSize: 0,
            legend: { position: 'none' },
        });

    print(chart);

    // Find the optimal number of trees
    var maxAccuracyFeature = accuracyFeatures.sort('accuracy', false).first();
    var optimalNumTrees = maxAccuracyFeature.get('numberOfTrees');
    print('Optimal Number of Trees:', optimalNumTrees);

    // Build tuned RF model
    var tunedRFClf = ee.Classifier.smileRandomForest({
        numberOfTrees: optimalNumTrees,
    }).train(training, 'class', bands);

    return evaluateModel(
        image,
        testing,
        tunedRFClf,
        'Tuned Random Forest',
        'Land Use Classification of Perak in ' + year + ' by Tuned RF'
    );
}

function calculateArea(result, year) {
    var classArea = ee.Dictionary(
        result
            .reduceRegion({
                reducer: ee.Reducer.frequencyHistogram(),
                geometry: roi,
                scale: 100,
                tileScale: 3,
                maxPixels: 250000000,
            })
            .values()
            .get(0)
    );

    var totalArea = classArea.values().reduce(ee.Reducer.sum());

    return ee.Feature(null, classArea.combine({ Year: year }));
}

var landuseFeatures = dataset.map(model);

print('--------------Final results--------------');

var combinedTable = ee.FeatureCollection(landuseFeatures);

var stackedBarChart = ui.Chart.feature
    .byFeature({
        features: combinedTable.select('[0-5]|Year'),
        xProperty: 'Year',
    })
    .setSeriesNames(classmap.values())
    .setChartType('ColumnChart')
    .setOptions({
        title: 'Land Use Area by Year',
        hAxis: { title: 'Year', titleTextStyle: { italic: false, bold: true } },
        vAxis: {
            title: 'Area (m²)',
            titleTextStyle: { italic: false, bold: true },
        },
        isStacked: 'absolute',
        colors: palette,
    });

print(stackedBarChart);

var palette = [
    '#00d308',
    '#a604ff',
    '#bbbbbf',
    '#d63000',
    '#f3f706',
    '#1488ff',
];

var pieChart2018 = ui.Chart.feature
    .byProperty({
        features: [
            landuseFeatures[0].select(classmap.keys(), classmap.values()),
        ],
    })
    .setChartType('PieChart')
    .setOptions({
        title: 'Land Use Distribution in 2018',
        colors: palette,
    });

var pieChart2023 = ui.Chart.feature
    .byProperty({
        features: [
            landuseFeatures[1].select(classmap.keys(), classmap.values()),
        ],
    })
    .setChartType('PieChart')
    .setOptions({
        title: 'Land Use Distribution in 2023',
        colors: palette,
    });

print(pieChart2018);
print(pieChart2023);
