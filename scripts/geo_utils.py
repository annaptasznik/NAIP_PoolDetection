import arcpy
import numpy
import os

'''
Take windows of interest and organize into a file structure to prepare for raster clipping.
'''
def prepare_window_file():
    arcpy.env.overwriteOutput = True

    full_layer = "E:\desktop/2019_11_15\pool_ID\data\squares.gdb/test_street"


    with arcpy.da.SearchCursor(full_layer,'OBJECTID') as cursor:
        for row in cursor:
            sql_clause =  'OBJECTID = ' + str(row[0]) 
            #print(sql_clause)

            
            arcpy.MakeFeatureLayer_management(full_layer, "lyr") 
            arcpy.env.workspace = "E:\desktop/2019_11_15\pool_ID\data\output/test_street/street_sq.gdb/"
            out_feature_class = "id_" + str(row[0])
            arcpy.SelectLayerByAttribute_management ("lyr", 'NEW_SELECTION', sql_clause)
            arcpy.CopyFeatures_management("lyr", out_feature_class)
            #print("did " + str(row[0]))

'''
Clip NAIP raster image to the shape of the window of interest.
'''
def clip_raster():
    raster_chunks = "E:/desktop/2019_11_15/pool_ID/data/raster_chunks2"

    for filename in os.listdir(raster_chunks):
        if filename.endswith(".png"):
            sat = os.path.join(raster_chunks, filename)
            #print(sat)

            squares = "E:\desktop/2019_11_15\pool_ID\data\output/test_lawn/lawn_sq.gdb/"
            arcpy.env.workspace = squares

            feat = arcpy.ListFeatureClasses()
            for fc in feat:
                #print(fc)

                clipto = os.path.join(squares, fc)

                # get extent
                feature_class = clipto

                for row in arcpy.da.SearchCursor(feature_class, ['SHAPE@']):
                    extent = row[0].extent

                    l = extent.XMin #-12351031.354159
                    b = extent.YMin #3795290.600965
                    r = extent.XMax #-12351016.114061
                    t = extent.YMax #3795305.840963

                    extent = str(l) +' '+ str(b) + ' '+ str(r) + ' '+ str(t)

                    ls = float(arcpy.GetRasterProperties_management(sat, "LEFT").getOutput(0))
                    bs = float(arcpy.GetRasterProperties_management(sat, "BOTTOM").getOutput(0))
                    rs = float(arcpy.GetRasterProperties_management(sat, "RIGHT").getOutput(0))
                    ts = float(arcpy.GetRasterProperties_management(sat, "TOP").getOutput(0))
                    #rast_prop = res.getOutput(0)

                    if (ls < l and bs < b and rs > r and ts > t):
                        output = os.path.join("E:\desktop/2019_11_15\pool_ID\data\output/test_lawn/",fc + ".png")
                        if(os.path.isfile(output) == False):
                            
                            try:
                                # do clip
                                arcpy.Clip_management(sat,extent,output, clipto, "0", 'ClippingGeometry' , 'MAINTAIN_EXTENT')
                                #print("successfully clipped " + fc)
                            except:
                                pass

        