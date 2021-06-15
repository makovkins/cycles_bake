#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdio.h>
#include <typeinfo>

#include "device/device.h"
#include "graph/node.h"
#include "render/bake.h"
#include "render/buffers.h"
#include "render/camera.h"
#include "render/geometry.h"
#include "render/graph.h"
#include "render/integrator.h"
#include "render/light.h"
#include "render/mesh.h"
#include "render/nodes.h"
#include "render/object.h"
#include "render/scene.h"
#include "render/session.h"
#include "render/shader.h"
#include "render/volume.h"
#include "util/util_path.h"

#include "OpenImageIO\imagebuf.h"
#include "OpenImageIO\imagebufalgo.h"
#include "OpenImageIO\imageio.h"

using namespace ccl;

#define DEG2RADF(_deg) ((_deg) * (float)(M_PI / 180.0))


typedef struct BakePixel
{
    int primitive_id, object_id;
    int seed;
    float uv[2];
    float du_dx, du_dy;
    float dv_dx, dv_dy;
} BakePixel;

typedef struct BakeImage
{
    struct Image *image;
    int width;
    int height;
    size_t offset;
} BakeImage;

/* span fill in method, is also used to localize data for zbuffering */
typedef struct ZSpan
{
    int rectx, recty; /* range for clipping */

    int miny1, maxy1, miny2, maxy2;             /* actual filled in range */
    const float *minp1, *maxp1, *minp2, *maxp2; /* vertex pointers detect min/max range in */
    float *span1, *span2;
} ZSpan;

typedef struct BakeDataZSpan
{
    BakePixel *pixel_array;
    int primitive_id;
    BakeImage *bk_image;
    ZSpan *zspan;
    float du_dx, du_dy;
    float dv_dx, dv_dy;
} BakeDataZSpan;

typedef struct MyBakeData
{
	int width, height;
	ImageBuf buffer_combined;
    ImageBuf buffer_primitive_id;
    ImageBuf buffer_differencial;
    ImageBuf buffer_light;
    ImageBuf buffer_uv;

    void init(int in_width, int in_height)
    {
        width = in_width;
        height = in_height;
    	
    	buffer_combined.reset(ImageSpec(width, height, 4, TypeDesc(TypeDesc::FLOAT)));
        buffer_primitive_id.reset(ImageSpec(width, height, 4, TypeDesc(TypeDesc::FLOAT)));
        buffer_differencial.reset(ImageSpec(width, height, 4, TypeDesc(TypeDesc::FLOAT)));
        buffer_light.reset(ImageSpec(width, height, 1, TypeDesc(TypeDesc::FLOAT)));
        buffer_uv.reset(ImageSpec(width, height, 2, TypeDesc(TypeDesc::FLOAT)));
    }

    void set(int x, int y, int seed, int primitive_id, float2 uv, float du_dx, float du_dy, float dv_dx, float dv_dy)
    {
        float prim[4] = {__int_as_float(seed), __int_as_float(primitive_id), uv.x, uv.y};
        buffer_primitive_id.setpixel(x, y, prim);
        float diff[4] = {du_dx, du_dy, dv_dx, dv_dy};
        buffer_differencial.setpixel(x, y, diff);        
        buffer_uv.setpixel(x, y, &uv.x);
    }
	
} MyBakeData;


MyBakeData bake_data;


void read_tile_from_buffer(RenderTile& rtile)
{
    RenderBuffers *buffers = rtile.buffers;
    BufferParams &params = rtile.buffers->params;
    int x = rtile.x;
    int y = rtile.y;
    int w = rtile.w;
    int h = rtile.h;

	for (Pass& pass : params.passes)
	{
        vector<float> pixels(w * h * pass.components);
        ROI rect = ROI(x, x + w, y, y + h);
		
        if (pass.type == PASS_COMBINED)
        {
	        bake_data.buffer_combined.get_pixels(rect, TypeDesc::FLOAT, &pixels[0]);
            buffers->set_pass_rect(pass.type, pass.components, &pixels[0], rtile.num_samples);
        }
        else if (pass.type == PASS_BAKE_PRIMITIVE)
        {
            bake_data.buffer_primitive_id.get_pixels(rect, TypeDesc::FLOAT, &pixels[0]);
            buffers->set_pass_rect(pass.type, pass.components, &pixels[0], rtile.num_samples);
        }
        else if (pass.type == PASS_BAKE_DIFFERENTIAL)
        {
	        bake_data.buffer_differencial.get_pixels(rect, TypeDesc::FLOAT, &pixels[0]);
            buffers->set_pass_rect(pass.type, pass.components, &pixels[0], rtile.num_samples);
        }
        else if (pass.type == PASS_LIGHT)
        {
            //bake_data.buffer_light.get_pixels(rect, TypeDesc::FLOAT, &pixels[0]);
            //buffers->set_pass_rect(pass.type, pass.components, &pixels[0], rtile.num_samples);
        }
        else
            assert(false);		
	}
}

void write_tile_to_buffer(RenderTile& rtile)
{
    RenderBuffers *buffers = rtile.buffers;
    BufferParams &params = rtile.buffers->params;
    int x = rtile.x;
    int y = rtile.y;
    int w = rtile.w;
    int h = rtile.h;

    vector<float> pixels(w * h * 4);
    float exposure = 1.0;
    if (!buffers->get_pass_rect("Combined", exposure, rtile.sample, 4, &pixels[0]))
        memset(&pixels[0], 0, pixels.size() * sizeof(float));

    OIIO::ROI rect = ROI(x, x + w, y, y + h);
    bake_data.buffer_combined.set_pixels(rect, TypeDesc::FLOAT, &pixels[0]);
}

void save_image(string file_name, ImageBuf& render_buffer)
{    
    ImageOutput::unique_ptr out = ImageOutput::create(file_name);
    if (!out)
        return;

	size_t width = render_buffer.spec().width;
    size_t height = render_buffer.spec().height;        

    ImageSpec spec(width, height, 4, TypeDesc::UINT8);
    if (!out->open(file_name, spec))
        return;

    ROI full_rect = ROI(0, width, 0, height);
    vector<float> result(width * height * 4);
    render_buffer.get_pixels(full_rect, TypeDesc::FLOAT, &result[0]);
    int scanlinesize = width * 4 * sizeof(result[0]);

    if (!out->write_image(TypeDesc::FLOAT,
                     (uchar *)&result[0] + (height - 1) * scanlinesize,
                     AutoStride,
                     -scanlinesize,
                     AutoStride))
    {
        std::cout << "Failed to write image '" << file_name << "'." << std::endl;
    }

    out->close();
}

void add_light(Scene *scene)
{    		    
    ShaderGraph *graph = new ShaderGraph();
	
    ValueNode *valueNode = graph->create_node<ValueNode>();
    ShaderNode *vn = graph->add(valueNode);
    valueNode->set_value(750);
	
    ColorNode *colorNode = graph->create_node<ColorNode>();
    ShaderNode *cn = graph->add(colorNode);
    colorNode->set_value(one_float3());    

	EmissionNode *emNode = graph->create_node<EmissionNode>();
	ShaderNode *emn = graph->add(emNode);
    emn->input("Color")->set(make_float3(1.0, 1.0, 1.0));
	
    graph->connect(cn->output("Color"), emn->input("Color"));
    graph->connect(vn->output("Value"), emn->input("Strength"));
	ShaderNode *out = graph->output();
    graph->connect(emn->output("Emission"), out->input("Surface"));

	Shader *shader = new Shader();
    shader->name = "lightShader";
    shader->set_graph(graph);
    shader->tag_update(scene);
    scene->shaders.push_back(shader);

	Light *light = new Light();
	light->set_shader(shader);
    light->set_light_type(LIGHT_POINT);
    Transform tfm = transform_identity();
    float3 co = make_float3(0, -3, 3);
    light->set_co(transform_point(&tfm, co));
    light->set_size(0.5f);
    light->tag_update(scene);
    scene->lights.push_back(light);
}

int add_cube_shader(Scene *scene)
{
    Shader *shader = new Shader();
    ShaderGraph *graph = new ShaderGraph();

    DiffuseBsdfNode *node = graph->create_node<DiffuseBsdfNode>();
    node->set_color(make_float3(1.0, 0.0, 0.0));
    graph->add(node);

    ShaderNode *out = graph->output();
    graph->connect(node->output("BSDF"), out->input("Surface"));
	
    shader->set_graph(graph);
    shader->tag_update(scene);
    scene->shaders.push_back(shader);
    return scene->shaders.size() - 1;
}

void add_mesh(Scene *scene, int shader_id)
{
    Geometry *geom = scene->create_node<Mesh>();

    array<Node *> used_shaders;
    used_shaders.push_back_slow(scene->shaders[shader_id]);
    geom->set_used_shaders(used_shaders);

    Object *object = new Object();
    object->set_geometry(geom);
    object->name = "cube";
    Transform tfm = transform_identity();
    tfm = tfm * transform_scale(0.5f, 0.5f, 0.5f)  * transform_euler(make_float3(0, 0, -45)) *
          transform_translate(make_float3(0, 0, 1));
    object->set_tfm(tfm);
    scene->objects.push_back(object);

    Mesh *mesh = static_cast<Mesh *>(geom);

    static float pArray[24] = {-1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1,
                               -1, -1, 1,  1, -1, 1,  -1, 1, 1,  1, 1, 1};
    int pArrayLength = 24;
    static int nvertsArrayLength = 6;
    static int nvertsArray[6] = {4, 4, 4, 4, 4, 4};
    static int vertsArray[24]{0, 2, 3, 1, 0, 1, 5, 4, 0, 4, 6, 2,
                              1, 3, 7, 5, 2, 6, 7, 3, 4, 5, 7, 6};
    int vertsArrayLength = 24;

    array<float3> P;
    vector<float> UV;
    vector<int> verts, nverts;

    size_t copyNum = 1;
    float cube_size = 1.5;

    for (size_t c = 0; c < copyNum; c++)
    {
        for (size_t i = 0; i < pArrayLength; i += 3)
        {
            P.push_back_slow(make_float3(
                cube_size * pArray[i + 0], cube_size * pArray[i + 1], cube_size * pArray[i + 2]));
        }
        for (int i = 0; i < vertsArrayLength; i++)
        {
            verts.push_back(vertsArray[i]);
        }
        for (int i = 0; i < nvertsArrayLength; i++)
        {
            nverts.push_back(nvertsArray[i]);
        }
    }

	size_t num_triangles = 0;
    for (size_t i = 0; i < nverts.size(); i++)
    {
        num_triangles += nverts[i] - 2;
    }
    mesh->reserve_mesh(P.size(), num_triangles);
    mesh->set_verts(P);

    // create triangles
    int index_offset = 0;

    for (size_t i = 0; i < nverts.size(); i++)  // iterate over polygons
    {
        for (int j = 0; j < nverts[i] - 2; j++)  // for each polygon by n-2
        {
            int v0 = verts[index_offset];
            int v1 = verts[index_offset + j + 1];
            int v2 = verts[index_offset + j + 2];
            mesh->add_triangle(v0, v1, v2, 0, false);
        }

        index_offset += nverts[i];
    }    

    /*
    {
        int numverts = P.size();
        int numfaces = nverts.size();
        int numtris = 0;
        int numcorners = 0;
        int numngons = 0;
        for (size_t i = 0; i < nverts.size(); i++)
        {
            numcorners += nverts[i];
            numngons += nverts[i] == 4 ? 0 : 1;
        }

        mesh->reserve_subd_faces(numfaces, numngons, numcorners);
        mesh->reserve_mesh(numverts, numtris);

        // set vertices
        for (size_t i = 0; i < P.size(); i++)
        {
            mesh->add_vertex(P[i]);
        }

        AttributeSet &attributes = mesh->subd_attributes;

        if (mesh->need_attribute(scene, ATTR_STD_GENERATED))
        {
            Attribute *attr = attributes.add(ATTR_STD_GENERATED);
            attr->flags |= ATTR_SUBDIVIDED;
            memcpy(attr->data_float3(),
                   mesh->get_verts().data(),
                   sizeof(float3) * mesh->get_verts().size());
        }

        // faces
        vector<int> vi;

        size_t index = 0;
        for (size_t p = 0; p < nverts.size(); p++)
        {
            int n = nverts[p];
            int shader = 0;
            bool smooth = false;

            vi.resize(n);
            for (int i = 0; i < n; i++)
            {
                vi[i] = verts[index];
                index++;
            }

            mesh->add_subd_face(&vi[0], n, shader, smooth);
        }

        // creases
        mesh->reserve_subd_creases(0);
        mesh->set_subd_dicing_rate(1.0);
        mesh->set_subd_max_level(4);
        mesh->set_subd_objecttoworld(object->get_tfm());

        scene->dicing_camera->update(scene);

        mesh->set_subdivision_type(Mesh::SUBDIVISION_CATMULL_CLARK);
    }*/
}

int add_plain_shader(Scene *scene)
{    
    ShaderGraph *graph = new ShaderGraph();

    DiffuseBsdfNode *node = graph->create_node<DiffuseBsdfNode>();
    node->set_color(make_float3(1.0, 1.0, 1.0));
    graph->add(node);

    ShaderNode *out = graph->output();
    graph->connect(node->output("BSDF"), out->input("Surface"));

/*    TextureCoordinateNode *texture_node = graph->create_node<TextureCoordinateNode>();
    graph->add(texture_node);
    SeparateXYZNode *separate_node = graph->create_node<SeparateXYZNode>();
    graph->add(separate_node);
    graph->connect(texture_node->output("UV"), separate_node->input("Vector"));
    CombineRGBNode *combine_node = graph->create_node<CombineRGBNode>();
    graph->add(combine_node);
    graph->connect(separate_node->output("X"), combine_node->input("R"));
    graph->connect(separate_node->output("Y"), combine_node->input("G"));
    graph->connect(separate_node->output("Z"), combine_node->input("B"));

    graph->connect(combine_node->output("Image"), node->input("Color"));
*/
	Shader *shader = new Shader();
    shader->set_graph(graph);
    shader->tag_update(scene);
    scene->shaders.push_back(shader);
	
    return scene->shaders.size() - 1;
}

void add_plane(Scene *scene, int shader_id)
{
    Mesh *mesh = scene->create_node<Mesh>();

    array<Node *> used_shaders;
    used_shaders.push_back_slow(scene->shaders[shader_id]);
    mesh->set_used_shaders(used_shaders);

    Object *object = new Object();
    object->set_geometry(mesh);
    object->name = "plane";
	
    Transform tfm = transform_identity();
    tfm = tfm * transform_euler(make_float3(0, 0, 0)) * transform_translate(make_float3(0, 0, -0.5));
    object->set_tfm(tfm);
    scene->objects.push_back(object);    

    mesh->reserve_mesh(4, 2);  // on plane 4 vertices, 2 triangles
    float plane_radius = 10.0f;
    array<float3> vertices(4);
    vertices[0] = make_float3(plane_radius, plane_radius, 0);
    vertices[1] = make_float3(-plane_radius, plane_radius, 0);
    vertices[2] = make_float3(-plane_radius, -plane_radius, 0);
    vertices[3] = make_float3(plane_radius, -plane_radius, 0);
    mesh->set_verts(vertices);

    // triangles
    mesh->add_triangle(0, 1, 2, 0, false);
    mesh->add_triangle(0, 2, 3, 0, false);

    // uvs
    Attribute *uv_attr = mesh->attributes.add(ATTR_STD_UV, ustring("std_uv"));
    float2 *default_uv = uv_attr->data_float2();
    default_uv[0] = make_float2(1.0, 1.0);
    default_uv[1] = make_float2(0.0, 1.0);
    default_uv[2] = make_float2(0.0, 0.0);
    default_uv[3] = make_float2(1.0, 1.0);
    default_uv[4] = make_float2(0.0, 0.0);
    default_uv[5] = make_float2(1.0, 0.0);
}

void setup_scene(Scene *scene, size_t width, size_t height)
{
    float3 translate = make_float3(0.0, -6, 1.0);
    float4 rotate = make_float4(-90.0f, 1.0f, 0.0f, 0.0f);

    Transform tfm = transform_identity();
    tfm = tfm * transform_translate(translate);
    tfm = tfm * transform_rotate(DEG2RADF(rotate.x), make_float3(rotate.y, rotate.z, rotate.w));
    rotate = make_float4(180.0f, 0.0f, 0.0f, 1.0f);
    tfm = tfm * transform_rotate(DEG2RADF(rotate.x), make_float3(rotate.y, rotate.z, rotate.w));

	scene->camera->set_matrix(tfm);

    // the same for dicing camera
    scene->dicing_camera->set_matrix(tfm);

    add_light(scene);

    bool use_volume = false;
    bool use_subdiv = false;
    int shader_id = add_cube_shader(scene);
    add_mesh(scene, shader_id);
    add_plane(scene, add_plain_shader(scene));

    scene->integrator->set_method(Integrator::BRANCHED_PATH);
    scene->integrator->set_subsurface_samples(2);
    scene->integrator->set_volume_samples(4);
    scene->integrator->set_volume_step_rate(1.0);
    scene->integrator->tag_update(scene, Integrator::UPDATE_ALL);
    scene->camera->set_full_width(width);
    scene->camera->set_full_height(height);
    scene->camera->set_camera_type(CAMERA_PERSPECTIVE);
    scene->camera->compute_auto_viewplane();
    scene->dicing_camera->set_full_width(width);
    scene->dicing_camera->set_full_height(height);
    scene->dicing_camera->compute_auto_viewplane();

    // setup background
/*    bool use_background = true;
	if (use_background)
	{
        ShaderGraph *bg_graph = new ShaderGraph();
		Shader *bg_shader = scene->default_background;		
		BackgroundNode *bg_node = bg_graph->create_node<BackgroundNode>();
		bg_node->input("Color")->set(make_float3(1.0, 1.0, 1.0));
		bg_node->input("Strength")->set(0.3);
		ShaderNode *bg_out = bg_graph->output();
		bg_graph->add(bg_node);
		bg_graph->connect(bg_node->output("Background"),
		bg_out->input("Surface")); bg_shader->set_graph(bg_graph);
		bg_shader->tag_update(scene);
	}*/
}

static ShaderEvalType get_shader_type(const string &pass_type)
{
    const char *shader_type = pass_type.c_str();

    if (strcmp(shader_type, "NORMAL") == 0)
        return SHADER_EVAL_NORMAL;
    else if (strcmp(shader_type, "UV") == 0)
        return SHADER_EVAL_UV;
    else if (strcmp(shader_type, "ROUGHNESS") == 0)
        return SHADER_EVAL_ROUGHNESS;
    else if (strcmp(shader_type, "DIFFUSE_COLOR") == 0)
        return SHADER_EVAL_DIFFUSE_COLOR;
    else if (strcmp(shader_type, "GLOSSY_COLOR") == 0)
        return SHADER_EVAL_GLOSSY_COLOR;
    else if (strcmp(shader_type, "TRANSMISSION_COLOR") == 0)
        return SHADER_EVAL_TRANSMISSION_COLOR;
    else if (strcmp(shader_type, "EMIT") == 0)
        return SHADER_EVAL_EMISSION;

    else if (strcmp(shader_type, "AO") == 0)
        return SHADER_EVAL_AO;
    else if (strcmp(shader_type, "COMBINED") == 0)
        return SHADER_EVAL_COMBINED;
    else if (strcmp(shader_type, "SHADOW") == 0)
        return SHADER_EVAL_SHADOW;
    else if (strcmp(shader_type, "DIFFUSE") == 0)
        return SHADER_EVAL_DIFFUSE;
    else if (strcmp(shader_type, "GLOSSY") == 0)
        return SHADER_EVAL_GLOSSY;
    else if (strcmp(shader_type, "TRANSMISSION") == 0)
        return SHADER_EVAL_TRANSMISSION;

    else if (strcmp(shader_type, "ENVIRONMENT") == 0)
        return SHADER_EVAL_ENVIRONMENT;

    else
        return SHADER_EVAL_BAKE;
}

static int bake_pass_filter_get(const int pass_filter)
{
    int flag = BAKE_FILTER_NONE;

    flag |= BAKE_FILTER_DIRECT;
    flag |= BAKE_FILTER_INDIRECT;
    flag |= BAKE_FILTER_COLOR;
    flag |= BAKE_FILTER_DIFFUSE;
    flag |= BAKE_FILTER_GLOSSY;
    flag |= BAKE_FILTER_TRANSMISSION;
    flag |= BAKE_FILTER_EMISSION;
    flag |= BAKE_FILTER_AO;

    return flag;
}

inline int max_ii(int a, int b)
{
    return (b < a) ? a : b;
}
inline float min_ff(float a, float b)
{
    return (a < b) ? a : b;
}
inline int min_ii(int a, int b)
{
    return (a < b) ? a : b;
}
inline float max_ff(float a, float b)
{
    return (a > b) ? a : b;
}

static void bake_differentials(BakeDataZSpan *bd,
                               const float *uv1,
                               const float *uv2,
                               const float *uv3)
{
    float A;

    /* assumes dPdu = P1 - P3 and dPdv = P2 - P3 */
    A = (uv2[0] - uv1[0]) * (uv3[1] - uv1[1]) - (uv3[0] - uv1[0]) * (uv2[1] - uv1[1]);

    if (fabsf(A) > FLT_EPSILON)
    {
        A = 0.5f / A;

        bd->du_dx = (uv2[1] - uv3[1]) * A;
        bd->dv_dx = (uv3[1] - uv1[1]) * A;

        bd->du_dy = (uv3[0] - uv2[0]) * A;
        bd->dv_dy = (uv1[0] - uv3[0]) * A;
    }
    else
    {
        bd->du_dx = bd->du_dy = 0.0f;
        bd->dv_dx = bd->dv_dy = 0.0f;
    }
}

static void zbuf_add_to_span(ZSpan *zspan, const float v1[2], const float v2[2])
{
    const float *minv, *maxv;
    float *span;
    float xx1, dx0, xs0;
    int y, my0, my2;

    if (v1[1] < v2[1])
    {
        minv = v1;
        maxv = v2;
    }
    else
    {
        minv = v2;
        maxv = v1;
    }

    my0 = ceil(minv[1]);
    my2 = floor(maxv[1]);

    if (my2 < 0 || my0 >= zspan->recty)
    {
        return;
    }

    /* clip top */
    if (my2 >= zspan->recty)
    {
        my2 = zspan->recty - 1;
    }
    /* clip bottom */
    if (my0 < 0)
    {
        my0 = 0;
    }

    if (my0 > my2)
    {
        return;
    }
    /* if (my0>my2) should still fill in, that way we get spans that skip nicely */

    xx1 = maxv[1] - minv[1];
    if (xx1 > FLT_EPSILON)
    {
        dx0 = (minv[0] - maxv[0]) / xx1;
        xs0 = dx0 * (minv[1] - my2) + minv[0];
    }
    else
    {
        dx0 = 0.0f;
        xs0 = min_ff(minv[0], maxv[0]);
    }

    /* empty span */
    if (zspan->maxp1 == NULL)
    {
        span = zspan->span1;
    }
    else
    { /* does it complete left span? */
        if (maxv == zspan->minp1 || minv == zspan->maxp1)
        {
            span = zspan->span1;
        }
        else
        {
            span = zspan->span2;
        }
    }

    if (span == zspan->span1)
    {
        //      printf("left span my0 %d my2 %d\n", my0, my2);
        if (zspan->minp1 == NULL || zspan->minp1[1] > minv[1])
        {
            zspan->minp1 = minv;
        }
        if (zspan->maxp1 == NULL || zspan->maxp1[1] < maxv[1])
        {
            zspan->maxp1 = maxv;
        }
        if (my0 < zspan->miny1)
        {
            zspan->miny1 = my0;
        }
        if (my2 > zspan->maxy1)
        {
            zspan->maxy1 = my2;
        }
    }
    else
    {
        //      printf("right span my0 %d my2 %d\n", my0, my2);
        if (zspan->minp2 == NULL || zspan->minp2[1] > minv[1])
        {
            zspan->minp2 = minv;
        }
        if (zspan->maxp2 == NULL || zspan->maxp2[1] < maxv[1])
        {
            zspan->maxp2 = maxv;
        }
        if (my0 < zspan->miny2)
        {
            zspan->miny2 = my0;
        }
        if (my2 > zspan->maxy2)
        {
            zspan->maxy2 = my2;
        }
    }

    for (y = my2; y >= my0; y--, xs0 += dx0)
    {
        /* xs0 is the xcoord! */
        span[y] = xs0;
    }
}

/* reset range for clipping */
static void zbuf_init_span(ZSpan *zspan)
{
    zspan->miny1 = zspan->miny2 = zspan->recty + 1;
    zspan->maxy1 = zspan->maxy2 = -1;
    zspan->minp1 = zspan->maxp1 = zspan->minp2 = zspan->maxp2 = NULL;
}

/* Scanconvert for strand triangles, calls func for each x, y coordinate
 * and gives UV barycentrics and z. */

void zspan_scanconvert(ZSpan *zspan,
                       BakeDataZSpan *handle,
                       float *v1,
                       float *v2,
                       float *v3,
                       void (*func)(BakeDataZSpan *, int, int, float, float))
{
    float x0, y0, x1, y1, x2, y2, z0, z1, z2;
    float u, v, uxd, uyd, vxd, vyd, uy0, vy0, xx1;
    const float *span1, *span2;
    int i, j, x, y, sn1, sn2, rectx = zspan->rectx, my0, my2;

    /* init */
    zbuf_init_span(zspan);

    /* set spans */
    zbuf_add_to_span(zspan, v1, v2);
    zbuf_add_to_span(zspan, v2, v3);
    zbuf_add_to_span(zspan, v3, v1);

    /* clipped */
    if (zspan->minp2 == NULL || zspan->maxp2 == NULL)
    {
        return;
    }

    my0 = max_ii(zspan->miny1, zspan->miny2);
    my2 = min_ii(zspan->maxy1, zspan->maxy2);

    //  printf("my %d %d\n", my0, my2);
    if (my2 < my0)
    {
        return;
    }

    /* ZBUF DX DY, in floats still */
    x1 = v1[0] - v2[0];
    x2 = v2[0] - v3[0];
    y1 = v1[1] - v2[1];
    y2 = v2[1] - v3[1];

    z1 = 1.0f; /* (u1 - u2) */
    z2 = 0.0f; /* (u2 - u3) */

    x0 = y1 * z2 - z1 * y2;
    y0 = z1 * x2 - x1 * z2;
    z0 = x1 * y2 - y1 * x2;

    if (z0 == 0.0f)
    {
        return;
    }

    xx1 = (x0 * v1[0] + y0 * v1[1]) / z0 + 1.0f;
    uxd = -(double)x0 / (double)z0;
    uyd = -(double)y0 / (double)z0;
    uy0 = ((double)my2) * uyd + (double)xx1;

    z1 = -1.0f; /* (v1 - v2) */
    z2 = 1.0f;  /* (v2 - v3) */

    x0 = y1 * z2 - z1 * y2;
    y0 = z1 * x2 - x1 * z2;

    xx1 = (x0 * v1[0] + y0 * v1[1]) / z0;
    vxd = -(double)x0 / (double)z0;
    vyd = -(double)y0 / (double)z0;
    vy0 = ((double)my2) * vyd + (double)xx1;

    /* correct span */
    span1 = zspan->span1 + my2;
    span2 = zspan->span2 + my2;

    for (i = 0, y = my2; y >= my0; i++, y--, span1--, span2--)
    {

        sn1 = floor(min_ff(*span1, *span2));
        sn2 = floor(max_ff(*span1, *span2));
        sn1++;

        if (sn2 >= rectx)
        {
            sn2 = rectx - 1;
        }
        if (sn1 < 0)
        {
            sn1 = 0;
        }

        u = (((double)sn1 * uxd) + uy0) - (i * uyd);
        v = (((double)sn1 * vxd) + vy0) - (i * vyd);

        for (j = 0, x = sn1; x <= sn2; j++, x++)
        {
            func(handle, x, y, u + (j * uxd), v + (j * vxd));
        }
    }
}

inline void copy_v2_fl2(float v[2], float x, float y)
{
    v[0] = x;
    v[1] = y;
}

static void store_bake_pixel(BakeDataZSpan *handle, int x, int y, float u, float v)
{
    BakeDataZSpan *bd = (BakeDataZSpan *)handle;
    BakePixel *pixel;

    const int width = bd->bk_image->width;
    const size_t offset = bd->bk_image->offset;
    const int i = offset + y * width + x;

    pixel = &bd->pixel_array[i];
    pixel->seed = rand();
    pixel->primitive_id = bd->primitive_id;    

    /* At this point object_id is always 0, since this function runs for the
     * low-poly mesh only. The object_id lookup indices are set afterwards. */

    copy_v2_fl2(pixel->uv, u, v);

    pixel->du_dx = bd->du_dx;
    pixel->du_dy = bd->du_dy;
    pixel->dv_dx = bd->dv_dx;
    pixel->dv_dy = bd->dv_dy;
    pixel->object_id = 0;
}

/* each zbuffer has coordinates transformed to local rect coordinates, so we can simply clip */
void zbuf_alloc_span(ZSpan *zspan, int rectx, int recty)
{
    memset(zspan, 0, sizeof(ZSpan));

    zspan->rectx = rectx;
    zspan->recty = recty;

    zspan->span1 = (float *)malloc(recty * sizeof(float));
    zspan->span2 = (float *)malloc(recty * sizeof(float));
}

void populate_bake_data(const Mesh &mesh, size_t uv_map_index, MyBakeData *data)
{
    int image_width = data->width;
    int image_height = data->height;
	
    size_t num_pixels = image_width * image_height;    

	/* initialize all pixel arrays so we know which ones are 'blank' */
    BakeDataZSpan bd;
    bd.bk_image = new BakeImage();
    bd.bk_image->width = image_width;
    bd.bk_image->height = image_height;
    bd.bk_image->offset = 0;
    bd.pixel_array = (BakePixel *)malloc(sizeof(BakePixel) * num_pixels);
    bd.zspan = new ZSpan();

    for (size_t i = 0; i < num_pixels; i++)
    {
        bd.pixel_array[i].primitive_id = -1;
        bd.pixel_array[i].object_id = 0;
    }
    zbuf_alloc_span(bd.zspan, image_width, image_height);

    /*ZSpan zspan;
    zspan.rectx = image_width;
    zspan.recty = image_height;*/

    Attribute *attributes = mesh.attributes.find(ATTR_STD_UV);
    float2 *fdata = attributes[uv_map_index].data_float2();
    size_t triangles_count = mesh.num_triangles();
	
    for (size_t i = 0; i < triangles_count; i++)
    {
        bd.primitive_id = i;
        float vec[3][2];

        Mesh::Triangle triangle = mesh.get_triangle(i);
        /* array<float3> p1 = mesh.verts[triangle.v[0]];
         array<float3> p2 = mesh.verts[triangle.v[1]];
         array<float3> p3 = mesh.verts[triangle.v[2]];*/
        for (size_t j = 0; j < 3; j++)
        {
            float2 uv = fdata[i*3 + j];
            vec[j][0] = uv[0] * (float)bd.bk_image->width - (0.5f + 0.001f);
            vec[j][1] = uv[1] * (float)bd.bk_image->height - (0.5f + 0.002f);
        }

        bake_differentials(&bd, vec[0], vec[1], vec[2]);
        zspan_scanconvert(bd.zspan, &bd, vec[0], vec[1], vec[2], store_bake_pixel);
    }	
	
    BakePixel *bp = bd.pixel_array;
    for (size_t y = 0; y < image_height; y++)
    {
	    for (size_t x = 0; x < image_width; x++)
	    {
	    	data->set(x, y, bp->seed, bp->primitive_id, make_float2(bp->uv[0], bp->uv[1]), bp->du_dx, bp->du_dy, bp->dv_dx, bp->dv_dy);
	    	bp++;
	    }
    }
	
    free(bd.pixel_array);
    delete bd.bk_image;
    delete bd.zspan;    
}

int main()
{
    size_t width = 1024;
    size_t height = 1024;
    size_t samples = 128;

    path_init();

    // init session section
    SessionParams session_params;
    session_params.samples = samples;
    session_params.background = true;
    session_params.progressive = false;
    session_params.progressive_refine = false;
    session_params.tile_size.x = 32;
    session_params.tile_size.y = 32;
#if DEBUG
    //session_params.threads = 1;
#endif

	vector<DeviceInfo> &devices = Device::available_devices();
    for (DeviceInfo &device : devices)
    {
        if (device.type == DeviceType::DEVICE_CPU)
            session_params.device = device;
    }
	
    Session *session = new Session(session_params);
    SceneParams scene_params;
    scene_params.shadingsystem = SHADINGSYSTEM_SVM;

    Scene *scene = new Scene(scene_params, session->device);   
    setup_scene(scene, width, height);
    session->scene = scene;
  
	
	for (auto object : scene->objects)
	{
        Geometry* geom = object->get_geometry();
        if (!geom || geom->geometry_type != Geometry::Type::MESH)
            continue;

		Mesh *mesh = (Mesh *)geom;

		// TODO: bake only a plane so far
		if (object->name != "plane")
            continue;

		std::cout << "Baking '" << object->name << "'..." << std::endl;

		bake_data.init(width, height);
        populate_bake_data(*mesh, 0, &bake_data);
		
		Pass::add(PASS_COMBINED, scene->passes, "Combined");
        session->read_bake_tile_cb = function_bind(&read_tile_from_buffer, _1);
        session->write_render_tile_cb = function_bind(&write_tile_to_buffer, _1);

        ShaderEvalType shader_type = get_shader_type("COMBINED");
        int bake_pass_filter = bake_pass_filter_get(0);
        scene->bake_manager->set(scene, object->name.c_str(), shader_type, bake_pass_filter);

        BufferParams buffer_params;
        buffer_params.width = width;
        buffer_params.height = height;
        buffer_params.full_width = width;
        buffer_params.full_height = height;
        buffer_params.passes = scene->passes;

        session->tile_manager.set_samples(session_params.samples);
        session->reset(buffer_params, session_params.samples);	

		session->start();
        session->wait();

		save_image(string("T_") + object->name.c_str() + ".png", bake_data.buffer_combined);
	}

    std::cout << "Finish baking" << std::endl;    
}