use bevy::ecs::world::World;
use bevy::render::mesh::{Indices, Mesh, VertexAttributeValues};
use bevy::render::pipeline::PrimitiveTopology;
use nalgebra::{point, vector, Point2, Point3, Vector2};
use bevy::ecs::prelude::{Res, With};
use bevy::asset::{Handle, Asset, Assets};
use bevy::ecs::system::{ResMut, Query};
use array_macro::array;
use std::slice::Iter;
use std::borrow::Cow;
use bevy::ecs::entity::Entity;

#[derive(Eq, PartialEq, Ord, PartialOrd, Copy, Clone, Hash, Debug)]
pub struct MaterialID {
    id: u32,
}

impl From<u32> for MaterialID {
    fn from(id: u32) -> Self {
        Self::from(id)
    }
}

impl MaterialID {
    const fn from(id: u32) -> Self {
        Self {
            id
        }
    }
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct Material {
    pub id: MaterialID,
    pub transparent : bool,
    pub custom_model : bool,
}

pub const AIR : Material = Material {
    id : MaterialID::from(0),
    transparent : true,
    custom_model : false,
};

#[derive(Eq, PartialEq, Debug, Copy, Clone, Hash)]
pub enum AADirection {
    XPositive = 0,
    XNegative,
    YPositive,
    YNegative,
    ZPositive,
    ZNegative,
}

impl AADirection {

    pub fn is_positive(&self) -> bool {
        match self {
            AADirection::XPositive => true,
            AADirection::XNegative => false,
            AADirection::YPositive => true,
            AADirection::YNegative => false,
            AADirection::ZPositive => true,
            AADirection::ZNegative => false,
        }
    }

    pub fn is_negative(&self) -> bool {
        !self.is_positive()
    }
}

#[derive(Clone)]
pub struct Block {
    pub material : MaterialID,
}

impl Default for Block {
    fn default() -> Self {
        Block {
            material : AIR.id,
        }
    }
}

pub struct Chunk {

    blocks : [Block; Chunk::SIZE_X * Chunk::SIZE_Y * Chunk::SIZE_Z],
    updated : bool,
    neighbors : [Option<Entity>; 6],
}

impl Chunk {

    pub const SIZE_X : usize = 16;
    pub const SIZE_Y : usize = 16;
    pub const SIZE_Z : usize = 16;

    pub fn empty() -> Self {
        Chunk {
            blocks : array![Block::default(); Chunk::SIZE_X * Chunk::SIZE_Y * Chunk::SIZE_Z],
            updated : true,
            neighbors : [None; 6],
        }
    }

    pub fn set_neighbor(&mut self, entity : Entity, direction : AADirection) {
        self.neighbors[direction as usize] = Some(entity)
    }

    pub fn get_neighbor(&self, direction : AADirection) -> Option<Entity> {
        self.neighbors[direction as usize]
    }

    pub fn neighbors(&self) -> &[Option<Entity>] {
        &self.neighbors
    }

    pub fn block_at(&self, x : usize, y : usize, z : usize) -> &Block {
        self.blocks
            .get(Chunk::SIZE_X * (Chunk::SIZE_Y * z + y) + x)
            .expect("Block index out of range")
    }

    pub fn block_at_mut(&mut self, x : usize, y : usize, z : usize) -> &mut Block {
        self.blocks
            .get_mut(Chunk::SIZE_X * (Chunk::SIZE_Y * z + y) + x)
            .expect("Block index out of range")
    }

    pub fn has_changed(&self) -> bool {
        self.updated
    }

    pub fn set_change(&mut self) {
        self.updated = true
    }

    pub fn clear_change(&mut self) {
        self.updated = false
    }
}

pub fn chunk_end_of_tick_system(mut q: Query<(&mut Chunk)>) {
    for mut c in q.iter_mut() {
        c.updated = false;
    }
}

struct ChunkMeshData {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u16>,
}

impl Default for ChunkMeshData {
    fn default() -> Self {
        ChunkMeshData {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: Vec::new()
        }
    }
}

impl Into<Mesh> for ChunkMeshData {

    fn into(self) -> Mesh {
        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
        mesh.set_indices(Some(Indices::U16(self.indices)));
        mesh.set_attribute(Mesh::ATTRIBUTE_POSITION, self.positions);
        mesh.set_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals);
        mesh.set_attribute(Mesh::ATTRIBUTE_UV_0, self.uvs);

        mesh
    }
}

pub trait ChunkMesher {
    fn generate_mesh(
        chunk: &Chunk,
    ) -> Mesh;
}

pub fn chunk_meshing_system<Mesher: ChunkMesher>(
    mut meshes: ResMut<Assets<Mesh>>,
    query: Query<(&Chunk, &Handle<Mesh>)>
) {
    for (chunk, mesh_handle) in query.iter() {

        if !chunk.has_changed() {
            continue;
        }

        let new_mesh = Mesher::generate_mesh(chunk);

        let mut mesh = meshes.get_mut(mesh_handle).unwrap();
        *mesh = new_mesh;
    }
}

trait FaceInserter {
    fn insert_face(
        block_pos: Point3<usize>,
        dimensions: Vector2<u8>,
        mesh: &mut ChunkMeshData
    );
}

struct FaceInserterImpl<const DIR: usize>;

impl FaceInserter for FaceInserterImpl<0> {
    fn insert_face(block_pos: Point3<usize>, dimensions: Vector2<u8>, mesh: &mut ChunkMeshData) {

        let base_vertex_idx = mesh.positions.len() as u16;
        let base_pos = point![block_pos.x as f32, block_pos.y as f32, block_pos.z as f32];
        let u = dimensions.x as f32;
        let v = dimensions.y as f32;

        // Top left
        mesh.positions.push([base_pos.x, base_pos.y + u, base_pos.z]);
        mesh.normals.push([1f32, 0f32, 0f32]);
        mesh.uvs.push([1f32, 1f32]);

        // Top right
        mesh.positions.push([base_pos.x, base_pos.y + u, base_pos.z + v]);
        mesh.normals.push([1f32, 0f32, 0f32]);
        mesh.uvs.push([1f32, 1f32]);

        // Bottom right
        mesh.positions.push([base_pos.x, base_pos.y, base_pos.z + v]);
        mesh.normals.push([1f32, 0f32, 0f32]);
        mesh.uvs.push([1f32, 1f32]);

        // Bottom left
        mesh.positions.push([base_pos.x, base_pos.y, base_pos.z]);
        mesh.normals.push([1f32, 0f32, 0f32]);
        mesh.uvs.push([1f32, 1f32]);

        // indices
        mesh.indices.push(base_vertex_idx);
        mesh.indices.push(base_vertex_idx + 1);
        mesh.indices.push(base_vertex_idx + 2);
        mesh.indices.push(base_vertex_idx + 2);
        mesh.indices.push(base_vertex_idx + 3);
        mesh.indices.push(base_vertex_idx);
    }
}

pub struct NaiveChunkMesher {}

impl ChunkMesher for NaiveChunkMesher {

    fn generate_mesh(chunk: &Chunk) -> Mesh {

        const DIMENSIONS : Vector2<u8> = vector![1, 1];

        let mut mesh = ChunkMeshData::default();

        for z in 0..Chunk::SIZE_Z {
            for y in 0..Chunk::SIZE_Y {
                for x in 0..Chunk::SIZE_X {

                    let block = chunk.block_at(x, y, z);
                    let block_pos = point![x, y, z];

                    FaceInserterImpl::<0>::insert_face(
                        point![block_pos.x + 1, block_pos.y, block_pos.z],
                        DIMENSIONS,
                        &mut mesh
                    );
                }
            }
        }

        mesh.into()
    }
}
