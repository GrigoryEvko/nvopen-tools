// Function: sub_371DF50
// Address: 0x371df50
//
__m128i *sub_371DF50()
{
  const __m128i *v0; // rax
  __m128i *v1; // r12
  __m128i v2; // xmm1
  __int8 v3; // al
  __m128i v4; // xmm0
  __int128 *v5; // rax
  __m128i v7; // [rsp+0h] [rbp-30h] BYREF
  void (__fastcall *v8)(__m128i *, __m128i *, __int64); // [rsp+10h] [rbp-20h]
  __int64 v9; // [rsp+18h] [rbp-18h]

  v0 = (const __m128i *)sub_22077B0(0xD0u);
  v1 = (__m128i *)v0;
  if ( v0 )
  {
    v0->m128i_i64[1] = 0;
    v0[1].m128i_i32[2] = 2;
    v2 = _mm_loadu_si128(v0 + 11);
    v0[1].m128i_i64[0] = (__int64)&unk_505099C;
    v0[3].m128i_i64[1] = (__int64)&v0[6].m128i_i64[1];
    v3 = qword_5050BA8[8];
    v1[7].m128i_i64[0] = (__int64)v1[10].m128i_i64;
    v1[5].m128i_i32[2] = 1065353216;
    v1[9].m128i_i32[0] = 1065353216;
    v4 = _mm_loadu_si128(&v7);
    v1->m128i_i64[0] = (__int64)&unk_4A3D0A0;
    v7 = v2;
    v1[11] = v4;
    v1[10].m128i_i8[9] = v3;
    v1[2].m128i_i64[0] = 0;
    v1[2].m128i_i64[1] = 0;
    v1[3].m128i_i64[0] = 0;
    v1[4].m128i_i64[0] = 1;
    v1[4].m128i_i64[1] = 0;
    v1[5].m128i_i64[0] = 0;
    v1[6].m128i_i64[0] = 0;
    v1[6].m128i_i64[1] = 0;
    v1[7].m128i_i64[1] = 1;
    v1[8].m128i_i64[0] = 0;
    v1[8].m128i_i64[1] = 0;
    v1[9].m128i_i64[1] = 0;
    v1[10].m128i_i64[0] = 0;
    v1[10].m128i_i8[8] = 0;
    v8 = 0;
    v1[12].m128i_i64[0] = (__int64)sub_371CD10;
    v9 = v1[12].m128i_i64[1];
    v1[12].m128i_i64[1] = (__int64)nullsub_1920;
    v5 = sub_BC2B00();
    sub_371DED0((__int64)v5);
    if ( v8 )
      v8(&v7, &v7, 3);
  }
  return v1;
}
