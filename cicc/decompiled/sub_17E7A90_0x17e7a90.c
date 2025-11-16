// Function: sub_17E7A90
// Address: 0x17e7a90
//
__m128i *sub_17E7A90()
{
  __m128i *v0; // rax
  __m128i *v1; // r12
  __m128i *v2; // rax
  __int64 v3; // rax
  bool v4; // zf
  __int64 v5; // rax
  __m128i *v7; // [rsp+0h] [rbp-30h] BYREF
  __int64 v8; // [rsp+8h] [rbp-28h]
  __m128i v9[2]; // [rsp+10h] [rbp-20h] BYREF

  v7 = v9;
  sub_17E2210((__int64 *)&v7, byte_3F871B3, (__int64)byte_3F871B3);
  v0 = (__m128i *)sub_22077B0(192);
  v1 = v0;
  if ( v0 )
  {
    v0->m128i_i64[1] = 0;
    v0[1].m128i_i64[0] = (__int64)&unk_4FA4E0C;
    v0[5].m128i_i64[0] = (__int64)v0[4].m128i_i64;
    v0[5].m128i_i64[1] = (__int64)v0[4].m128i_i64;
    v0[8].m128i_i64[0] = (__int64)v0[7].m128i_i64;
    v0[8].m128i_i64[1] = (__int64)v0[7].m128i_i64;
    v0->m128i_i64[0] = (__int64)off_49F0580;
    v0[10].m128i_i64[0] = (__int64)v0[11].m128i_i64;
    v2 = v7;
    v1[1].m128i_i32[2] = 5;
    v1[2].m128i_i64[0] = 0;
    v1[2].m128i_i64[1] = 0;
    v1[3].m128i_i64[0] = 0;
    v1[4].m128i_i32[0] = 0;
    v1[4].m128i_i64[1] = 0;
    v1[6].m128i_i64[0] = 0;
    v1[7].m128i_i32[0] = 0;
    v1[7].m128i_i64[1] = 0;
    v1[9].m128i_i64[0] = 0;
    v1[9].m128i_i8[8] = 0;
    if ( v2 == v9 )
    {
      v1[11] = _mm_load_si128(v9);
    }
    else
    {
      v1[10].m128i_i64[0] = (__int64)v2;
      v1[11].m128i_i64[0] = v9[0].m128i_i64[0];
    }
    v3 = v8;
    v4 = qword_4FA59E8 == 0;
    v7 = v9;
    v8 = 0;
    v1[10].m128i_i64[1] = v3;
    v9[0].m128i_i8[0] = 0;
    if ( !v4 )
      sub_2240AE0(&v1[10], &qword_4FA59E0);
    v5 = sub_163A1D0();
    sub_17E79A0(v5);
  }
  if ( v7 != v9 )
    j_j___libc_free_0(v7, v9[0].m128i_i64[0] + 1);
  return v1;
}
