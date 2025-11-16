// Function: sub_17E7C40
// Address: 0x17e7c40
//
__m128i *__fastcall sub_17E7C40(_BYTE *a1, __int64 a2)
{
  __m128i *v2; // rax
  __m128i *v3; // r12
  __m128i *v4; // rax
  __int64 v5; // rax
  bool v6; // zf
  __int64 v7; // rax
  __m128i *v9; // [rsp+0h] [rbp-30h] BYREF
  __int64 v10; // [rsp+8h] [rbp-28h]
  __m128i v11[2]; // [rsp+10h] [rbp-20h] BYREF

  if ( a1 )
  {
    v9 = v11;
    sub_17E2210((__int64 *)&v9, a1, (__int64)&a1[a2]);
  }
  else
  {
    v10 = 0;
    v9 = v11;
    v11[0].m128i_i8[0] = 0;
  }
  v2 = (__m128i *)sub_22077B0(192);
  v3 = v2;
  if ( v2 )
  {
    v2->m128i_i64[1] = 0;
    v2[1].m128i_i64[0] = (__int64)&unk_4FA4E0C;
    v2[5].m128i_i64[0] = (__int64)v2[4].m128i_i64;
    v2[5].m128i_i64[1] = (__int64)v2[4].m128i_i64;
    v2[8].m128i_i64[0] = (__int64)v2[7].m128i_i64;
    v2[8].m128i_i64[1] = (__int64)v2[7].m128i_i64;
    v2->m128i_i64[0] = (__int64)off_49F0580;
    v2[10].m128i_i64[0] = (__int64)v2[11].m128i_i64;
    v4 = v9;
    v3[1].m128i_i32[2] = 5;
    v3[2].m128i_i64[0] = 0;
    v3[2].m128i_i64[1] = 0;
    v3[3].m128i_i64[0] = 0;
    v3[4].m128i_i32[0] = 0;
    v3[4].m128i_i64[1] = 0;
    v3[6].m128i_i64[0] = 0;
    v3[7].m128i_i32[0] = 0;
    v3[7].m128i_i64[1] = 0;
    v3[9].m128i_i64[0] = 0;
    v3[9].m128i_i8[8] = 0;
    if ( v4 == v11 )
    {
      v3[11] = _mm_load_si128(v11);
    }
    else
    {
      v3[10].m128i_i64[0] = (__int64)v4;
      v3[11].m128i_i64[0] = v11[0].m128i_i64[0];
    }
    v5 = v10;
    v6 = qword_4FA59E8 == 0;
    v9 = v11;
    v10 = 0;
    v3[10].m128i_i64[1] = v5;
    v11[0].m128i_i8[0] = 0;
    if ( !v6 )
      sub_2240AE0(&v3[10], &qword_4FA59E0);
    v7 = sub_163A1D0();
    sub_17E79A0(v7);
  }
  if ( v9 != v11 )
    j_j___libc_free_0(v9, v11[0].m128i_i64[0] + 1);
  return v3;
}
