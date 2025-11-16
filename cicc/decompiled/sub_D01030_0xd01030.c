// Function: sub_D01030
// Address: 0xd01030
//
__int64 __fastcall sub_D01030(__m128i *a1, const __m128i *a2)
{
  __int64 v3; // rax
  __m128i v4; // xmm0
  __int64 v5; // rax
  __int64 v6; // rdx
  unsigned int v7; // r12d
  bool v8; // cc
  __int64 v9; // rdi
  __int64 result; // rax
  __int64 v11; // rdi
  __int32 v12; // eax
  __int64 v13; // rdi
  __int32 v14; // eax
  __int64 v15; // [rsp+0h] [rbp-30h] BYREF
  __int64 v16; // [rsp+8h] [rbp-28h]

  v3 = a2[1].m128i_i64[0];
  v4 = _mm_loadu_si128(a2);
  a1[2].m128i_i32[0] = 1;
  a1[1].m128i_i64[1] = 0;
  a1[1].m128i_i64[0] = v3;
  a1[3].m128i_i32[0] = 1;
  a1[2].m128i_i64[1] = 0;
  a1[3].m128i_i16[4] = 257;
  v5 = a2->m128i_i64[0];
  *a1 = v4;
  v15 = sub_BCAE30(*(_QWORD *)(v5 + 8));
  v16 = v6;
  v7 = sub_CA1930(&v15) + a2->m128i_i32[2] + a2->m128i_i32[3] - a2[1].m128i_i32[0];
  LODWORD(v16) = v7;
  if ( v7 > 0x40 )
  {
    sub_C43690((__int64)&v15, 1, 0);
    if ( a1[2].m128i_i32[0] > 0x40u )
    {
      v13 = a1[1].m128i_i64[1];
      if ( v13 )
        j_j___libc_free_0_0(v13);
    }
    a1[1].m128i_i64[1] = v15;
    v14 = v16;
    LODWORD(v16) = v7;
    a1[2].m128i_i32[0] = v14;
    sub_C43690((__int64)&v15, 0, 0);
  }
  else
  {
    v8 = a1[2].m128i_i32[0] <= 0x40u;
    v15 = 1;
    if ( v8 || (v11 = a1[1].m128i_i64[1]) == 0 )
    {
      a1[1].m128i_i64[1] = 1;
      a1[2].m128i_i32[0] = v7;
    }
    else
    {
      j_j___libc_free_0_0(v11);
      a1[1].m128i_i64[1] = v15;
      v12 = v16;
      LODWORD(v16) = v7;
      a1[2].m128i_i32[0] = v12;
    }
    v15 = 0;
  }
  if ( a1[3].m128i_i32[0] > 0x40u )
  {
    v9 = a1[2].m128i_i64[1];
    if ( v9 )
      j_j___libc_free_0_0(v9);
  }
  a1[2].m128i_i64[1] = v15;
  result = (unsigned int)v16;
  a1[3].m128i_i32[0] = v16;
  return result;
}
