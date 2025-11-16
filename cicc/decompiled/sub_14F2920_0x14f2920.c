// Function: sub_14F2920
// Address: 0x14f2920
//
__m128i **__fastcall sub_14F2920(__m128i **a1, _QWORD *a2, __int64 *a3)
{
  __m128i *v6; // r14
  _BYTE *v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rdi
  __m128i **result; // rax
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rsi
  __int64 v14[2]; // [rsp+0h] [rbp-40h] BYREF
  __m128i v15[3]; // [rsp+10h] [rbp-30h] BYREF

  v6 = a1[1];
  if ( v6 == a1[2] )
    return sub_14F2560(a1, a1[1], (__int64)a2, a3);
  v7 = (_BYTE *)*a2;
  v8 = a2[1];
  v14[0] = (__int64)v15;
  sub_14E9FB0(v14, v7, (__int64)&v7[v8]);
  v9 = *a3;
  result = (__m128i **)a3[1];
  *a3 = 0;
  a3[1] = 0;
  v11 = a3[2];
  a3[2] = 0;
  if ( v6 )
  {
    v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
    if ( (__m128i *)v14[0] == v15 )
    {
      v6[1] = _mm_load_si128(v15);
    }
    else
    {
      v6->m128i_i64[0] = v14[0];
      v6[1].m128i_i64[0] = v15[0].m128i_i64[0];
    }
    v12 = v14[1];
    v6[2].m128i_i64[0] = v9;
    v6[2].m128i_i64[1] = (__int64)result;
    v6->m128i_i64[1] = v12;
    v6[3].m128i_i64[0] = v11;
  }
  else
  {
    v13 = v11 - v9;
    if ( v9 )
      result = (__m128i **)j_j___libc_free_0(v9, v13);
    if ( (__m128i *)v14[0] != v15 )
      result = (__m128i **)j_j___libc_free_0(v14[0], v15[0].m128i_i64[0] + 1);
  }
  a1[1] = (__m128i *)((char *)a1[1] + 56);
  return result;
}
