// Function: sub_B180C0
// Address: 0xb180c0
//
__m128i *__fastcall sub_B180C0(__int64 a1, unsigned __int64 a2)
{
  __m128i *v3; // rbx
  __int64 v4; // rcx
  __m128i *v5; // r13
  __int64 v6; // r9
  int v7; // edx
  __m128i *result; // rax
  __int64 v9; // rcx
  __m128i *v10; // rcx
  __int64 v11; // rcx
  __m128i v12; // xmm0
  __int64 v13; // r15
  __int64 v14; // r14
  unsigned __int64 v15; // rbx
  __m128i *v16; // rsi
  __int64 v17; // rdi
  int v18; // r15d
  __m128i *v19; // rsi
  __int64 v20; // rdi
  int v21; // r15d
  _DWORD v22[14]; // [rsp+8h] [rbp-38h] BYREF

  v3 = (__m128i *)a2;
  v4 = *(unsigned int *)(a1 + 88);
  v5 = *(__m128i **)(a1 + 80);
  v6 = v4 + 1;
  v7 = *(_DWORD *)(a1 + 88);
  if ( v4 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 92) )
  {
    v13 = a1 + 80;
    v14 = a1 + 96;
    if ( (unsigned __int64)v5 > a2 || a2 >= (unsigned __int64)&v5[5 * v4] )
    {
      v19 = (__m128i *)sub_C8D7D0(a1 + 80, v14, v6, 80, v22);
      v5 = v19;
      sub_B17F60(v13, v19);
      v20 = *(_QWORD *)(a1 + 80);
      v21 = v22[0];
      if ( v14 != v20 )
        _libc_free(v20, v19);
      v4 = *(unsigned int *)(a1 + 88);
      *(_QWORD *)(a1 + 80) = v19;
      *(_DWORD *)(a1 + 92) = v21;
      v7 = v4;
    }
    else
    {
      v15 = a2 - (_QWORD)v5;
      v16 = (__m128i *)sub_C8D7D0(a1 + 80, v14, v6, 80, v22);
      v5 = v16;
      sub_B17F60(v13, v16);
      v17 = *(_QWORD *)(a1 + 80);
      v18 = v22[0];
      if ( v14 != v17 )
        _libc_free(v17, v16);
      v4 = *(unsigned int *)(a1 + 88);
      *(_QWORD *)(a1 + 80) = v16;
      v3 = (__m128i *)((char *)v16 + v15);
      *(_DWORD *)(a1 + 92) = v18;
      v7 = v4;
    }
  }
  result = &v5[5 * v4];
  if ( result )
  {
    result->m128i_i64[0] = (__int64)result[1].m128i_i64;
    if ( (__m128i *)v3->m128i_i64[0] == &v3[1] )
    {
      result[1] = _mm_loadu_si128(v3 + 1);
    }
    else
    {
      result->m128i_i64[0] = v3->m128i_i64[0];
      result[1].m128i_i64[0] = v3[1].m128i_i64[0];
    }
    v9 = v3->m128i_i64[1];
    v3->m128i_i64[0] = (__int64)v3[1].m128i_i64;
    v3->m128i_i64[1] = 0;
    result->m128i_i64[1] = v9;
    v3[1].m128i_i8[0] = 0;
    result[2].m128i_i64[0] = (__int64)result[3].m128i_i64;
    v10 = (__m128i *)v3[2].m128i_i64[0];
    if ( v10 == &v3[3] )
    {
      result[3] = _mm_loadu_si128(v3 + 3);
    }
    else
    {
      result[2].m128i_i64[0] = (__int64)v10;
      result[3].m128i_i64[0] = v3[3].m128i_i64[0];
    }
    v11 = v3[2].m128i_i64[1];
    v3[2].m128i_i64[0] = (__int64)v3[3].m128i_i64;
    v3[2].m128i_i64[1] = 0;
    result[2].m128i_i64[1] = v11;
    v12 = _mm_loadu_si128(v3 + 4);
    v3[3].m128i_i8[0] = 0;
    result[4] = v12;
    v7 = *(_DWORD *)(a1 + 88);
  }
  *(_DWORD *)(a1 + 88) = v7 + 1;
  return result;
}
