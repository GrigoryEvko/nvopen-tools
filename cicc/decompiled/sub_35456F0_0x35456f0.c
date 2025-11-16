// Function: sub_35456F0
// Address: 0x35456f0
//
__m128i *__fastcall sub_35456F0(_QWORD *a1, __int64 a2, const __m128i *a3)
{
  __int64 v4; // rbx
  __int64 v5; // r9
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // r8
  __m128i *result; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // r8
  const void *v13; // rsi
  __int8 *v14; // r12
  __int64 v15; // rdi
  const void *v16; // rsi
  __int8 *v17; // r12

  v4 = sub_3545690(a1, a2);
  if ( a2 == (a3->m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v10 = *(unsigned int *)(v4 + 152);
    v11 = *(_QWORD *)(v4 + 144);
    v12 = v10 + 1;
    if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(v4 + 156) )
    {
      v15 = v4 + 144;
      v16 = (const void *)(v4 + 160);
      if ( v11 > (unsigned __int64)a3 || (unsigned __int64)a3 >= v11 + 32 * v10 )
      {
        sub_C8D5F0(v15, v16, v12, 0x20u, v12, v5);
        v11 = *(_QWORD *)(v4 + 144);
        v10 = *(unsigned int *)(v4 + 152);
      }
      else
      {
        v17 = &a3->m128i_i8[-v11];
        sub_C8D5F0(v15, v16, v12, 0x20u, v12, v5);
        v11 = *(_QWORD *)(v4 + 144);
        v10 = *(unsigned int *)(v4 + 152);
        a3 = (const __m128i *)&v17[v11];
      }
    }
    result = (__m128i *)(v11 + 32 * v10);
    *result = _mm_loadu_si128(a3);
    result[1] = _mm_loadu_si128(a3 + 1);
    ++*(_DWORD *)(v4 + 152);
  }
  else
  {
    v6 = *(unsigned int *)(v4 + 8);
    v7 = *(_QWORD *)v4;
    v8 = v6 + 1;
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(v4 + 12) )
    {
      v13 = (const void *)(v4 + 16);
      if ( v7 > (unsigned __int64)a3 || (unsigned __int64)a3 >= v7 + 32 * v6 )
      {
        sub_C8D5F0(v4, v13, v8, 0x20u, v8, v5);
        v7 = *(_QWORD *)v4;
        v6 = *(unsigned int *)(v4 + 8);
      }
      else
      {
        v14 = &a3->m128i_i8[-v7];
        sub_C8D5F0(v4, v13, v8, 0x20u, v8, v5);
        v7 = *(_QWORD *)v4;
        v6 = *(unsigned int *)(v4 + 8);
        a3 = (const __m128i *)&v14[*(_QWORD *)v4];
      }
    }
    result = (__m128i *)(v7 + 32 * v6);
    *result = _mm_loadu_si128(a3);
    result[1] = _mm_loadu_si128(a3 + 1);
    ++*(_DWORD *)(v4 + 8);
  }
  return result;
}
