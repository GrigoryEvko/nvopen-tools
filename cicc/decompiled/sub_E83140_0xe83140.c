// Function: sub_E83140
// Address: 0xe83140
//
__m128i *__fastcall sub_E83140(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i *result; // rax
  const __m128i *v7; // r12
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // r8
  __int64 v13; // rdi
  const void *v14; // rsi
  char *v15; // r12
  _QWORD v16[7]; // [rsp-38h] [rbp-38h] BYREF

  result = *(__m128i **)(a2 + 16);
  if ( (result->m128i_i8[8] & 2) == 0 )
  {
    result = *(__m128i **)(a3 + 16);
    if ( (result->m128i_i8[8] & 2) == 0 )
    {
      v7 = (const __m128i *)v16;
      v8 = *(_QWORD *)(a1 + 296);
      v16[2] = a4;
      v16[1] = a3;
      v9 = *(_QWORD *)(v8 + 24);
      v16[0] = a2;
      v10 = *(unsigned int *)(v9 + 96);
      v11 = *(_QWORD *)(v9 + 88);
      v12 = v10 + 1;
      if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(v9 + 100) )
      {
        v13 = v9 + 88;
        v14 = (const void *)(v9 + 104);
        if ( v11 > (unsigned __int64)v16 || (unsigned __int64)v16 >= v11 + 24 * v10 )
        {
          sub_C8D5F0(v13, v14, v12, 0x18u, v12, a6);
          v11 = *(_QWORD *)(v9 + 88);
          v10 = *(unsigned int *)(v9 + 96);
        }
        else
        {
          v15 = (char *)v16 - v11;
          sub_C8D5F0(v13, v14, v12, 0x18u, v12, a6);
          v11 = *(_QWORD *)(v9 + 88);
          v10 = *(unsigned int *)(v9 + 96);
          v7 = (const __m128i *)&v15[v11];
        }
      }
      result = (__m128i *)(v11 + 24 * v10);
      *result = _mm_loadu_si128(v7);
      result[1].m128i_i64[0] = v7[1].m128i_i64[0];
      ++*(_DWORD *)(v9 + 96);
    }
  }
  return result;
}
