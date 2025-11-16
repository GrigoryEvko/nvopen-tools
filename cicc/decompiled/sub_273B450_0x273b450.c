// Function: sub_273B450
// Address: 0x273b450
//
__m128i *__fastcall sub_273B450(__int64 a1, const __m128i *a2)
{
  __int64 v2; // rbp
  unsigned __int64 v3; // rcx
  unsigned __int64 v4; // r8
  unsigned __int64 v5; // r9
  __m128i *result; // rax
  int v7; // edx
  unsigned __int64 v8; // rdx
  const __m128i *v9; // rbx
  __m128i v10; // xmm5
  __m128i v11; // xmm6
  __m128i v12; // xmm7
  __m128i *v13; // rcx
  const void *v14; // rsi
  char *v15; // rbx
  _OWORD v16[5]; // [rsp-58h] [rbp-58h] BYREF
  __int64 v17; // [rsp-8h] [rbp-8h]

  v3 = *(unsigned int *)(a1 + 8);
  v4 = *(_QWORD *)a1;
  v5 = *(unsigned int *)(a1 + 12);
  result = (__m128i *)(*(_QWORD *)a1 + (v3 << 6));
  if ( v3 >= v5 )
  {
    v17 = v2;
    v8 = v3 + 1;
    v9 = (const __m128i *)v16;
    v10 = _mm_loadu_si128(a2 + 1);
    v11 = _mm_loadu_si128(a2 + 2);
    v12 = _mm_loadu_si128(a2 + 3);
    v16[0] = _mm_loadu_si128(a2);
    v16[1] = v10;
    v16[2] = v11;
    v16[3] = v12;
    if ( v5 < v3 + 1 )
    {
      v14 = (const void *)(a1 + 16);
      if ( v4 > (unsigned __int64)v16 || result <= (__m128i *)v16 )
      {
        result = (__m128i *)sub_C8D5F0(a1, v14, v8, 0x40u, v4, v5);
        v4 = *(_QWORD *)a1;
        v3 = *(unsigned int *)(a1 + 8);
      }
      else
      {
        v15 = (char *)v16 - v4;
        result = (__m128i *)sub_C8D5F0(a1, v14, v8, 0x40u, v4, v5);
        v4 = *(_QWORD *)a1;
        v3 = *(unsigned int *)(a1 + 8);
        v9 = (const __m128i *)&v15[*(_QWORD *)a1];
      }
    }
    v13 = (__m128i *)(v4 + (v3 << 6));
    *v13 = _mm_loadu_si128(v9);
    v13[1] = _mm_loadu_si128(v9 + 1);
    v13[2] = _mm_loadu_si128(v9 + 2);
    v13[3] = _mm_loadu_si128(v9 + 3);
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    v7 = *(_DWORD *)(a1 + 8);
    if ( result )
    {
      *result = _mm_loadu_si128(a2);
      result[1] = _mm_loadu_si128(a2 + 1);
      result[2] = _mm_loadu_si128(a2 + 2);
      result[3] = _mm_loadu_si128(a2 + 3);
      v7 = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = v7 + 1;
  }
  return result;
}
