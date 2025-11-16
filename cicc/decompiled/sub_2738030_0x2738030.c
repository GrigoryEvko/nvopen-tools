// Function: sub_2738030
// Address: 0x2738030
//
__m128i *__fastcall sub_2738030(__int64 a1, const __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // r8
  __m128i *result; // rax
  const void *v11; // rsi
  __int8 *v12; // r12

  v7 = *(unsigned int *)(a1 + 8);
  v8 = *(_QWORD *)a1;
  v9 = v7 + 1;
  if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    v11 = (const void *)(a1 + 16);
    if ( v8 > (unsigned __int64)a2 || (unsigned __int64)a2 >= v8 + (v7 << 6) )
    {
      sub_C8D5F0(a1, v11, v9, 0x40u, v9, a6);
      v8 = *(_QWORD *)a1;
      v7 = *(unsigned int *)(a1 + 8);
    }
    else
    {
      v12 = &a2->m128i_i8[-v8];
      sub_C8D5F0(a1, v11, v9, 0x40u, v9, a6);
      v8 = *(_QWORD *)a1;
      v7 = *(unsigned int *)(a1 + 8);
      a2 = (const __m128i *)&v12[*(_QWORD *)a1];
    }
  }
  result = (__m128i *)(v8 + (v7 << 6));
  *result = _mm_loadu_si128(a2);
  result[1] = _mm_loadu_si128(a2 + 1);
  result[2] = _mm_loadu_si128(a2 + 2);
  result[3] = _mm_loadu_si128(a2 + 3);
  ++*(_DWORD *)(a1 + 8);
  return result;
}
