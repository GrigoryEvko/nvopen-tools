// Function: sub_395A000
// Address: 0x395a000
//
__int64 __fastcall sub_395A000(__int64 a1, __m128i *a2, const __m128i *a3, __int64 a4, __int64 a5, int a6)
{
  unsigned __int64 v9; // rcx
  __int64 v10; // rdx
  unsigned __int64 v11; // r8
  unsigned int v12; // eax
  __int64 v13; // rdi
  __m128i *v14; // rsi
  const __m128i *v15; // rcx
  __int64 v16; // rax
  __int64 result; // rax
  __int8 *v18; // r12

  v9 = *(unsigned int *)(a1 + 8);
  v10 = *(_QWORD *)a1;
  v11 = *(unsigned int *)(a1 + 12);
  v12 = *(_DWORD *)(a1 + 8);
  v13 = 24 * v9;
  v14 = (__m128i *)(v10 + 24 * v9);
  if ( v14 == a2 )
  {
    if ( (unsigned int)v9 >= (unsigned int)v11 )
    {
      sub_16CD150(a1, (const void *)(a1 + 16), 0, 24, v11, a6);
      a2 = (__m128i *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
    }
    *a2 = _mm_loadu_si128(a3);
    result = a3[1].m128i_i64[0];
    a2[1].m128i_i64[0] = result;
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    if ( v9 >= v11 )
    {
      v18 = &a2->m128i_i8[-v10];
      sub_16CD150(a1, (const void *)(a1 + 16), 0, 24, v11, a6);
      v10 = *(_QWORD *)a1;
      a2 = (__m128i *)&v18[*(_QWORD *)a1];
      v12 = *(_DWORD *)(a1 + 8);
      v13 = 24LL * v12;
      v14 = (__m128i *)(*(_QWORD *)a1 + v13);
    }
    v15 = (const __m128i *)(v10 + v13 - 24);
    if ( v14 )
    {
      *v14 = _mm_loadu_si128(v15);
      v14[1].m128i_i64[0] = v15[1].m128i_i64[0];
      v10 = *(_QWORD *)a1;
      v12 = *(_DWORD *)(a1 + 8);
      v13 = 24LL * v12;
      v15 = (const __m128i *)(*(_QWORD *)a1 + v13 - 24);
    }
    if ( a2 != v15 )
    {
      memmove((void *)(v10 + v13 - ((char *)v15 - (char *)a2)), a2, (char *)v15 - (char *)a2);
      v12 = *(_DWORD *)(a1 + 8);
    }
    v16 = v12 + 1;
    *(_DWORD *)(a1 + 8) = v16;
    if ( a3 >= a2 && (unsigned __int64)a3 < *(_QWORD *)a1 + 24 * v16 )
      a3 = (const __m128i *)((char *)a3 + 24);
    *a2 = _mm_loadu_si128(a3);
    result = a3[1].m128i_i64[0];
    a2[1].m128i_i64[0] = result;
  }
  return result;
}
