// Function: sub_308A020
// Address: 0x308a020
//
__int64 __fastcall sub_308A020(__int64 a1, __m128i *a2, const __m128i *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v9; // rdx
  __int64 result; // rax
  unsigned __int64 v11; // r8
  unsigned __int64 v12; // rcx
  __int64 v13; // rdi
  __m128i *v14; // rsi
  const __m128i *v15; // rcx
  __int64 v16; // rax
  __int8 *v17; // r12
  __int64 v18; // r9
  __int8 *v19; // r13
  const void *v20; // rsi
  __int8 *v21; // r13

  v9 = *(_QWORD *)a1;
  result = *(unsigned int *)(a1 + 8);
  v11 = result + 1;
  v12 = *(unsigned int *)(a1 + 12);
  v13 = 32 * result;
  v14 = (__m128i *)(v9 + 32 * result);
  if ( v14 == a2 )
  {
    if ( v11 > v12 )
    {
      v20 = (const void *)(a1 + 16);
      if ( v9 > (unsigned __int64)a3 || a2 <= a3 )
      {
        result = sub_C8D5F0(a1, v20, result + 1, 0x20u, v11, a6);
        a2 = (__m128i *)(*(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8));
      }
      else
      {
        v21 = &a3->m128i_i8[-v9];
        result = sub_C8D5F0(a1, v20, result + 1, 0x20u, v11, a6);
        a3 = (const __m128i *)&v21[*(_QWORD *)a1];
        a2 = (__m128i *)(*(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8));
      }
    }
    *a2 = _mm_loadu_si128(a3);
    a2[1] = _mm_loadu_si128(a3 + 1);
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    if ( v11 > v12 )
    {
      v17 = &a2->m128i_i8[-v9];
      v18 = a1 + 16;
      if ( v9 > (unsigned __int64)a3 || v14 <= a3 )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), result + 1, 0x20u, v11, v18);
        v9 = *(_QWORD *)a1;
      }
      else
      {
        v19 = &a3->m128i_i8[-v9];
        sub_C8D5F0(a1, (const void *)(a1 + 16), result + 1, 0x20u, v11, v18);
        v9 = *(_QWORD *)a1;
        a3 = (const __m128i *)&v19[*(_QWORD *)a1];
      }
      a2 = (__m128i *)&v17[v9];
      result = *(unsigned int *)(a1 + 8);
      v13 = 32 * result;
      v14 = (__m128i *)(v9 + 32 * result);
    }
    v15 = (const __m128i *)(v9 + v13 - 32);
    if ( v14 )
    {
      *v14 = _mm_loadu_si128(v15);
      v14[1] = _mm_loadu_si128(v15 + 1);
      v9 = *(_QWORD *)a1;
      result = *(unsigned int *)(a1 + 8);
      v13 = 32 * result;
      v15 = (const __m128i *)(*(_QWORD *)a1 + 32 * result - 32);
    }
    if ( a2 != v15 )
    {
      memmove((void *)(v9 + v13 - ((char *)v15 - (char *)a2)), a2, (char *)v15 - (char *)a2);
      LODWORD(result) = *(_DWORD *)(a1 + 8);
      v9 = *(_QWORD *)a1;
    }
    v16 = (unsigned int)(result + 1);
    *(_DWORD *)(a1 + 8) = v16;
    if ( a2 <= a3 && (unsigned __int64)a3 < v9 + 32 * v16 )
      a3 += 2;
    *a2 = _mm_loadu_si128(a3);
    a2[1].m128i_i64[0] = a3[1].m128i_i64[0];
    result = a3[1].m128i_u32[2];
    a2[1].m128i_i32[2] = result;
  }
  return result;
}
