// Function: sub_2CB0E10
// Address: 0x2cb0e10
//
__int64 __fastcall sub_2CB0E10(__int64 a1, __m128i *a2, const __m128i *a3, __int64 a4, __int64 a5, __int64 a6)
{
  const __m128i *v7; // r12
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  __int64 v11; // rdi
  unsigned __int64 v12; // r8
  unsigned __int64 v13; // rax
  __int64 v14; // rdi
  __m128i *v15; // rsi
  const __m128i *v16; // rax
  __int64 v17; // rdx
  __int64 result; // rax
  __int8 *v19; // r13
  __int64 v20; // r9
  __int8 *v21; // r12
  const void *v22; // rsi
  __int8 *v23; // r12

  v7 = a3;
  v9 = *(unsigned int *)(a1 + 8);
  v10 = *(_QWORD *)a1;
  v11 = 3 * v9;
  LODWORD(a3) = v9;
  v12 = v9 + 1;
  v13 = *(unsigned int *)(a1 + 12);
  v14 = 8 * v11;
  v15 = (__m128i *)(v10 + v14);
  if ( (__m128i *)(v10 + v14) == a2 )
  {
    if ( v12 > v13 )
    {
      v22 = (const void *)(a1 + 16);
      if ( v10 > (unsigned __int64)v7 || a2 <= v7 )
      {
        sub_C8D5F0(a1, v22, v12, 0x18u, v12, a6);
        a2 = (__m128i *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
      }
      else
      {
        v23 = &v7->m128i_i8[-v10];
        sub_C8D5F0(a1, v22, v12, 0x18u, v12, a6);
        v7 = (const __m128i *)&v23[*(_QWORD *)a1];
        a2 = (__m128i *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
      }
    }
    *a2 = _mm_loadu_si128(v7);
    result = v7[1].m128i_i64[0];
    a2[1].m128i_i64[0] = result;
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    if ( v12 > v13 )
    {
      v19 = &a2->m128i_i8[-v10];
      v20 = a1 + 16;
      if ( v10 > (unsigned __int64)v7 || v15 <= v7 )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v12, 0x18u, v12, v20);
        v10 = *(_QWORD *)a1;
      }
      else
      {
        v21 = &v7->m128i_i8[-v10];
        sub_C8D5F0(a1, (const void *)(a1 + 16), v12, 0x18u, v12, v20);
        v10 = *(_QWORD *)a1;
        v7 = (const __m128i *)&v21[*(_QWORD *)a1];
      }
      a2 = (__m128i *)&v19[v10];
      a3 = (const __m128i *)*(unsigned int *)(a1 + 8);
      v14 = 24LL * (_QWORD)a3;
      v15 = (__m128i *)(v10 + 24LL * (_QWORD)a3);
    }
    v16 = (const __m128i *)(v10 + v14 - 24);
    if ( v15 )
    {
      *v15 = _mm_loadu_si128(v16);
      v15[1].m128i_i64[0] = v16[1].m128i_i64[0];
      v10 = *(_QWORD *)a1;
      LODWORD(a3) = *(_DWORD *)(a1 + 8);
      v14 = 24LL * (unsigned int)a3;
      v16 = (const __m128i *)(*(_QWORD *)a1 + v14 - 24);
    }
    if ( a2 != v16 )
    {
      memmove((void *)(v10 + v14 - ((char *)v16 - (char *)a2)), a2, (char *)v16 - (char *)a2);
      LODWORD(a3) = *(_DWORD *)(a1 + 8);
      v10 = *(_QWORD *)a1;
    }
    v17 = (unsigned int)((_DWORD)a3 + 1);
    *(_DWORD *)(a1 + 8) = v17;
    if ( a2 <= v7 && (unsigned __int64)v7 < v10 + 24 * v17 )
      v7 = (const __m128i *)((char *)v7 + 24);
    *a2 = _mm_loadu_si128(v7);
    result = v7[1].m128i_i64[0];
    a2[1].m128i_i64[0] = result;
  }
  return result;
}
