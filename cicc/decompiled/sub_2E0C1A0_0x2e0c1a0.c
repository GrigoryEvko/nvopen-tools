// Function: sub_2E0C1A0
// Address: 0x2e0c1a0
//
__int64 __fastcall sub_2E0C1A0(__int64 a1, __m128i *a2, const __m128i *a3)
{
  const __m128i *v4; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rcx
  __int64 v8; // rdi
  unsigned __int64 v9; // r8
  unsigned __int64 v10; // rax
  __int64 v11; // rdi
  __m128i *v12; // rsi
  const __m128i *v13; // rax
  __int64 v14; // rdx
  __int64 v16; // rax
  __int8 *v17; // r13
  __int64 v18; // r9
  __int8 *v19; // r12
  __int64 v20; // r9
  __int8 *v21; // r12

  v4 = a3;
  v6 = *(unsigned int *)(a1 + 8);
  v7 = *(_QWORD *)a1;
  v8 = 3 * v6;
  LODWORD(a3) = v6;
  v9 = v6 + 1;
  v10 = *(unsigned int *)(a1 + 12);
  v11 = 8 * v8;
  v12 = (__m128i *)(v7 + v11);
  if ( a2 == (__m128i *)(v7 + v11) )
  {
    if ( v9 > v10 )
    {
      v20 = a1 + 16;
      if ( v7 > (unsigned __int64)v4 || v12 <= v4 )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v9, 0x18u, v9, v20);
        v12 = (__m128i *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
      }
      else
      {
        v21 = &v4->m128i_i8[-v7];
        sub_C8D5F0(a1, (const void *)(a1 + 16), v9, 0x18u, v9, v20);
        v4 = (const __m128i *)&v21[*(_QWORD *)a1];
        v12 = (__m128i *)(*(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8));
      }
    }
    *v12 = _mm_loadu_si128(v4);
    v12[1].m128i_i64[0] = v4[1].m128i_i64[0];
    v16 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
    *(_DWORD *)(a1 + 8) = v16;
    return *(_QWORD *)a1 + 24 * v16 - 24;
  }
  else
  {
    if ( v9 > v10 )
    {
      v17 = &a2->m128i_i8[-v7];
      v18 = a1 + 16;
      if ( v7 > (unsigned __int64)v4 || v12 <= v4 )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v9, 0x18u, v9, v18);
        v7 = *(_QWORD *)a1;
        LODWORD(a3) = *(_DWORD *)(a1 + 8);
        a2 = (__m128i *)&v17[*(_QWORD *)a1];
        v11 = 24LL * (unsigned int)a3;
        v12 = (__m128i *)(*(_QWORD *)a1 + v11);
      }
      else
      {
        v19 = &v4->m128i_i8[-v7];
        sub_C8D5F0(a1, (const void *)(a1 + 16), v9, 0x18u, v9, v18);
        v7 = *(_QWORD *)a1;
        a3 = (const __m128i *)*(unsigned int *)(a1 + 8);
        v4 = (const __m128i *)&v19[*(_QWORD *)a1];
        a2 = (__m128i *)&v17[*(_QWORD *)a1];
        v11 = 24LL * (_QWORD)a3;
        v12 = (__m128i *)(*(_QWORD *)a1 + 24LL * (_QWORD)a3);
      }
    }
    v13 = (const __m128i *)(v7 + v11 - 24);
    if ( v12 )
    {
      *v12 = _mm_loadu_si128(v13);
      v12[1].m128i_i64[0] = v13[1].m128i_i64[0];
      v7 = *(_QWORD *)a1;
      LODWORD(a3) = *(_DWORD *)(a1 + 8);
      v11 = 24LL * (unsigned int)a3;
      v13 = (const __m128i *)(*(_QWORD *)a1 + v11 - 24);
    }
    if ( a2 != v13 )
    {
      memmove((void *)(v7 + v11 - ((char *)v13 - (char *)a2)), a2, (char *)v13 - (char *)a2);
      v7 = *(_QWORD *)a1;
      LODWORD(a3) = *(_DWORD *)(a1 + 8);
    }
    v14 = (unsigned int)((_DWORD)a3 + 1);
    *(_DWORD *)(a1 + 8) = v14;
    if ( a2 <= v4 && (unsigned __int64)v4 < v7 + 24 * v14 )
      v4 = (const __m128i *)((char *)v4 + 24);
    *a2 = _mm_loadu_si128(v4);
    a2[1].m128i_i64[0] = v4[1].m128i_i64[0];
    return (__int64)a2;
  }
}
