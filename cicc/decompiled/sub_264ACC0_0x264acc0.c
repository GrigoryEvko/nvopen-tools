// Function: sub_264ACC0
// Address: 0x264acc0
//
void __fastcall sub_264ACC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  unsigned __int64 *v8; // r12
  unsigned __int64 v9; // rdx
  unsigned __int64 *v10; // rbx
  __m128i *v11; // rbx
  __int64 v12; // r15
  unsigned __int64 j; // r12
  unsigned __int64 *v14; // rax
  unsigned __int64 *v15; // rbx
  unsigned __int64 v16; // rdi
  unsigned __int64 *i; // r12
  __m128i *v18; // rbx
  unsigned __int64 *v19; // r15
  unsigned __int64 *v20; // rdi
  __m128i v21; // xmm2
  __m128i *v22; // r14
  unsigned __int64 *v23; // rbx
  unsigned __int64 *v24; // rdi
  __m128i v25; // xmm1
  unsigned int v26; // [rsp-44h] [rbp-44h]
  unsigned __int64 v27; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v8 = *(unsigned __int64 **)a1;
    v9 = *(unsigned int *)(a1 + 8);
    v26 = *(_DWORD *)(a2 + 8);
    v7 = v26;
    v10 = *(unsigned __int64 **)a1;
    if ( v26 <= v9 )
    {
      v14 = *(unsigned __int64 **)a1;
      if ( v26 )
      {
        v18 = *(__m128i **)a2;
        v19 = &v8[10 * v26];
        do
        {
          sub_2240AE0(v8, (unsigned __int64 *)v18);
          v20 = v8 + 4;
          v8 += 10;
          sub_2240AE0(v20, (unsigned __int64 *)&v18[2]);
          v21 = _mm_loadu_si128(v18 + 4);
          v18 += 5;
          *((__m128i *)v8 - 1) = v21;
        }
        while ( v8 != v19 );
        v14 = *(unsigned __int64 **)a1;
        v9 = *(unsigned int *)(a1 + 8);
      }
      v15 = &v14[10 * v9];
      while ( v8 != v15 )
      {
        v15 -= 10;
        v16 = v15[4];
        if ( (unsigned __int64 *)v16 != v15 + 6 )
          j_j___libc_free_0(v16);
        if ( (unsigned __int64 *)*v15 != v15 + 2 )
          j_j___libc_free_0(*v15);
      }
    }
    else
    {
      if ( v26 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        for ( i = &v8[10 * v9]; i != v10; sub_2240A30(i) )
        {
          i -= 10;
          sub_2240A30(i + 4);
        }
        *(_DWORD *)(a1 + 8) = 0;
        sub_11F02D0(a1, v26, v9, a4, a5, a6);
        v7 = *(unsigned int *)(a2 + 8);
        v10 = *(unsigned __int64 **)a1;
        v9 = 0;
      }
      else if ( *(_DWORD *)(a1 + 8) )
      {
        v22 = *(__m128i **)a2;
        v9 *= 80LL;
        v23 = (unsigned __int64 *)((char *)v8 + v9);
        do
        {
          v27 = v9;
          sub_2240AE0(v8, (unsigned __int64 *)v22);
          v24 = v8 + 4;
          v8 += 10;
          sub_2240AE0(v24, (unsigned __int64 *)&v22[2]);
          v25 = _mm_loadu_si128(v22 + 4);
          v22 += 5;
          v9 = v27;
          *((__m128i *)v8 - 1) = v25;
        }
        while ( v8 != v23 );
        v7 = *(unsigned int *)(a2 + 8);
        v10 = *(unsigned __int64 **)a1;
      }
      v11 = (__m128i *)((char *)v10 + v9);
      v12 = *(_QWORD *)a2 + 80 * v7;
      for ( j = v9 + *(_QWORD *)a2; v12 != j; v11 += 5 )
      {
        if ( v11 )
        {
          v11->m128i_i64[0] = (__int64)v11[1].m128i_i64;
          sub_2640340(v11->m128i_i64, *(_BYTE **)j, *(_QWORD *)j + *(_QWORD *)(j + 8));
          v11[2].m128i_i64[0] = (__int64)v11[3].m128i_i64;
          sub_2640340(v11[2].m128i_i64, *(_BYTE **)(j + 32), *(_QWORD *)(j + 32) + *(_QWORD *)(j + 40));
          v11[4] = _mm_loadu_si128((const __m128i *)(j + 64));
        }
        j += 80LL;
      }
    }
    *(_DWORD *)(a1 + 8) = v26;
  }
}
