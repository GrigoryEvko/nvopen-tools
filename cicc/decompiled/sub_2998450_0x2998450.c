// Function: sub_2998450
// Address: 0x2998450
//
void __fastcall sub_2998450(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  unsigned __int64 *v8; // r12
  unsigned __int64 v9; // rdx
  unsigned __int64 *v10; // rbx
  __m128i *v11; // rbx
  __int64 v12; // r15
  unsigned __int64 i; // r12
  unsigned __int64 *v14; // rax
  unsigned __int64 *v15; // rbx
  unsigned __int64 v16; // rdi
  unsigned __int64 *v17; // r12
  unsigned __int64 v18; // rdi
  __m128i *v19; // rbx
  unsigned __int64 *v20; // r15
  unsigned __int64 *v21; // rdi
  __m128i v22; // xmm2
  __m128i *v23; // r14
  unsigned __int64 *v24; // rbx
  unsigned __int64 *v25; // rdi
  __m128i v26; // xmm1
  unsigned int v27; // [rsp-44h] [rbp-44h]
  unsigned __int64 v28; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v8 = *(unsigned __int64 **)a1;
    v9 = *(unsigned int *)(a1 + 8);
    v27 = *(_DWORD *)(a2 + 8);
    v7 = v27;
    v10 = *(unsigned __int64 **)a1;
    if ( v27 <= v9 )
    {
      v14 = *(unsigned __int64 **)a1;
      if ( v27 )
      {
        v19 = *(__m128i **)a2;
        v20 = &v8[10 * v27];
        do
        {
          sub_2240AE0(v8, (unsigned __int64 *)v19);
          v21 = v8 + 4;
          v8 += 10;
          sub_2240AE0(v21, (unsigned __int64 *)&v19[2]);
          v22 = _mm_loadu_si128(v19 + 4);
          v19 += 5;
          *((__m128i *)v8 - 1) = v22;
        }
        while ( v8 != v20 );
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
      if ( v27 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        v17 = &v8[10 * v9];
        while ( v17 != v10 )
        {
          while ( 1 )
          {
            v17 -= 10;
            v18 = v17[4];
            if ( (unsigned __int64 *)v18 != v17 + 6 )
              j_j___libc_free_0(v18);
            if ( (unsigned __int64 *)*v17 == v17 + 2 )
              break;
            j_j___libc_free_0(*v17);
            if ( v17 == v10 )
              goto LABEL_24;
          }
        }
LABEL_24:
        *(_DWORD *)(a1 + 8) = 0;
        sub_11F02D0(a1, v27, v9, a4, a5, a6);
        v7 = *(unsigned int *)(a2 + 8);
        v10 = *(unsigned __int64 **)a1;
        v9 = 0;
      }
      else if ( *(_DWORD *)(a1 + 8) )
      {
        v23 = *(__m128i **)a2;
        v9 *= 80LL;
        v24 = (unsigned __int64 *)((char *)v8 + v9);
        do
        {
          v28 = v9;
          sub_2240AE0(v8, (unsigned __int64 *)v23);
          v25 = v8 + 4;
          v8 += 10;
          sub_2240AE0(v25, (unsigned __int64 *)&v23[2]);
          v26 = _mm_loadu_si128(v23 + 4);
          v23 += 5;
          v9 = v28;
          *((__m128i *)v8 - 1) = v26;
        }
        while ( v24 != v8 );
        v7 = *(unsigned int *)(a2 + 8);
        v10 = *(unsigned __int64 **)a1;
      }
      v11 = (__m128i *)((char *)v10 + v9);
      v12 = *(_QWORD *)a2 + 80 * v7;
      for ( i = v9 + *(_QWORD *)a2; v12 != i; v11 += 5 )
      {
        if ( v11 )
        {
          v11->m128i_i64[0] = (__int64)v11[1].m128i_i64;
          sub_2996E50(v11->m128i_i64, *(_BYTE **)i, *(_QWORD *)i + *(_QWORD *)(i + 8));
          v11[2].m128i_i64[0] = (__int64)v11[3].m128i_i64;
          sub_2996E50(v11[2].m128i_i64, *(_BYTE **)(i + 32), *(_QWORD *)(i + 32) + *(_QWORD *)(i + 40));
          v11[4] = _mm_loadu_si128((const __m128i *)(i + 64));
        }
        i += 80LL;
      }
    }
    *(_DWORD *)(a1 + 8) = v27;
  }
}
