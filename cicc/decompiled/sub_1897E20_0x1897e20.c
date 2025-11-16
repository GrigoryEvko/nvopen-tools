// Function: sub_1897E20
// Address: 0x1897e20
//
void __fastcall sub_1897E20(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  _QWORD *v4; // r15
  unsigned __int64 v5; // rdx
  _QWORD *v6; // rbx
  __m128i *v7; // rbx
  __int64 v8; // r15
  unsigned __int64 i; // r12
  _QWORD *v10; // rax
  _QWORD *v11; // rbx
  _QWORD *v12; // rdi
  _QWORD *v13; // r15
  _QWORD *v14; // rdi
  const __m128i *v15; // rbx
  _QWORD *v16; // r12
  _QWORD *v17; // rdi
  __m128i v18; // xmm2
  const __m128i *v19; // r14
  _QWORD *v20; // rbx
  _QWORD *v21; // rdi
  __m128i v22; // xmm1
  unsigned int v23; // [rsp-44h] [rbp-44h]
  unsigned __int64 v24; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v4 = *(_QWORD **)a1;
    v5 = *(unsigned int *)(a1 + 8);
    v23 = *(_DWORD *)(a2 + 8);
    v3 = v23;
    v6 = *(_QWORD **)a1;
    if ( v23 <= v5 )
    {
      v10 = *(_QWORD **)a1;
      if ( v23 )
      {
        v15 = *(const __m128i **)a2;
        v16 = &v4[11 * v23];
        do
        {
          sub_2240AE0(v4, v15);
          v17 = v4 + 4;
          v4 += 11;
          sub_2240AE0(v17, &v15[2]);
          v18 = _mm_loadu_si128(v15 + 4);
          v15 = (const __m128i *)((char *)v15 + 88);
          *(__m128i *)(v4 - 3) = v18;
          *(v4 - 1) = v15[-1].m128i_i64[1];
        }
        while ( v4 != v16 );
        v10 = *(_QWORD **)a1;
        v5 = *(unsigned int *)(a1 + 8);
      }
      v11 = &v10[11 * v5];
      while ( v4 != v11 )
      {
        v11 -= 11;
        v12 = (_QWORD *)v11[4];
        if ( v12 != v11 + 6 )
          j_j___libc_free_0(v12, v11[6] + 1LL);
        if ( (_QWORD *)*v11 != v11 + 2 )
          j_j___libc_free_0(*v11, v11[2] + 1LL);
      }
    }
    else
    {
      if ( v23 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        v13 = &v4[11 * v5];
        while ( v13 != v6 )
        {
          while ( 1 )
          {
            v13 -= 11;
            v14 = (_QWORD *)v13[4];
            if ( v14 != v13 + 6 )
              j_j___libc_free_0(v14, v13[6] + 1LL);
            if ( (_QWORD *)*v13 == v13 + 2 )
              break;
            j_j___libc_free_0(*v13, v13[2] + 1LL);
            if ( v13 == v6 )
              goto LABEL_24;
          }
        }
LABEL_24:
        *(_DWORD *)(a1 + 8) = 0;
        sub_14B3F20(a1, v23);
        v3 = *(unsigned int *)(a2 + 8);
        v6 = *(_QWORD **)a1;
        v5 = 0;
      }
      else if ( *(_DWORD *)(a1 + 8) )
      {
        v19 = *(const __m128i **)a2;
        v5 *= 88LL;
        v20 = (_QWORD *)((char *)v4 + v5);
        do
        {
          v24 = v5;
          sub_2240AE0(v4, v19);
          v21 = v4 + 4;
          v4 += 11;
          sub_2240AE0(v21, &v19[2]);
          v22 = _mm_loadu_si128(v19 + 4);
          v19 = (const __m128i *)((char *)v19 + 88);
          v5 = v24;
          *(__m128i *)(v4 - 3) = v22;
          *(v4 - 1) = v19[-1].m128i_i64[1];
        }
        while ( v20 != v4 );
        v3 = *(unsigned int *)(a2 + 8);
        v6 = *(_QWORD **)a1;
      }
      v7 = (__m128i *)((char *)v6 + v5);
      v8 = *(_QWORD *)a2 + 88 * v3;
      for ( i = v5 + *(_QWORD *)a2; v8 != i; v7 = (__m128i *)((char *)v7 + 88) )
      {
        if ( v7 )
        {
          v7->m128i_i64[0] = (__int64)v7[1].m128i_i64;
          sub_1897040(v7->m128i_i64, *(_BYTE **)i, *(_QWORD *)i + *(_QWORD *)(i + 8));
          v7[2].m128i_i64[0] = (__int64)v7[3].m128i_i64;
          sub_1897040(v7[2].m128i_i64, *(_BYTE **)(i + 32), *(_QWORD *)(i + 32) + *(_QWORD *)(i + 40));
          v7[4] = _mm_loadu_si128((const __m128i *)(i + 64));
          v7[5].m128i_i64[0] = *(_QWORD *)(i + 80);
        }
        i += 88LL;
      }
    }
    *(_DWORD *)(a1 + 8) = v23;
  }
}
