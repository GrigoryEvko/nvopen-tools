// Function: sub_1ECEFD0
// Address: 0x1ecefd0
//
__int64 __fastcall sub_1ECEFD0(__int64 *a1, const __m128i *a2, __int64 *a3)
{
  const __m128i *v5; // rbx
  __int64 v6; // rsi
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdx
  bool v10; // cf
  unsigned __int64 v11; // rax
  char *v12; // rdx
  __int64 v13; // rcx
  char *v14; // rax
  __int64 v15; // rdx
  __m128i v16; // xmm2
  __int64 v17; // rdx
  __int64 v18; // rdx
  const __m128i *v19; // r14
  __int64 i; // r13
  __int64 v21; // rcx
  __int64 v22; // rcx
  volatile signed __int32 *v23; // rdi
  const __m128i *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdi
  __m128i v27; // xmm0
  __int64 v28; // rdi
  const __m128i *v29; // rdi
  __int64 v31; // r13
  __int64 v32; // rax
  __int64 *v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+18h] [rbp-48h]
  const __m128i *v35; // [rsp+20h] [rbp-40h]
  __int64 v36; // [rsp+20h] [rbp-40h]
  __int64 v37; // [rsp+28h] [rbp-38h]

  v5 = (const __m128i *)a1[1];
  v35 = (const __m128i *)*a1;
  v6 = (__int64)v5->m128i_i64 - *a1;
  v7 = 0xAAAAAAAAAAAAAAABLL * (v6 >> 4);
  if ( v7 == 0x2AAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v7 )
    v8 = 0xAAAAAAAAAAAAAAABLL * (v6 >> 4);
  v10 = __CFADD__(v8, v7);
  v11 = v8 - 0x5555555555555555LL * (v6 >> 4);
  v12 = (char *)((char *)a2 - (char *)v35);
  if ( v10 )
  {
    v31 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v11 )
    {
      v34 = 0;
      v13 = 48;
      v37 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0x2AAAAAAAAAAAAAALL )
      v11 = 0x2AAAAAAAAAAAAAALL;
    v31 = 48 * v11;
  }
  v33 = a3;
  v32 = sub_22077B0(v31);
  v12 = (char *)((char *)a2 - (char *)v35);
  a3 = v33;
  v37 = v32;
  v13 = v32 + 48;
  v34 = v32 + v31;
LABEL_7:
  v14 = &v12[v37];
  if ( &v12[v37] )
  {
    v15 = *a3;
    v16 = _mm_loadu_si128((const __m128i *)a3 + 2);
    *a3 = 0;
    *(_QWORD *)v14 = v15;
    v17 = a3[1];
    a3[1] = 0;
    *((_QWORD *)v14 + 1) = v17;
    v18 = *(__int64 *)((char *)a3 + 20);
    *((__m128i *)v14 + 2) = v16;
    *(_QWORD *)(v14 + 20) = v18;
  }
  v19 = v35;
  if ( a2 != v35 )
  {
    for ( i = v37; ; i += 48 )
    {
      if ( i )
      {
        *(_QWORD *)i = v19->m128i_i64[0];
        v21 = v19->m128i_i64[1];
        v19->m128i_i64[1] = 0;
        *(_QWORD *)(i + 8) = v21;
        v22 = *(__int64 *)((char *)v19[1].m128i_i64 + 4);
        v19->m128i_i64[0] = 0;
        *(_QWORD *)(i + 20) = v22;
        *(__m128i *)(i + 32) = _mm_loadu_si128(v19 + 2);
      }
      v23 = (volatile signed __int32 *)v19->m128i_i64[1];
      if ( v23 )
        sub_A191D0(v23);
      v19 += 3;
      if ( v19 == a2 )
        break;
    }
    v13 = i + 96;
  }
  if ( a2 != v5 )
  {
    v24 = a2;
    v25 = v13;
    do
    {
      v26 = v24->m128i_i64[0];
      v27 = _mm_loadu_si128(v24 + 2);
      v24 += 3;
      v25 += 48;
      *(_QWORD *)(v25 - 48) = v26;
      v28 = v24[-3].m128i_i64[1];
      *(__m128i *)(v25 - 16) = v27;
      *(_QWORD *)(v25 - 40) = v28;
      *(_QWORD *)(v25 - 28) = *(__int64 *)((char *)v24[-2].m128i_i64 + 4);
    }
    while ( v24 != v5 );
    v13 += 16
         * (3 * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)v24 - (char *)a2 - 48) >> 4)) & 0xFFFFFFFFFFFFFFFLL)
          + 3);
  }
  v29 = v35;
  if ( v35 )
  {
    v36 = v13;
    j_j___libc_free_0(v29, a1[2] - (_QWORD)v29);
    v13 = v36;
  }
  a1[1] = v13;
  *a1 = v37;
  a1[2] = v34;
  return v34;
}
