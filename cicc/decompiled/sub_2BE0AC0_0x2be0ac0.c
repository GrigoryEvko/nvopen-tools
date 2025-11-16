// Function: sub_2BE0AC0
// Address: 0x2be0ac0
//
unsigned __int64 *__fastcall sub_2BE0AC0(unsigned __int64 *a1, char *a2, _QWORD *a3, const __m128i **a4)
{
  char *v6; // r12
  unsigned __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rax
  char *v11; // r14
  unsigned __int64 v12; // rbx
  _BOOL8 v13; // rax
  char *v14; // rcx
  bool v15; // zf
  char *v16; // rcx
  char *v17; // rbx
  const __m128i *v18; // r10
  __int64 v19; // rax
  const __m128i *v20; // r9
  unsigned __int64 v21; // r13
  __int64 v22; // rdi
  __m128i *v23; // rsi
  const __m128i *v24; // rax
  char *v25; // r13
  __int64 i; // rbx
  unsigned __int64 v27; // rdi
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rsi
  unsigned __int64 v32; // r10
  __int64 v33; // rax
  _QWORD *v34; // [rsp+10h] [rbp-60h]
  __int64 v35; // [rsp+18h] [rbp-58h]
  __int64 v36; // [rsp+20h] [rbp-50h]
  unsigned __int64 v38; // [rsp+30h] [rbp-40h]
  __int64 v39; // [rsp+38h] [rbp-38h]

  v6 = (char *)a1[1];
  v7 = *a1;
  v8 = 0x3FFFFFFFFFFFFFFLL;
  v38 = v7;
  v9 = (__int64)&v6[-v7] >> 5;
  if ( v9 == 0x3FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = 1;
  v11 = a2;
  if ( v9 )
    v10 = v9;
  v12 = v9 + v10;
  v13 = __CFADD__(v9, v10);
  v36 = v12;
  v14 = &a2[-v38];
  if ( v13 )
  {
    v32 = 0x7FFFFFFFFFFFFFE0LL;
    v36 = 0x3FFFFFFFFFFFFFFLL;
  }
  else
  {
    if ( !v12 )
    {
      v39 = 0;
      goto LABEL_7;
    }
    if ( v12 <= 0x3FFFFFFFFFFFFFFLL )
      v8 = v12;
    v36 = v8;
    v32 = 32 * v8;
  }
  v8 = v32;
  v34 = a3;
  v33 = sub_22077B0(v32);
  v14 = &a2[-v38];
  a3 = v34;
  v39 = v33;
LABEL_7:
  v15 = &v14[v39] == 0;
  v16 = &v14[v39];
  v17 = v16;
  if ( !v15 )
  {
    v18 = a4[1];
    v19 = *a3;
    *((_QWORD *)v16 + 1) = 0;
    v20 = *a4;
    *((_QWORD *)v16 + 2) = 0;
    *(_QWORD *)v16 = v19;
    *((_QWORD *)v16 + 3) = 0;
    v21 = (char *)v18 - (char *)v20;
    if ( v18 == v20 )
    {
      v22 = 0;
    }
    else
    {
      if ( v21 > 0x7FFFFFFFFFFFFFF8LL )
        sub_4261EA(v8, a4, a3);
      v22 = sub_22077B0((char *)v18 - (char *)v20);
      v18 = a4[1];
      v20 = *a4;
    }
    *((_QWORD *)v17 + 1) = v22;
    *((_QWORD *)v17 + 2) = v22;
    *((_QWORD *)v17 + 3) = v22 + v21;
    if ( v20 != v18 )
    {
      v23 = (__m128i *)v22;
      v24 = v20;
      do
      {
        if ( v23 )
        {
          *v23 = _mm_loadu_si128(v24);
          v23[1].m128i_i64[0] = v24[1].m128i_i64[0];
        }
        v24 = (const __m128i *)((char *)v24 + 24);
        v23 = (__m128i *)((char *)v23 + 24);
      }
      while ( v24 != v18 );
      v22 += 8 * ((unsigned __int64)((char *)&v24[-2].m128i_u64[1] - (char *)v20) >> 3) + 24;
    }
    *((_QWORD *)v17 + 2) = v22;
  }
  v25 = (char *)v38;
  for ( i = v39; v25 != a2; i = 32 )
  {
    while ( i )
    {
      *(_QWORD *)i = *(_QWORD *)v25;
      *(_QWORD *)(i + 8) = *((_QWORD *)v25 + 1);
      *(_QWORD *)(i + 16) = *((_QWORD *)v25 + 2);
      *(_QWORD *)(i + 24) = *((_QWORD *)v25 + 3);
      *((_QWORD *)v25 + 3) = 0;
      *((_QWORD *)v25 + 1) = 0;
LABEL_21:
      v25 += 32;
      i += 32;
      if ( v25 == a2 )
        goto LABEL_25;
    }
    v27 = *((_QWORD *)v25 + 1);
    if ( !v27 )
      goto LABEL_21;
    j_j___libc_free_0(v27);
    v25 += 32;
  }
LABEL_25:
  v28 = i + 32;
  if ( a2 != v6 )
  {
    v29 = i + 32;
    do
    {
      v30 = *(_QWORD *)v11;
      v11 += 32;
      v29 += 32;
      *(_QWORD *)(v29 - 32) = v30;
      *(_QWORD *)(v29 - 24) = *((_QWORD *)v11 - 3);
      *(_QWORD *)(v29 - 16) = *((_QWORD *)v11 - 2);
      *(_QWORD *)(v29 - 8) = *((_QWORD *)v11 - 1);
    }
    while ( v11 != v6 );
    v28 += v6 - a2;
  }
  if ( v38 )
  {
    v35 = v28;
    j_j___libc_free_0(v38);
    v28 = v35;
  }
  *a1 = v39;
  a1[1] = v28;
  a1[2] = v39 + 32 * v36;
  return a1;
}
