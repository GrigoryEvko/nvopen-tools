// Function: sub_142E3D0
// Address: 0x142e3d0
//
__int8 *__fastcall sub_142E3D0(__int64 a1, const __m128i *a2, const __m128i *a3)
{
  __int64 v4; // rsi
  const __m128i *v5; // r12
  __int64 v6; // rcx
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rbx
  _QWORD *v9; // r13
  bool v10; // cf
  unsigned __int64 v11; // rbx
  signed __int64 v12; // r8
  __m128i *v13; // rbx
  _BYTE *v14; // rax
  __m128i v15; // xmm2
  _BYTE *v16; // rsi
  unsigned __int64 v17; // r9
  __int64 v18; // rax
  char *v19; // rdi
  size_t v20; // r15
  const __m128i *v21; // r15
  __int64 i; // rbx
  __int64 v23; // rdi
  char *v24; // r8
  const __m128i *v25; // rax
  char *v26; // rdx
  __int64 v27; // rsi
  __m128i v28; // xmm0
  __int64 v29; // rsi
  const __m128i *v30; // rdi
  __int8 *result; // rax
  __int64 v32; // rax
  const __m128i *v33; // [rsp+8h] [rbp-58h]
  const __m128i *v34; // [rsp+8h] [rbp-58h]
  unsigned __int64 v35; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+18h] [rbp-48h]
  const __m128i *v37; // [rsp+20h] [rbp-40h]
  char *v38; // [rsp+20h] [rbp-40h]
  __m128i *v39; // [rsp+28h] [rbp-38h]

  v4 = 0x333333333333333LL;
  v5 = *(const __m128i **)(a1 + 8);
  v37 = *(const __m128i **)a1;
  v6 = (__int64)v5->m128i_i64 - *(_QWORD *)a1;
  v7 = 0xCCCCCCCCCCCCCCCDLL * (v6 >> 3);
  if ( v7 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  v9 = (_QWORD *)a1;
  if ( v7 )
    v8 = 0xCCCCCCCCCCCCCCCDLL * (v6 >> 3);
  v10 = __CFADD__(v7, v8);
  v11 = v7 + v8;
  v36 = v11;
  v12 = (char *)a2 - (char *)v37;
  if ( v10 )
  {
    a1 = 0x7FFFFFFFFFFFFFF8LL;
    v36 = 0x333333333333333LL;
  }
  else
  {
    if ( !v11 )
    {
      v39 = 0;
      goto LABEL_7;
    }
    if ( v11 <= 0x333333333333333LL )
      v4 = v11;
    v36 = v4;
    a1 = 40 * v4;
  }
  v34 = a3;
  v32 = sub_22077B0(a1);
  v12 = (char *)a2 - (char *)v37;
  a3 = v34;
  v39 = (__m128i *)v32;
LABEL_7:
  v13 = (__m128i *)((char *)v39 + v12);
  if ( &v39->m128i_i8[v12] )
  {
    v14 = (_BYTE *)a3[1].m128i_i64[1];
    v15 = _mm_loadu_si128(a3);
    v13[1].m128i_i64[0] = 0;
    v16 = (_BYTE *)a3[1].m128i_i64[0];
    v13[1].m128i_i64[1] = 0;
    v13[2].m128i_i64[0] = 0;
    *v13 = v15;
    v17 = v14 - v16;
    if ( v14 == v16 )
    {
      v20 = 0;
      v19 = 0;
    }
    else
    {
      if ( v17 > 0x7FFFFFFFFFFFFFF8LL )
        sub_4261EA(a1, v16, a3);
      v33 = a3;
      v35 = v14 - v16;
      v18 = sub_22077B0(v17);
      v17 = v35;
      v19 = (char *)v18;
      v14 = (_BYTE *)v33[1].m128i_i64[1];
      v16 = (_BYTE *)v33[1].m128i_i64[0];
      v20 = v14 - v16;
    }
    v13[1].m128i_i64[0] = (__int64)v19;
    v13[1].m128i_i64[1] = (__int64)v19;
    v13[2].m128i_i64[0] = (__int64)&v19[v17];
    if ( v14 != v16 )
      v19 = (char *)memmove(v19, v16, v20);
    v13[1].m128i_i64[1] = (__int64)&v19[v20];
  }
  v21 = v37;
  for ( i = (__int64)v39; v21 != a2; i = 40 )
  {
    while ( i )
    {
      *(__m128i *)i = _mm_loadu_si128(v21);
      *(_QWORD *)(i + 16) = v21[1].m128i_i64[0];
      *(_QWORD *)(i + 24) = v21[1].m128i_i64[1];
      *(_QWORD *)(i + 32) = v21[2].m128i_i64[0];
      v21[2].m128i_i64[0] = 0;
      v21[1].m128i_i64[0] = 0;
LABEL_17:
      v21 = (const __m128i *)((char *)v21 + 40);
      i += 40;
      if ( v21 == a2 )
        goto LABEL_21;
    }
    v23 = v21[1].m128i_i64[0];
    if ( !v23 )
      goto LABEL_17;
    j_j___libc_free_0(v23, v21[2].m128i_i64[0] - v23);
    v21 = (const __m128i *)((char *)v21 + 40);
  }
LABEL_21:
  v24 = (char *)(i + 40);
  if ( a2 != v5 )
  {
    v25 = a2;
    v26 = (char *)(i + 40);
    do
    {
      v27 = v25[1].m128i_i64[0];
      v28 = _mm_loadu_si128(v25);
      v25 = (const __m128i *)((char *)v25 + 40);
      v26 += 40;
      *((_QWORD *)v26 - 3) = v27;
      v29 = v25[-1].m128i_i64[0];
      *(__m128i *)(v26 - 40) = v28;
      *((_QWORD *)v26 - 2) = v29;
      *((_QWORD *)v26 - 1) = v25[-1].m128i_i64[1];
    }
    while ( v25 != v5 );
    v24 += 8 * ((unsigned __int64)((char *)v25 - (char *)a2 - 40) >> 3) + 40;
  }
  v30 = v37;
  if ( v37 )
  {
    v38 = v24;
    j_j___libc_free_0(v30, v9[2] - (_QWORD)v30);
    v24 = v38;
  }
  v9[1] = v24;
  *v9 = v39;
  result = &v39->m128i_i8[40 * v36];
  v9[2] = result;
  return result;
}
