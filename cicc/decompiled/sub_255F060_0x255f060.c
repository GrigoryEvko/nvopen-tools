// Function: sub_255F060
// Address: 0x255f060
//
__m128i *__fastcall sub_255F060(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const __m128i *v7; // r15
  const __m128i *v8; // r14
  char **v9; // r9
  __int64 v10; // rsi
  const __m128i *v11; // rcx
  unsigned __int64 v12; // r8
  const __m128i *v13; // rdx
  __m128i *v14; // rcx
  const __m128i *v15; // rax
  char **v16; // r14
  char **v17; // rax
  unsigned __int64 v18; // rsi
  size_t v19; // rdx
  char **v20; // r13
  size_t v21; // rdx
  char *v22; // rsi
  __int64 v23; // r8
  __int64 v24; // r9
  __m128i *v25; // rax
  const __m128i *v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rsi
  const __m128i *v30; // rdx
  _QWORD *v31; // r12
  __int64 v32; // rsi
  char *v33; // rax
  unsigned __int64 v34; // rsi
  size_t v35; // rdx
  char *v36; // rsi
  char *v37; // r12
  size_t v38; // rdx
  char *v39; // rsi
  int v40; // [rsp+8h] [rbp-108h]
  char **v41; // [rsp+8h] [rbp-108h]
  char *v42; // [rsp+8h] [rbp-108h]
  void *base; // [rsp+10h] [rbp-100h] BYREF
  __int64 v44; // [rsp+18h] [rbp-F8h]
  char *v45; // [rsp+20h] [rbp-F0h] BYREF
  size_t v46; // [rsp+28h] [rbp-E8h]
  char v47; // [rsp+30h] [rbp-E0h] BYREF
  char *v48[4]; // [rsp+40h] [rbp-D0h] BYREF
  char *v49; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v50; // [rsp+68h] [rbp-A8h]
  char v51; // [rsp+70h] [rbp-A0h] BYREF
  __m128i v52[2]; // [rsp+80h] [rbp-90h] BYREF
  __m128i v53; // [rsp+A0h] [rbp-70h] BYREF
  const __m128i *v54; // [rsp+B0h] [rbp-60h]
  const __m128i *v55; // [rsp+B8h] [rbp-58h]
  __m128i *v56; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v57; // [rsp+C8h] [rbp-48h]
  __m128i v58; // [rsp+D0h] [rbp-40h] BYREF

  v7 = *(const __m128i **)(a2 + 112);
  v8 = &v7[*(unsigned int *)(a2 + 128)];
  if ( *(_DWORD *)(a2 + 120) && v7 != v8 )
  {
    while ( v7->m128i_i64[0] == -1 || v7->m128i_i64[0] == -2 )
    {
      if ( v8 == ++v7 )
        goto LABEL_2;
    }
    v44 = 0;
    base = &v45;
    if ( v8 != v7 )
    {
      v11 = v7;
      v12 = 0;
      while ( 1 )
      {
        v13 = v11 + 1;
        if ( v8 == &v11[1] )
          break;
        while ( 1 )
        {
          v11 = v13;
          if ( v13->m128i_i64[0] < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v8 == ++v13 )
            goto LABEL_14;
        }
        ++v12;
        if ( v8 == v13 )
          goto LABEL_15;
      }
LABEL_14:
      ++v12;
LABEL_15:
      v40 = v12;
      sub_C8D5F0((__int64)&base, &v45, v12, 0x10u, v12, a6);
      v14 = (__m128i *)((char *)base + 16 * (unsigned int)v44);
      do
      {
        if ( v14 )
          *v14 = _mm_loadu_si128(v7);
        v15 = v7 + 1;
        if ( v8 == &v7[1] )
          break;
        while ( 1 )
        {
          v7 = v15;
          if ( v15->m128i_i64[0] < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v8 == ++v15 )
            goto LABEL_21;
        }
        ++v14;
      }
      while ( v8 != v15 );
LABEL_21:
      v9 = (char **)base;
      LODWORD(v44) = v40 + v44;
      v10 = 2LL * (unsigned int)v44;
      if ( (unsigned int)v44 > 1uLL )
      {
        qsort(base, (v10 * 8) >> 4, 0x10u, (__compar_fn_t)sub_A16990);
        v9 = (char **)base;
        v10 = 2LL * (unsigned int)v44;
      }
      goto LABEL_23;
    }
  }
  else
  {
LABEL_2:
    HIDWORD(v44) = 0;
    base = &v45;
  }
  LODWORD(v44) = 0;
  v9 = &v45;
  v10 = 0;
LABEL_23:
  v16 = &v9[v10];
  v46 = 0;
  v45 = &v47;
  v47 = 0;
  if ( &v9[v10] != v9 )
  {
    v17 = v9;
    v18 = ((v10 * 8) >> 4) - 1;
    do
    {
      v18 += (unsigned __int64)v17[1];
      v17 += 2;
    }
    while ( v16 != v17 );
    v41 = v9;
    sub_2240E30((__int64)&v45, v18);
    v19 = (size_t)v41[1];
    if ( v19 > 0x3FFFFFFFFFFFFFFFLL - v46 )
      goto LABEL_60;
    sub_2241490((unsigned __int64 *)&v45, *v41, v19);
    v20 = v41 + 2;
    if ( v16 != v41 + 2 )
    {
      while ( v46 != 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490((unsigned __int64 *)&v45, ",", 1u);
        v21 = (size_t)v20[1];
        v22 = *v20;
        if ( v21 > 0x3FFFFFFFFFFFFFFFLL - v46 )
          break;
        v20 += 2;
        sub_2241490((unsigned __int64 *)&v45, v22, v21);
        if ( v16 == v20 )
          goto LABEL_31;
      }
LABEL_60:
      sub_4262D8((__int64)"basic_string::append");
    }
  }
LABEL_31:
  sub_253C590((__int64 *)v48, "Universal");
  if ( !*(_BYTE *)(a2 + 136) )
  {
    v27 = *(const __m128i **)(a2 + 152);
    v28 = *(unsigned int *)(a2 + 160);
    v29 = *(_QWORD *)(a2 + 144);
    v30 = &v27[*(unsigned int *)(a2 + 168)];
    if ( (_DWORD)v28 )
    {
      for ( ; v30 != v27; ++v27 )
      {
        v28 = v27->m128i_i64[0];
        if ( v27->m128i_i64[0] != -1 && v28 != -2 )
          break;
      }
    }
    else
    {
      v27 += *(unsigned int *)(a2 + 168);
    }
    v53.m128i_i64[0] = a2 + 144;
    v53.m128i_i64[1] = v29;
    v54 = v27;
    v56 = (__m128i *)(a2 + 144);
    v57 = v29;
    v55 = v30;
    v58.m128i_i64[0] = (__int64)v30;
    v58.m128i_i64[1] = (__int64)v30;
    LODWORD(v44) = 0;
    sub_255EF70((__int64)&base, v29, (__int64)v30, v28, v23, v24, a2 + 144, v29, v27, v30, a2 + 144, v29, v30);
    v31 = base;
    v57 = 0;
    v58.m128i_i8[0] = 0;
    v32 = 16LL * (unsigned int)v44;
    v42 = (char *)base + v32;
    v56 = &v58;
    if ( base != (char *)base + v32 )
    {
      v33 = (char *)base;
      v34 = (v32 >> 4) - 1;
      do
      {
        v34 += *((_QWORD *)v33 + 1);
        v33 += 16;
      }
      while ( v42 != v33 );
      sub_2240E30((__int64)&v56, v34);
      v35 = v31[1];
      v36 = (char *)*v31;
      if ( v35 > 0x3FFFFFFFFFFFFFFFLL - v57 )
        goto LABEL_60;
      v37 = (char *)(v31 + 2);
      sub_2241490((unsigned __int64 *)&v56, v36, v35);
      if ( v37 != v42 )
      {
        while ( v57 != 0x3FFFFFFFFFFFFFFFLL )
        {
          sub_2241490((unsigned __int64 *)&v56, ",", 1u);
          v38 = *((_QWORD *)v37 + 1);
          v39 = *(char **)v37;
          if ( v38 > 0x3FFFFFFFFFFFFFFFLL - v57 )
            break;
          v37 += 16;
          sub_2241490((unsigned __int64 *)&v56, v39, v38);
          if ( v42 == v37 )
            goto LABEL_52;
        }
        goto LABEL_60;
      }
    }
LABEL_52:
    sub_2240D70((__int64)v48, &v56);
    sub_2240A30((unsigned __int64 *)&v56);
  }
  v50 = 0;
  v49 = &v51;
  v51 = 0;
  sub_2240E30((__int64)&v49, v46 + 7);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v50) <= 6 )
    goto LABEL_60;
  sub_2241490((unsigned __int64 *)&v49, "Known [", 7u);
  sub_2241490((unsigned __int64 *)&v49, v45, v46);
  sub_94F930(v52, (__int64)&v49, "],");
  sub_94F930(&v53, (__int64)v52, " Assumed [");
  v25 = (__m128i *)sub_2241490((unsigned __int64 *)&v53, v48[0], (size_t)v48[1]);
  v56 = &v58;
  if ( (__m128i *)v25->m128i_i64[0] == &v25[1] )
  {
    v58 = _mm_loadu_si128(v25 + 1);
  }
  else
  {
    v56 = (__m128i *)v25->m128i_i64[0];
    v58.m128i_i64[0] = v25[1].m128i_i64[0];
  }
  v57 = v25->m128i_i64[1];
  v25->m128i_i64[0] = (__int64)v25[1].m128i_i64;
  v25->m128i_i64[1] = 0;
  v25[1].m128i_i8[0] = 0;
  sub_94F930(a1, (__int64)&v56, "]");
  sub_2240A30((unsigned __int64 *)&v56);
  sub_2240A30((unsigned __int64 *)&v53);
  sub_2240A30((unsigned __int64 *)v52);
  sub_2240A30((unsigned __int64 *)&v49);
  sub_2240A30((unsigned __int64 *)v48);
  sub_2240A30((unsigned __int64 *)&v45);
  if ( base != &v45 )
    _libc_free((unsigned __int64)base);
  return a1;
}
