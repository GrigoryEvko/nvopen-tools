// Function: sub_315A230
// Address: 0x315a230
//
void __fastcall sub_315A230(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i *v7; // rdi
  __m128i *v8; // rsi
  __int64 v9; // r8
  __int64 v10; // r9
  __m128i *v11; // rcx
  __m128i *v12; // rdx
  unsigned __int64 v13; // rbx
  __m128i *v14; // rax
  __m128i *v15; // rcx
  __m128i *v16; // rcx
  unsigned __int64 v17; // rbx
  __m128i *v18; // rax
  __int64 v19; // rcx
  __int64 v20; // r9
  __m128i *v21; // rcx
  __int64 v22; // rax
  unsigned __int64 v23; // r13
  const __m128i *v24; // rax
  unsigned __int64 v25; // rbx
  unsigned __int64 v26; // rcx
  __int64 v27; // r8
  unsigned __int64 v28; // r14
  __int64 v29; // rax
  unsigned __int64 v30; // rdi
  const __m128i *v31; // rax
  __int64 v32; // rsi
  unsigned __int64 v33; // rbx
  __int64 *v34; // rsi
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  unsigned __int64 v39; // rax
  __int64 v40; // [rsp+0h] [rbp-170h] BYREF
  __m128i *v41; // [rsp+8h] [rbp-168h]
  __int64 v42; // [rsp+10h] [rbp-160h]
  __int8 *v43; // [rsp+18h] [rbp-158h]
  __int64 v44; // [rsp+20h] [rbp-150h] BYREF
  __m128i *v45; // [rsp+28h] [rbp-148h]
  __m128i *v46; // [rsp+30h] [rbp-140h]
  __int8 *v47; // [rsp+38h] [rbp-138h]
  __int64 v48; // [rsp+40h] [rbp-130h] BYREF
  unsigned __int64 v49; // [rsp+48h] [rbp-128h]
  __int64 v50; // [rsp+50h] [rbp-120h]
  unsigned __int64 v51; // [rsp+58h] [rbp-118h]
  __m128i v52; // [rsp+60h] [rbp-110h] BYREF
  __m128i *v53; // [rsp+70h] [rbp-100h]
  __int64 v54; // [rsp+80h] [rbp-F0h]
  __m128i *v55; // [rsp+88h] [rbp-E8h]
  __m128i *v56; // [rsp+90h] [rbp-E0h]
  __m128i v57; // [rsp+A0h] [rbp-D0h] BYREF
  unsigned __int64 v58; // [rsp+B0h] [rbp-C0h]
  __m128i *v59; // [rsp+C8h] [rbp-A8h]
  __m128i *v60; // [rsp+D0h] [rbp-A0h]
  __int64 v61; // [rsp+E0h] [rbp-90h] BYREF
  __int64 *v62; // [rsp+E8h] [rbp-88h]
  __int64 v63; // [rsp+F0h] [rbp-80h]
  int v64; // [rsp+F8h] [rbp-78h]
  char v65; // [rsp+FCh] [rbp-74h]
  __int64 v66; // [rsp+100h] [rbp-70h] BYREF

  v62 = &v66;
  v63 = 0x100000008LL;
  v64 = 0;
  v65 = 1;
  v66 = a3;
  v61 = 1;
  if ( (_BYTE)a4 )
  {
    v57.m128i_i64[0] = a2;
    v7 = &v52;
    v8 = &v57;
    sub_3159360(v52.m128i_i64, &v57, (__int64)&v61, a4, (__int64)&v61, a6);
    v11 = v56;
    v45 = 0;
    v12 = v55;
    v46 = 0;
    v44 = v54;
    v47 = 0;
    v13 = (char *)v56 - (char *)v55;
    if ( v56 == v55 )
    {
      v14 = 0;
    }
    else
    {
      if ( v13 > 0x7FFFFFFFFFFFFFE0LL )
        goto LABEL_74;
      v7 = (__m128i *)((char *)v56 - (char *)v55);
      v14 = (__m128i *)sub_22077B0((char *)v56 - (char *)v55);
      v11 = v56;
      v12 = v55;
    }
    v45 = v14;
    v46 = v14;
    v47 = &v14->m128i_i8[v13];
    if ( v12 == v11 )
    {
      v15 = v14;
    }
    else
    {
      v15 = (__m128i *)((char *)v14 + (char *)v11 - (char *)v12);
      do
      {
        if ( v14 )
        {
          *v14 = _mm_loadu_si128(v12);
          v14[1] = _mm_loadu_si128(v12 + 1);
        }
        v14 += 2;
        v12 += 2;
      }
      while ( v15 != v14 );
    }
    v46 = v15;
    v16 = v53;
    v12 = (__m128i *)v52.m128i_i64[1];
    v41 = 0;
    v42 = 0;
    v40 = v52.m128i_i64[0];
    v43 = 0;
    v17 = (unsigned __int64)v53 - v52.m128i_i64[1];
    if ( v53 == (__m128i *)v52.m128i_i64[1] )
    {
      v18 = 0;
      goto LABEL_13;
    }
    if ( v17 <= 0x7FFFFFFFFFFFFFE0LL )
    {
      v18 = (__m128i *)sub_22077B0((unsigned __int64)v53 - v52.m128i_i64[1]);
      v16 = v53;
      v12 = (__m128i *)v52.m128i_i64[1];
LABEL_13:
      v41 = v18;
      v42 = (__int64)v18;
      v43 = &v18->m128i_i8[v17];
      if ( v16 == v12 )
      {
        v19 = (__int64)v18;
      }
      else
      {
        v19 = (__int64)v18->m128i_i64 + (char *)v16 - (char *)v12;
        do
        {
          if ( v18 )
          {
            *v18 = _mm_loadu_si128(v12);
            v18[1] = _mm_loadu_si128(v12 + 1);
          }
          v18 += 2;
          v12 += 2;
        }
        while ( (__m128i *)v19 != v18 );
      }
      v42 = v19;
      sub_315A010(a5, &v40, (__int64)&v44, v19, v9, v10);
      if ( v41 )
        j_j___libc_free_0((unsigned __int64)v41);
      if ( v45 )
        j_j___libc_free_0((unsigned __int64)v45);
      if ( v55 )
        j_j___libc_free_0((unsigned __int64)v55);
      if ( v52.m128i_i64[1] )
        j_j___libc_free_0(v52.m128i_u64[1]);
      goto LABEL_26;
    }
LABEL_74:
    sub_4261EA(v7, v8, v12);
  }
  v52.m128i_i64[0] = a2;
  v8 = &v52;
  sub_26D9210(v57.m128i_i64, &v52, (__int64)&v61, a4, (__int64)&v61, a6);
  v21 = v60;
  v7 = v59;
  if ( v60 == v59 )
  {
    v23 = 0;
  }
  else
  {
    if ( (unsigned __int64)((char *)v60 - (char *)v59) > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_74;
    v22 = sub_22077B0((char *)v60 - (char *)v59);
    v21 = v60;
    v7 = v59;
    v23 = v22;
  }
  if ( v7 == v21 )
  {
    v25 = v23;
  }
  else
  {
    v12 = (__m128i *)v23;
    v24 = v7;
    do
    {
      if ( v12 )
      {
        *v12 = _mm_loadu_si128(v24);
        v8 = (__m128i *)v24[1].m128i_i64[0];
        v12[1].m128i_i64[0] = (__int64)v8;
      }
      v24 = (const __m128i *)((char *)v24 + 24);
      v12 = (__m128i *)((char *)v12 + 24);
    }
    while ( v24 != v21 );
    v25 = v23 + 8 * ((unsigned __int64)((char *)&v24[-2].m128i_u64[1] - (char *)v7) >> 3) + 24;
  }
  v26 = v58;
  v49 = 0;
  v27 = v57.m128i_i64[1];
  v50 = 0;
  v48 = v57.m128i_i64[0];
  v51 = 0;
  v28 = v58 - v57.m128i_i64[1];
  if ( v58 == v57.m128i_i64[1] )
  {
    v30 = 0;
  }
  else
  {
    if ( v28 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_74;
    v29 = sub_22077B0(v58 - v57.m128i_i64[1]);
    v26 = v58;
    v27 = v57.m128i_i64[1];
    v30 = v29;
  }
  v49 = v30;
  v50 = v30;
  v51 = v30 + v28;
  if ( v26 == v27 )
  {
    v32 = v30;
  }
  else
  {
    v12 = (__m128i *)v30;
    v31 = (const __m128i *)v27;
    do
    {
      if ( v12 )
      {
        *v12 = _mm_loadu_si128(v31);
        v12[1].m128i_i64[0] = v31[1].m128i_i64[0];
      }
      v31 = (const __m128i *)((char *)v31 + 24);
      v12 = (__m128i *)((char *)v12 + 24);
    }
    while ( (const __m128i *)v26 != v31 );
    v26 = (v26 - 24 - v27) >> 3;
    v32 = v30 + 8 * v26 + 24;
  }
  v50 = v32;
  v33 = v25 - v23;
  while ( 1 )
  {
    if ( v33 != v32 - v30 )
      goto LABEL_47;
    if ( v30 == v32 )
      break;
    v12 = (__m128i *)v23;
    v39 = v30;
    while ( 1 )
    {
      v26 = v12->m128i_i64[0];
      if ( *(_QWORD *)v39 != v12->m128i_i64[0] )
        break;
      v26 = *(unsigned __int8 *)(v39 + 16);
      if ( (_BYTE)v26 != v12[1].m128i_i8[0] )
        break;
      if ( (_BYTE)v26 )
      {
        v26 = v12->m128i_u64[1];
        if ( *(_QWORD *)(v39 + 8) != v26 )
          break;
      }
      v39 += 24LL;
      v12 = (__m128i *)((char *)v12 + 24);
      if ( v39 == v32 )
        goto LABEL_55;
    }
LABEL_47:
    v34 = (__int64 *)(v32 - 24);
    sub_31599A0(a5, v34, (__int64)v12, v26, v27, v20);
    sub_3158930((unsigned __int64 *)&v48, (__int64)v34, v35, v36, v37, v38);
    v30 = v49;
    v32 = v50;
  }
LABEL_55:
  if ( v30 )
    j_j___libc_free_0(v30);
  if ( v23 )
    j_j___libc_free_0(v23);
  if ( v59 )
    j_j___libc_free_0((unsigned __int64)v59);
  if ( !v57.m128i_i64[1] )
  {
LABEL_26:
    if ( !v65 )
      goto LABEL_63;
    return;
  }
  j_j___libc_free_0(v57.m128i_u64[1]);
  if ( !v65 )
LABEL_63:
    _libc_free((unsigned __int64)v62);
}
