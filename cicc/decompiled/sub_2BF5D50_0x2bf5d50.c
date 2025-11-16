// Function: sub_2BF5D50
// Address: 0x2bf5d50
//
__int64 __fastcall sub_2BF5D50(__int64 *a1)
{
  __int64 v1; // rdx
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  _BYTE *v15; // rsi
  __int64 *v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // r9
  unsigned __int64 v19; // rcx
  __int64 v20; // r8
  unsigned __int64 v21; // rbx
  __int64 v22; // rax
  unsigned __int64 v23; // rdi
  __m128i *v24; // rdx
  const __m128i *v25; // rax
  __int64 v26; // r9
  unsigned __int64 v27; // rcx
  __int64 v28; // r8
  unsigned __int64 v29; // rbx
  __int64 v30; // rax
  unsigned __int64 v31; // rdi
  __m128i *v32; // rdx
  const __m128i *v33; // rax
  __int64 v34; // rax
  unsigned __int64 v35; // r13
  unsigned __int64 v36; // rdx
  __int64 v37; // rbx
  __int64 *v39; // rax
  __int64 *v40; // rdx
  __int64 v41; // r15
  __int64 *v42; // rax
  unsigned __int64 v43; // rax
  char v44; // si
  char v45; // dl
  unsigned __int64 v46[16]; // [rsp+20h] [rbp-320h] BYREF
  __m128i v47; // [rsp+A0h] [rbp-2A0h] BYREF
  __int64 v48; // [rsp+B0h] [rbp-290h]
  int v49; // [rsp+B8h] [rbp-288h]
  char v50; // [rsp+BCh] [rbp-284h]
  _QWORD v51[8]; // [rsp+C0h] [rbp-280h] BYREF
  unsigned __int64 v52; // [rsp+100h] [rbp-240h] BYREF
  unsigned __int64 v53; // [rsp+108h] [rbp-238h]
  unsigned __int64 v54; // [rsp+110h] [rbp-230h]
  __int64 v55; // [rsp+120h] [rbp-220h] BYREF
  __int64 *v56; // [rsp+128h] [rbp-218h]
  unsigned int v57; // [rsp+130h] [rbp-210h]
  unsigned int v58; // [rsp+134h] [rbp-20Ch]
  char v59; // [rsp+13Ch] [rbp-204h]
  _BYTE v60[64]; // [rsp+140h] [rbp-200h] BYREF
  unsigned __int64 v61; // [rsp+180h] [rbp-1C0h] BYREF
  unsigned __int64 v62; // [rsp+188h] [rbp-1B8h]
  unsigned __int64 v63; // [rsp+190h] [rbp-1B0h]
  char v64[8]; // [rsp+1A0h] [rbp-1A0h] BYREF
  unsigned __int64 v65; // [rsp+1A8h] [rbp-198h]
  char v66; // [rsp+1BCh] [rbp-184h]
  _BYTE v67[64]; // [rsp+1C0h] [rbp-180h] BYREF
  unsigned __int64 v68; // [rsp+200h] [rbp-140h]
  unsigned __int64 v69; // [rsp+208h] [rbp-138h]
  unsigned __int64 v70; // [rsp+210h] [rbp-130h]
  __m128i v71; // [rsp+220h] [rbp-120h] BYREF
  char v72; // [rsp+230h] [rbp-110h]
  char v73; // [rsp+23Ch] [rbp-104h]
  char v74[64]; // [rsp+240h] [rbp-100h] BYREF
  unsigned __int64 v75; // [rsp+280h] [rbp-C0h]
  unsigned __int64 v76; // [rsp+288h] [rbp-B8h]
  unsigned __int64 v77; // [rsp+290h] [rbp-B0h]
  char v78[8]; // [rsp+298h] [rbp-A8h] BYREF
  unsigned __int64 v79; // [rsp+2A0h] [rbp-A0h]
  char v80; // [rsp+2B4h] [rbp-8Ch]
  char v81[64]; // [rsp+2B8h] [rbp-88h] BYREF
  unsigned __int64 v82; // [rsp+2F8h] [rbp-48h]
  unsigned __int64 v83; // [rsp+300h] [rbp-40h]
  unsigned __int64 v84; // [rsp+308h] [rbp-38h]

  v1 = *a1;
  v47.m128i_i64[1] = (__int64)v51;
  memset(v46, 0, 0x78u);
  v46[1] = (unsigned __int64)&v46[4];
  v48 = 0x100000008LL;
  v51[0] = v1;
  v71.m128i_i64[0] = v1;
  LODWORD(v46[2]) = 8;
  BYTE4(v46[3]) = 1;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v49 = 0;
  v50 = 1;
  v47.m128i_i64[0] = 1;
  v72 = 0;
  sub_2BF5D10(&v52, &v71);
  sub_C8CF70((__int64)v64, v67, 8, (__int64)&v46[4], (__int64)v46);
  v2 = v46[12];
  memset(&v46[12], 0, 24);
  v68 = v2;
  v69 = v46[13];
  v70 = v46[14];
  sub_C8CF70((__int64)&v55, v60, 8, (__int64)v51, (__int64)&v47);
  v3 = v52;
  v52 = 0;
  v61 = v3;
  v4 = v53;
  v53 = 0;
  v62 = v4;
  v5 = v54;
  v54 = 0;
  v63 = v5;
  sub_C8CF70((__int64)&v71, v74, 8, (__int64)v60, (__int64)&v55);
  v6 = v61;
  v61 = 0;
  v75 = v6;
  v7 = v62;
  v62 = 0;
  v76 = v7;
  v8 = v63;
  v63 = 0;
  v77 = v8;
  sub_C8CF70((__int64)v78, v81, 8, (__int64)v67, (__int64)v64);
  v12 = v68;
  v68 = 0;
  v82 = v12;
  v13 = v69;
  v69 = 0;
  v83 = v13;
  v14 = v70;
  v70 = 0;
  v84 = v14;
  if ( v61 )
    j_j___libc_free_0(v61);
  if ( !v59 )
    _libc_free((unsigned __int64)v56);
  if ( v68 )
    j_j___libc_free_0(v68);
  if ( !v66 )
    _libc_free(v65);
  if ( v52 )
    j_j___libc_free_0(v52);
  if ( !v50 )
    _libc_free(v47.m128i_u64[1]);
  if ( v46[12] )
    j_j___libc_free_0(v46[12]);
  if ( !BYTE4(v46[3]) )
    _libc_free(v46[1]);
  v15 = v60;
  v16 = &v55;
  sub_C8CD80((__int64)&v55, (__int64)v60, (__int64)&v71, v9, v10, v11);
  v19 = v76;
  v20 = v75;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v21 = v76 - v75;
  if ( v76 == v75 )
  {
    v23 = 0;
  }
  else
  {
    if ( v21 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_84;
    v22 = sub_22077B0(v76 - v75);
    v19 = v76;
    v20 = v75;
    v23 = v22;
  }
  v61 = v23;
  v62 = v23;
  v63 = v23 + v21;
  if ( v19 != v20 )
  {
    v24 = (__m128i *)v23;
    v25 = (const __m128i *)v20;
    do
    {
      if ( v24 )
      {
        *v24 = _mm_loadu_si128(v25);
        v24[1].m128i_i64[0] = v25[1].m128i_i64[0];
      }
      v25 = (const __m128i *)((char *)v25 + 24);
      v24 = (__m128i *)((char *)v24 + 24);
    }
    while ( (const __m128i *)v19 != v25 );
    v19 = (v19 - 24 - v20) >> 3;
    v23 += 8 * v19 + 24;
  }
  v62 = v23;
  v15 = v67;
  v16 = (__int64 *)v64;
  sub_C8CD80((__int64)v64, (__int64)v67, (__int64)v78, v19, v20, v18);
  v27 = v83;
  v28 = v82;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v29 = v83 - v82;
  if ( v83 != v82 )
  {
    if ( v29 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v30 = sub_22077B0(v83 - v82);
      v27 = v83;
      v28 = v82;
      v31 = v30;
      goto LABEL_29;
    }
LABEL_84:
    sub_4261EA(v16, v15, v17);
  }
  v31 = 0;
LABEL_29:
  v68 = v31;
  v69 = v31;
  v70 = v31 + v29;
  if ( v28 == v27 )
  {
    v34 = v31;
  }
  else
  {
    v32 = (__m128i *)v31;
    v33 = (const __m128i *)v28;
    do
    {
      if ( v32 )
      {
        *v32 = _mm_loadu_si128(v33);
        v32[1].m128i_i64[0] = v33[1].m128i_i64[0];
      }
      v33 = (const __m128i *)((char *)v33 + 24);
      v32 = (__m128i *)((char *)v32 + 24);
    }
    while ( (const __m128i *)v27 != v33 );
    v27 = (v27 - 24 - v28) >> 3;
    v34 = v31 + 8 * v27 + 24;
  }
  v35 = v62;
  v36 = v61;
  v69 = v34;
  if ( v62 - v61 == v34 - v31 )
    goto LABEL_69;
  while ( 1 )
  {
    v37 = *(_QWORD *)(v35 - 24);
    if ( !*(_BYTE *)(v37 + 8) )
      break;
    while ( 1 )
    {
      if ( *(_BYTE *)(v35 - 8) )
        goto LABEL_59;
      v39 = *(__int64 **)(v37 + 80);
      *(_BYTE *)(v35 - 8) = 1;
      *(_QWORD *)(v35 - 16) = v39;
LABEL_60:
      if ( v39 != (__int64 *)(*(_QWORD *)(v37 + 80) + 8LL * *(unsigned int *)(v37 + 88)) )
        break;
      v62 -= 24LL;
      v36 = v61;
      v35 = v62;
      if ( v62 == v61 )
        goto LABEL_68;
      v37 = *(_QWORD *)(v62 - 24);
    }
    v40 = v39 + 1;
    *(_QWORD *)(v35 - 16) = v39 + 1;
    v41 = *v39;
    if ( !v59 )
    {
LABEL_76:
      sub_C8CC70((__int64)&v55, v41, (__int64)v40, v27, v28, v26);
      if ( v45 )
        goto LABEL_67;
      goto LABEL_59;
    }
    v42 = v56;
    v40 = &v56[v58];
    if ( v56 != v40 )
    {
      while ( v41 != *v42 )
      {
        if ( v40 == ++v42 )
          goto LABEL_65;
      }
LABEL_59:
      v39 = *(__int64 **)(v35 - 16);
      goto LABEL_60;
    }
LABEL_65:
    if ( v58 >= v57 )
      goto LABEL_76;
    ++v58;
    *v40 = v41;
    ++v55;
LABEL_67:
    v47.m128i_i64[0] = v41;
    LOBYTE(v48) = 0;
    sub_2BF5D10(&v61, &v47);
    v36 = v61;
    v35 = v62;
LABEL_68:
    v31 = v68;
    if ( v35 - v36 == v69 - v68 )
    {
LABEL_69:
      if ( v36 == v35 )
      {
LABEL_75:
        v37 = 0;
        sub_2BF0140((__int64)v64);
        sub_2BF0140((__int64)&v55);
        sub_2BF0140((__int64)v78);
        sub_2BF0140((__int64)&v71);
        return v37;
      }
      v43 = v31;
      while ( *(_QWORD *)v36 == *(_QWORD *)v43 )
      {
        v44 = *(_BYTE *)(v36 + 16);
        if ( v44 != *(_BYTE *)(v43 + 16) || v44 && *(_QWORD *)(v36 + 8) != *(_QWORD *)(v43 + 8) )
          break;
        v36 += 24LL;
        v43 += 24LL;
        if ( v36 == v35 )
          goto LABEL_75;
      }
    }
  }
  if ( *(_BYTE *)(v37 + 128) )
    v37 = 0;
  if ( v31 )
    j_j___libc_free_0(v31);
  if ( !v66 )
    _libc_free(v65);
  if ( v61 )
    j_j___libc_free_0(v61);
  if ( !v59 )
    _libc_free((unsigned __int64)v56);
  if ( v82 )
    j_j___libc_free_0(v82);
  if ( !v80 )
    _libc_free(v79);
  if ( v75 )
    j_j___libc_free_0(v75);
  if ( !v73 )
    _libc_free(v71.m128i_u64[1]);
  return v37;
}
