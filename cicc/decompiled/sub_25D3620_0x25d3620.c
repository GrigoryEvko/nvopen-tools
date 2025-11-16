// Function: sub_25D3620
// Address: 0x25d3620
//
__int64 __fastcall sub_25D3620(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _DWORD *v6; // rbx
  __int64 v7; // rax
  int v8; // edx
  __int64 v9; // r14
  __int64 v10; // r12
  _QWORD *v11; // rax
  unsigned int v12; // ebx
  const void *v13; // rsi
  int v15; // r13d
  __int64 v16; // r15
  __int64 v17; // rax
  _QWORD *v18; // rdi
  __int64 v19; // rsi
  __int64 v20; // rax
  int v21; // r14d
  int v22; // eax
  int v23; // eax
  int v24; // r13d
  char *v25; // rdi
  size_t v26; // rdx
  int v27; // r10d
  unsigned int i; // ecx
  __int64 v29; // r14
  const void *v30; // rsi
  bool v31; // al
  unsigned int v32; // ecx
  int v33; // eax
  unsigned int v34; // ecx
  __int64 v35; // rdx
  _QWORD *v36; // rax
  _QWORD *v37; // rdx
  __int64 v38; // rax
  __m128i si128; // xmm0
  unsigned __int64 v40; // rdx
  void *v41; // rdi
  __int64 v42; // r13
  void **v43; // r15
  __int64 v44; // r13
  int v45; // eax
  __int64 v46; // rcx
  const void *v47; // rdi
  int v48; // r10d
  __int64 v49; // r15
  unsigned int v50; // r13d
  size_t v51; // r14
  __int64 v52; // rbx
  const void *v53; // r12
  int v54; // eax
  __int64 v55; // rcx
  int v56; // eax
  unsigned int v57; // r13d
  _QWORD *v58; // rdi
  __int64 v59; // rsi
  unsigned int v60; // eax
  int v61; // eax
  unsigned __int64 v62; // rax
  unsigned __int64 v63; // rax
  int v64; // ebx
  __int64 v65; // r12
  _QWORD *v66; // rax
  _QWORD *j; // rdx
  __int64 v68; // r14
  _QWORD *v69; // rsi
  size_t v70; // [rsp+8h] [rbp-D8h]
  int v71; // [rsp+8h] [rbp-D8h]
  unsigned int v72; // [rsp+10h] [rbp-D0h]
  __int64 v73; // [rsp+18h] [rbp-C8h]
  int v74; // [rsp+20h] [rbp-C0h]
  unsigned int v75; // [rsp+28h] [rbp-B8h]
  _DWORD *v76; // [rsp+28h] [rbp-B8h]
  __m128i v77; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v78; // [rsp+40h] [rbp-A0h]
  __int64 v79; // [rsp+48h] [rbp-98h]
  __int64 v80; // [rsp+58h] [rbp-88h] BYREF
  int v81; // [rsp+60h] [rbp-80h]
  __int64 v82; // [rsp+68h] [rbp-78h]
  size_t n[2]; // [rsp+70h] [rbp-70h] BYREF
  __int64 v84; // [rsp+80h] [rbp-60h] BYREF
  _QWORD *v85; // [rsp+88h] [rbp-58h]
  __int64 v86; // [rsp+90h] [rbp-50h]
  __int64 v87; // [rsp+98h] [rbp-48h]
  void *src; // [rsp+A0h] [rbp-40h] BYREF
  size_t nmemb; // [rsp+A8h] [rbp-38h]
  _BYTE v90[48]; // [rsp+B0h] [rbp-30h] BYREF

  src = v90;
  v6 = (_DWORD *)a2[2];
  v7 = *((unsigned int *)a2 + 8);
  v8 = *((_DWORD *)a2 + 6);
  v78 = a1;
  v84 = 0;
  v9 = *a2;
  v85 = 0;
  v10 = (__int64)&v6[v7];
  v86 = 0;
  v87 = 0;
  nmemb = 0;
  if ( !v8 || (_DWORD *)v10 == v6 )
    goto LABEL_2;
  while ( *v6 > 0xFFFFFFFD )
  {
    if ( (_DWORD *)v10 == ++v6 )
      goto LABEL_2;
  }
  if ( (_DWORD *)v10 == v6 )
  {
LABEL_2:
    v84 = 1;
    goto LABEL_3;
  }
  v79 = v9;
  v15 = 0;
  v16 = 0;
  while ( 2 )
  {
    v17 = *(_QWORD *)(v79 + 32) + 32LL * (*v6 >> 1);
    v18 = *(_QWORD **)v17;
    v19 = *(_QWORD *)(v17 + 8);
    v81 = *v6 & 1;
    v20 = *(_QWORD *)(v17 + 16);
    n[0] = (size_t)v18;
    v82 = v20;
    n[1] = v19;
    if ( !v15 )
    {
      ++v84;
      v80 = 0;
LABEL_15:
      sub_BA8070((__int64)&v84, 2 * v15);
      v21 = v87;
      if ( (_DWORD)v87 )
      {
        v44 = (__int64)v85;
        v45 = sub_C94890((_QWORD *)n[0], n[1]);
        a6 = (unsigned int)(v21 - 1);
        v77.m128i_i64[0] = v10;
        v46 = v44;
        v76 = v6;
        v47 = (const void *)n[0];
        v48 = 1;
        v49 = 0;
        v50 = a6 & v45;
        v51 = n[1];
        while ( 1 )
        {
          v52 = v46 + 16LL * v50;
          v53 = *(const void **)v52;
          if ( *(_QWORD *)v52 == -1 )
            break;
          if ( v53 == (const void *)-2LL )
          {
            if ( v47 == (const void *)-2LL )
              goto LABEL_66;
          }
          else if ( v51 == *(_QWORD *)(v52 + 8) )
          {
            v71 = v48;
            v72 = a6;
            v73 = v46;
            if ( !v51 || (v54 = memcmp(v47, v53, v51), v46 = v73, a6 = v72, v48 = v71, !v54) )
            {
LABEL_66:
              a5 = v52;
              v10 = v77.m128i_i64[0];
              v6 = v76;
LABEL_67:
              v80 = a5;
              goto LABEL_17;
            }
            if ( v53 == (const void *)-1LL )
            {
              a5 = v52;
              v10 = v77.m128i_i64[0];
              v68 = v49;
              v6 = v76;
              goto LABEL_85;
            }
          }
          if ( v49 || v53 != (const void *)-2LL )
            v52 = v49;
          v57 = v48 + v50;
          v49 = v52;
          ++v48;
          v50 = a6 & v57;
        }
        a5 = v46 + 16LL * v50;
        v10 = v77.m128i_i64[0];
        v68 = v49;
        v6 = v76;
        if ( v47 == (const void *)-1LL )
          goto LABEL_67;
LABEL_85:
        if ( !v68 )
          v68 = a5;
        v80 = v68;
        a5 = v68;
      }
      else
      {
        v80 = 0;
        a5 = 0;
      }
LABEL_17:
      v22 = v86 + 1;
      goto LABEL_50;
    }
    v23 = sub_C94890(v18, v19);
    v24 = v15 - 1;
    v25 = (char *)n[0];
    a5 = 0;
    v26 = n[1];
    v27 = 1;
    for ( i = v24 & v23; ; i = v24 & v32 )
    {
      v29 = v16 + 16LL * i;
      v30 = *(const void **)v29;
      v31 = v25 + 1 == 0;
      if ( *(_QWORD *)v29 != -1 )
      {
        v31 = v25 + 2 == 0;
        if ( v30 != (const void *)-2LL )
        {
          if ( *(_QWORD *)(v29 + 8) != v26 )
            goto LABEL_22;
          v74 = v27;
          v75 = i;
          v77.m128i_i64[0] = a5;
          if ( !v26 )
            goto LABEL_29;
          v70 = v26;
          v33 = memcmp(v25, v30, v26);
          v26 = v70;
          a5 = v77.m128i_i64[0];
          i = v75;
          v27 = v74;
          v31 = v33 == 0;
        }
      }
      if ( v31 )
        goto LABEL_29;
      if ( v30 == (const void *)-1LL )
        break;
LABEL_22:
      if ( v30 == (const void *)-2LL && !a5 )
        a5 = v29;
      v32 = v27 + i;
      ++v27;
    }
    v15 = v87;
    if ( !a5 )
      a5 = v29;
    ++v84;
    v22 = v86 + 1;
    v80 = a5;
    if ( 4 * ((int)v86 + 1) >= (unsigned int)(3 * v87) )
      goto LABEL_15;
    if ( (int)v87 - (v22 + HIDWORD(v86)) <= (unsigned int)v87 >> 3 )
    {
      sub_BA8070((__int64)&v84, v87);
      sub_B9B010((__int64)&v84, n, &v80);
      a5 = v80;
      v22 = v86 + 1;
    }
LABEL_50:
    LODWORD(v86) = v22;
    if ( *(_QWORD *)a5 != -1 )
      --HIDWORD(v86);
    *(__m128i *)a5 = _mm_load_si128((const __m128i *)n);
    v38 = (unsigned int)nmemb;
    si128 = _mm_load_si128((const __m128i *)n);
    v40 = (unsigned int)nmemb + 1LL;
    if ( v40 > HIDWORD(nmemb) )
    {
      v77 = si128;
      sub_C8D5F0((__int64)&src, v90, v40, 0x10u, a5, a6);
      v38 = (unsigned int)nmemb;
      si128 = _mm_load_si128(&v77);
    }
    *((__m128i *)src + v38) = si128;
    LODWORD(nmemb) = nmemb + 1;
LABEL_29:
    if ( ++v6 != (_DWORD *)v10 )
    {
      while ( *v6 > 0xFFFFFFFD )
      {
        if ( (_DWORD *)v10 == ++v6 )
          goto LABEL_32;
      }
      if ( (_DWORD *)v10 != v6 )
      {
        v16 = (__int64)v85;
        v15 = v87;
        continue;
      }
    }
    break;
  }
LABEL_32:
  ++v84;
  if ( (_DWORD)v86 )
  {
    v34 = 4 * v86;
    a5 = 64;
    v35 = (unsigned int)v87;
    if ( (unsigned int)(4 * v86) < 0x40 )
      v34 = 64;
    if ( v34 >= (unsigned int)v87 )
    {
LABEL_36:
      v36 = v85;
      v37 = &v85[2 * v35];
      if ( v85 != v37 )
      {
        do
        {
          *v36 = -1;
          v36 += 2;
          *(v36 - 1) = 0;
        }
        while ( v37 != v36 );
      }
      v86 = 0;
      goto LABEL_3;
    }
    v58 = v85;
    v59 = 2LL * (unsigned int)v87;
    if ( (_DWORD)v86 == 1 )
    {
      v65 = 2048;
      v64 = 128;
      goto LABEL_79;
    }
    _BitScanReverse(&v60, v86 - 1);
    v61 = 1 << (33 - (v60 ^ 0x1F));
    if ( v61 < 64 )
      v61 = 64;
    if ( v61 == (_DWORD)v87 )
    {
      v86 = 0;
      v69 = &v85[v59];
      do
      {
        if ( v58 )
        {
          *v58 = -1;
          v58[1] = 0;
        }
        v58 += 2;
      }
      while ( v69 != v58 );
    }
    else
    {
      v62 = (4 * v61 / 3u + 1) | ((unsigned __int64)(4 * v61 / 3u + 1) >> 1);
      v63 = ((v62 | (v62 >> 2)) >> 4) | v62 | (v62 >> 2) | ((((v62 | (v62 >> 2)) >> 4) | v62 | (v62 >> 2)) >> 8);
      v64 = (v63 | (v63 >> 16)) + 1;
      v65 = 16 * ((v63 | (v63 >> 16)) + 1);
LABEL_79:
      sub_C7D6A0((__int64)v85, v59 * 8, 8);
      LODWORD(v87) = v64;
      v66 = (_QWORD *)sub_C7D670(v65, 8);
      v86 = 0;
      v85 = v66;
      for ( j = &v66[2 * (unsigned int)v87]; j != v66; v66 += 2 )
      {
        if ( v66 )
        {
          *v66 = -1;
          v66[1] = 0;
        }
      }
    }
  }
  else if ( HIDWORD(v86) )
  {
    v35 = (unsigned int)v87;
    if ( (unsigned int)v87 <= 0x40 )
      goto LABEL_36;
    sub_C7D6A0((__int64)v85, 16LL * (unsigned int)v87, 8);
    v85 = 0;
    v86 = 0;
    LODWORD(v87) = 0;
  }
LABEL_3:
  v11 = (_QWORD *)v78;
  v12 = nmemb;
  v13 = (const void *)(v78 + 16);
  *(_QWORD *)(v78 + 8) = 0;
  *v11 = v13;
  if ( v12 )
  {
    v41 = src;
    v42 = 16LL * v12;
    if ( src == v90 )
    {
      v43 = (void **)v78;
      sub_C8D5F0(v78, v13, v12, 0x10u, a5, a6);
      v41 = *v43;
      if ( 16LL * (unsigned int)nmemb )
      {
        memcpy(v41, src, 16LL * (unsigned int)nmemb);
        v41 = *v43;
      }
      LODWORD(nmemb) = 0;
      *(_DWORD *)(v78 + 8) = v12;
      if ( v42 == 16 )
        goto LABEL_4;
    }
    else
    {
      v55 = v78;
      v56 = HIDWORD(nmemb);
      nmemb = 0;
      *(_DWORD *)(v78 + 12) = v56;
      *(_QWORD *)v55 = v41;
      *(_DWORD *)(v55 + 8) = v12;
      src = v90;
      if ( v42 == 16 )
        goto LABEL_6;
    }
    qsort(v41, v12, 0x10u, (__compar_fn_t)sub_A16990);
  }
LABEL_4:
  if ( src != v90 )
    _libc_free((unsigned __int64)src);
LABEL_6:
  sub_C7D6A0((__int64)v85, 16LL * (unsigned int)v87, 8);
  return v78;
}
