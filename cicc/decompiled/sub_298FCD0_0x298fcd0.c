// Function: sub_298FCD0
// Address: 0x298fcd0
//
void __fastcall sub_298FCD0(__int64 a1)
{
  _QWORD *v2; // rdi
  _QWORD *v3; // rax
  __int64 *v4; // rsi
  __m128i *v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // rcx
  __int64 v12; // r9
  unsigned __int64 v13; // r12
  __int64 v14; // rax
  unsigned __int64 v15; // rdi
  __m128i *v16; // rdx
  const __m128i *v17; // rax
  const __m128i *v18; // rcx
  __int64 v19; // r9
  unsigned __int64 v20; // r14
  __int64 v21; // rax
  unsigned __int64 v22; // rdi
  __m128i *v23; // rdx
  const __m128i *v24; // rax
  unsigned __int64 v25; // rcx
  unsigned __int64 v26; // r14
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // rsi
  char v29; // al
  unsigned __int64 v31; // rax
  int v32; // r8d
  __int64 v33; // rdx
  _QWORD *v34; // rax
  _QWORD *i; // rdx
  _QWORD *v36; // rax
  _QWORD *v37; // rax
  _QWORD *v38; // rdx
  unsigned int v39; // r12d
  __int64 v40; // rsi
  __int64 v41; // r14
  __int64 v42; // r8
  __int64 v43; // r9
  _QWORD *v44; // rdi
  unsigned __int64 j; // r10
  __int64 v46; // rax
  _QWORD *v47; // rax
  unsigned int v48; // ecx
  __int64 v49; // rdx
  _QWORD *v50; // rdx
  unsigned int *v51; // rdx
  unsigned int v52; // r15d
  int v53; // eax
  _QWORD *v54; // rax
  _QWORD *v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // rax
  int v58; // edx
  unsigned int *v59; // rcx
  __int64 v60; // r15
  unsigned int v61; // eax
  int v62; // ecx
  __int64 v63; // rdi
  int v64; // ecx
  char v65; // al
  _QWORD *v66; // rdx
  _QWORD *v67; // rax
  _QWORD *v68; // rax
  _QWORD *v69; // rdx
  int v70; // [rsp+Ch] [rbp-254h]
  unsigned int v71; // [rsp+Ch] [rbp-254h]
  __int64 v72; // [rsp+30h] [rbp-230h] BYREF
  _QWORD *v73; // [rsp+38h] [rbp-228h]
  __int64 v74; // [rsp+40h] [rbp-220h]
  int v75; // [rsp+48h] [rbp-218h]
  char v76; // [rsp+4Ch] [rbp-214h]
  _QWORD *v77; // [rsp+50h] [rbp-210h] BYREF
  const __m128i *v78; // [rsp+90h] [rbp-1D0h] BYREF
  const __m128i *v79; // [rsp+98h] [rbp-1C8h]
  __int64 v80; // [rsp+A0h] [rbp-1C0h]
  _QWORD v81[16]; // [rsp+B0h] [rbp-1B0h] BYREF
  _BYTE *v82; // [rsp+130h] [rbp-130h] BYREF
  unsigned __int64 v83; // [rsp+138h] [rbp-128h]
  _BYTE v84[16]; // [rsp+140h] [rbp-120h] BYREF
  _BYTE v85[64]; // [rsp+150h] [rbp-110h] BYREF
  unsigned __int64 v86; // [rsp+190h] [rbp-D0h]
  unsigned __int64 v87; // [rsp+198h] [rbp-C8h]
  unsigned __int64 v88; // [rsp+1A0h] [rbp-C0h]
  __m128i v89; // [rsp+1B0h] [rbp-B0h] BYREF
  __int64 v90; // [rsp+1C0h] [rbp-A0h]
  __int64 v91; // [rsp+1C8h] [rbp-98h]
  __int64 v92; // [rsp+1D0h] [rbp-90h] BYREF
  unsigned __int64 v93; // [rsp+1D8h] [rbp-88h]
  __int64 v94; // [rsp+1E0h] [rbp-80h]
  __int64 v95; // [rsp+1E8h] [rbp-78h]
  _QWORD *v96; // [rsp+1F0h] [rbp-70h]
  _QWORD *v97; // [rsp+1F8h] [rbp-68h]
  __int64 v98; // [rsp+200h] [rbp-60h]
  unsigned __int64 v99; // [rsp+208h] [rbp-58h]
  unsigned __int64 v100; // [rsp+210h] [rbp-50h]
  unsigned __int64 v101; // [rsp+218h] [rbp-48h]
  unsigned __int64 v102; // [rsp+220h] [rbp-40h]

  sub_22DE030(*(_QWORD **)(a1 + 40), **(_QWORD **)(a1 + 40) & 0xFFFFFFFFFFFFFFF8LL);
  memset(v81, 0, 0x78u);
  v2 = *(_QWORD **)(a1 + 40);
  LODWORD(v81[2]) = 8;
  v81[1] = &v81[4];
  BYTE4(v81[3]) = 1;
  v3 = sub_22DE030(v2, *v2 & 0xFFFFFFFFFFFFFFF8LL);
  v74 = 0x100000008LL;
  v73 = &v77;
  v77 = v3;
  v89.m128i_i64[0] = (__int64)v3;
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v75 = 0;
  v76 = 1;
  v72 = 1;
  LOBYTE(v92) = 0;
  sub_298C240((unsigned __int64 *)&v78, &v89);
  v4 = &v92;
  v5 = &v89;
  sub_C8CD80((__int64)&v89, (__int64)&v92, (__int64)v81, v6, v7, v8);
  v11 = v81[13];
  v12 = v81[12];
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v13 = v81[13] - v81[12];
  if ( v81[13] == v81[12] )
  {
    v15 = 0;
  }
  else
  {
    if ( v13 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_125;
    v14 = sub_22077B0(v81[13] - v81[12]);
    v11 = v81[13];
    v12 = v81[12];
    v15 = v14;
  }
  v100 = v15;
  v101 = v15;
  v102 = v15 + v13;
  if ( v12 != v11 )
  {
    v16 = (__m128i *)v15;
    v17 = (const __m128i *)v12;
    do
    {
      if ( v16 )
      {
        *v16 = _mm_loadu_si128(v17);
        v16[1] = _mm_loadu_si128(v17 + 1);
        v16[2].m128i_i64[0] = v17[2].m128i_i64[0];
      }
      v17 = (const __m128i *)((char *)v17 + 40);
      v16 = (__m128i *)((char *)v16 + 40);
    }
    while ( v17 != (const __m128i *)v11 );
    v15 += 8 * (((unsigned __int64)&v17[-3].m128i_u64[1] - v12) >> 3) + 40;
  }
  v101 = v15;
  v4 = (__int64 *)v85;
  v5 = (__m128i *)&v82;
  sub_C8CD80((__int64)&v82, (__int64)v85, (__int64)&v72, v11, v10, v12);
  v18 = v79;
  v19 = (__int64)v78;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v20 = (char *)v79 - (char *)v78;
  if ( v79 == v78 )
  {
    v22 = 0;
    goto LABEL_13;
  }
  if ( v20 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_125:
    sub_4261EA(v5, v4, v9);
  v21 = sub_22077B0((char *)v79 - (char *)v78);
  v18 = v79;
  v19 = (__int64)v78;
  v22 = v21;
LABEL_13:
  v86 = v22;
  v87 = v22;
  v88 = v22 + v20;
  if ( v18 == (const __m128i *)v19 )
  {
    v25 = v22;
  }
  else
  {
    v23 = (__m128i *)v22;
    v24 = (const __m128i *)v19;
    do
    {
      if ( v23 )
      {
        *v23 = _mm_loadu_si128(v24);
        v23[1] = _mm_loadu_si128(v24 + 1);
        v23[2].m128i_i64[0] = v24[2].m128i_i64[0];
      }
      v24 = (const __m128i *)((char *)v24 + 40);
      v23 = (__m128i *)((char *)v23 + 40);
    }
    while ( v18 != v24 );
    v25 = v22 + 8 * (((unsigned __int64)&v18[-3].m128i_u64[1] - v19) >> 3) + 40;
  }
  v87 = v25;
  v26 = 0;
  while ( 1 )
  {
    v27 = v100;
    if ( v25 - v22 != v101 - v100 )
      goto LABEL_20;
    if ( v25 == v22 )
      break;
    v28 = v22;
    while ( *(_QWORD *)v28 == *(_QWORD *)v27 )
    {
      v29 = *(_BYTE *)(v28 + 32);
      if ( v29 != *(_BYTE *)(v27 + 32) )
        break;
      if ( v29 )
      {
        if ( !(((*(__int64 *)(v28 + 8) >> 1) & 3) != 0
             ? ((*(__int64 *)(v27 + 8) >> 1) & 3) == ((*(__int64 *)(v28 + 8) >> 1) & 3)
             : *(_DWORD *)(v28 + 24) == *(_DWORD *)(v27 + 24)) )
          break;
      }
      v28 += 40LL;
      v27 += 40LL;
      if ( v25 == v28 )
        goto LABEL_31;
    }
LABEL_20:
    ++v26;
    sub_22DE410((__int64)&v82);
    v22 = v86;
    v25 = v87;
  }
LABEL_31:
  if ( v22 )
    j_j___libc_free_0(v22);
  if ( !v84[12] )
    _libc_free(v83);
  if ( v100 )
    j_j___libc_free_0(v100);
  if ( !BYTE4(v91) )
    _libc_free(v89.m128i_u64[1]);
  v31 = *(unsigned int *)(a1 + 72);
  if ( v26 != v31 )
  {
    v32 = v26;
    if ( v26 < v31 )
    {
      *(_DWORD *)(a1 + 72) = v26;
    }
    else
    {
      if ( v26 > *(unsigned int *)(a1 + 76) )
      {
        sub_C8D5F0(a1 + 64, (const void *)(a1 + 80), v26, 8u, (unsigned int)v26, v19);
        v31 = *(unsigned int *)(a1 + 72);
        v32 = v26;
      }
      v33 = *(_QWORD *)(a1 + 64);
      v34 = (_QWORD *)(v33 + 8 * v31);
      for ( i = (_QWORD *)(v33 + 8 * v26); i != v34; ++v34 )
      {
        if ( v34 )
          *v34 = 0;
      }
      *(_DWORD *)(a1 + 72) = v32;
    }
  }
  if ( v78 )
    j_j___libc_free_0((unsigned __int64)v78);
  if ( !v76 )
    _libc_free((unsigned __int64)v73);
  if ( v81[12] )
    j_j___libc_free_0(v81[12]);
  if ( !BYTE4(v81[3]) )
    _libc_free(v81[1]);
  if ( *(_DWORD *)(a1 + 72) )
  {
    v36 = &v81[2];
    v81[0] = 0;
    v81[1] = 1;
    do
      *v36++ = -4096;
    while ( v36 != &v81[6] );
    v37 = sub_22DE030(*(_QWORD **)(a1 + 40), **(_QWORD **)(a1 + 40) & 0xFFFFFFFFFFFFFFF8LL);
    v38 = 0;
    v39 = 0;
    v40 = (__int64)v37;
    v41 = a1;
    v82 = v84;
    v83 = 0x800000000LL;
    while ( 2 )
    {
      v89 = 0u;
      v90 = 0;
      v91 = 0;
      v92 = 0;
      v93 = 0;
      v94 = 0;
      v95 = 0;
      v96 = 0;
      v97 = 0;
      v98 = 0;
      v99 = 0;
      v100 = 0;
      v101 = 0;
      sub_298E970((__int64)&v89, v40, (__int64)v38);
      sub_298F2D0((__int64)&v89);
      v44 = v97;
      for ( j = (unsigned __int64)v96; v96 != v97; j = (unsigned __int64)v96 )
      {
        v46 = (__int64)((__int64)v44 - j) >> 4;
        if ( (unsigned int)v46 > 2 )
        {
          v56 = (unsigned int)v83;
          v57 = v39 + (unsigned int)v46;
          v58 = v83;
          if ( (unsigned int)v83 >= (unsigned __int64)HIDWORD(v83) )
          {
            v60 = (v57 << 32) | v39;
            if ( HIDWORD(v83) < (unsigned __int64)(unsigned int)v83 + 1 )
            {
              sub_C8D5F0((__int64)&v82, v84, (unsigned int)v83 + 1LL, 8u, v42, v43);
              v56 = (unsigned int)v83;
            }
            *(_QWORD *)&v82[8 * v56] = v60;
            v44 = v97;
            LODWORD(v83) = v83 + 1;
            j = (unsigned __int64)v96;
          }
          else
          {
            v59 = (unsigned int *)&v82[8 * (unsigned int)v83];
            if ( v59 )
            {
              *v59 = v39;
              v59[1] = v57;
              v58 = v83;
              v44 = v97;
              j = (unsigned __int64)v96;
            }
            LODWORD(v83) = v58 + 1;
          }
        }
        if ( v44 != (_QWORD *)j )
        {
          v47 = (_QWORD *)j;
          v48 = v39;
          do
          {
            v49 = v48++;
            *(_QWORD *)(*(_QWORD *)(v41 + 64) + 8 * v49) = *v47;
            v50 = v47;
            v47 += 2;
          }
          while ( v44 != v47 );
          v39 += (((unsigned __int64)v50 - j) >> 4) + 1;
        }
        sub_298F2D0((__int64)&v89);
        v44 = v97;
      }
      if ( v99 )
        j_j___libc_free_0(v99);
      if ( v96 )
        j_j___libc_free_0((unsigned __int64)v96);
      if ( v93 )
        j_j___libc_free_0(v93);
      sub_C7D6A0(v90, 24LL * (unsigned int)v92, 8);
      if ( !(_DWORD)v83 )
      {
        if ( v82 != v84 )
          _libc_free((unsigned __int64)v82);
        if ( (v81[1] & 1) == 0 )
          sub_C7D6A0(v81[2], 8LL * LODWORD(v81[3]), 8);
        return;
      }
      v51 = (unsigned int *)&v82[8 * (unsigned int)v83 - 8];
      v39 = *v51;
      v52 = v51[1];
      LODWORD(v83) = v83 - 1;
      ++v81[0];
      v53 = LODWORD(v81[1]) >> 1;
      if ( !(LODWORD(v81[1]) >> 1) && !HIDWORD(v81[1]) )
        goto LABEL_82;
      if ( (v81[1] & 1) != 0 )
      {
        v55 = &v81[6];
        v54 = &v81[2];
        goto LABEL_80;
      }
      if ( 4 * v53 >= LODWORD(v81[3]) || LODWORD(v81[3]) <= 0x40 )
      {
        v54 = (_QWORD *)v81[2];
        v55 = (_QWORD *)(v81[2] + 8LL * LODWORD(v81[3]));
        if ( (_QWORD *)v81[2] != v55 )
        {
          do
LABEL_80:
            *v54++ = -4096;
          while ( v55 != v54 );
        }
        v81[1] = v81[1] & 1;
LABEL_82:
        sub_298FB00(
          (__int64)v81,
          (__int64 *)(*(_QWORD *)(v41 + 64) + 8LL * v39),
          (__int64 *)(*(_QWORD *)(v41 + 64) + 8LL * v52 - 8));
        v40 = *(_QWORD *)(*(_QWORD *)(v41 + 64) + 8LL * (v52 - 1));
        v38 = v81;
        continue;
      }
      break;
    }
    if ( v53 && (v61 = v53 - 1) != 0 )
    {
      _BitScanReverse(&v61, v61);
      v62 = 1 << (33 - (v61 ^ 0x1F));
      if ( (unsigned int)(v62 - 5) <= 0x3A )
      {
        sub_C7D6A0(v81[2], 8LL * LODWORD(v81[3]), 8);
        v63 = 512;
        v64 = 64;
        v65 = v81[1];
        goto LABEL_104;
      }
      if ( LODWORD(v81[3]) == v62 )
      {
        v81[1] = v81[1] & 1;
        if ( v81[1] )
        {
          v69 = &v81[6];
          v68 = &v81[2];
        }
        else
        {
          v68 = (_QWORD *)v81[2];
          v69 = (_QWORD *)(v81[2] + 8LL * LODWORD(v81[3]));
        }
        do
        {
          if ( v68 )
            *v68 = -4096;
          ++v68;
        }
        while ( v68 != v69 );
        goto LABEL_82;
      }
      v71 = 1 << (33 - (v61 ^ 0x1F));
      sub_C7D6A0(v81[2], 8LL * LODWORD(v81[3]), 8);
      v64 = v71;
      v65 = LOBYTE(v81[1]) | 1;
      LOBYTE(v81[1]) |= 1u;
      if ( v71 > 4 )
      {
        v63 = 8LL * v71;
LABEL_104:
        v70 = v64;
        LOBYTE(v81[1]) = v65 & 0xFE;
        v81[2] = sub_C7D670(v63, 8);
        LODWORD(v81[3]) = v70;
      }
    }
    else
    {
      sub_C7D6A0(v81[2], 8LL * LODWORD(v81[3]), 8);
      LOBYTE(v81[1]) |= 1u;
    }
    v81[1] = v81[1] & 1;
    if ( v81[1] )
    {
      v66 = &v81[6];
      v67 = &v81[2];
    }
    else
    {
      v67 = (_QWORD *)v81[2];
      v66 = (_QWORD *)(v81[2] + 8LL * LODWORD(v81[3]));
      if ( (_QWORD *)v81[2] == v66 )
        goto LABEL_82;
    }
    do
    {
      if ( v67 )
        *v67 = -4096;
      ++v67;
    }
    while ( v66 != v67 );
    goto LABEL_82;
  }
}
