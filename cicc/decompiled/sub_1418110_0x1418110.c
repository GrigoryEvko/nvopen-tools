// Function: sub_1418110
// Address: 0x1418110
//
const __m128i **__fastcall sub_1418110(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // r13
  __int64 v4; // rdi
  unsigned int v5; // esi
  __int64 v6; // rcx
  __int64 v7; // rdx
  char *v8; // rbx
  __int64 v9; // rax
  __int64 *v10; // r8
  __int64 *v11; // r14
  __int64 *v12; // r15
  __int64 v13; // r8
  __int64 v14; // rax
  __m128i *v15; // r8
  __m128i *v16; // rdi
  __int64 v17; // r14
  unsigned __int64 v18; // rax
  __m128i *i; // rsi
  unsigned __int64 v20; // rcx
  __int64 v21; // rdi
  __int64 *v22; // rdx
  const __m128i *j; // rax
  __m128i v24; // xmm0
  unsigned __int8 v25; // al
  __int64 v26; // rsi
  unsigned __int8 v27; // al
  _BYTE *v28; // r9
  __int64 v29; // rdx
  __int64 v30; // rax
  _BYTE *v31; // r11
  _QWORD *v32; // r15
  char v33; // dl
  _QWORD *v34; // rbx
  _QWORD *v35; // rax
  _QWORD *v36; // r10
  __int64 v37; // rax
  char *v38; // r14
  __int64 v39; // rdx
  char *v40; // rbx
  __int64 v41; // r15
  __int64 v43; // rsi
  unsigned __int64 v44; // rsi
  _QWORD *v45; // rcx
  unsigned __int64 v46; // rax
  int v47; // ecx
  __int64 *v48; // rsi
  unsigned __int64 *v49; // rax
  _QWORD *v50; // rsi
  _QWORD *v51; // rax
  _QWORD *v52; // rcx
  unsigned __int64 *v53; // rdi
  unsigned int v54; // r8d
  unsigned __int64 *v55; // rcx
  __m128i *v56; // rsi
  __int64 v57; // rdx
  char *v58; // r14
  __int64 v59; // rax
  char *v60; // rbx
  __int64 v61; // r15
  int v62; // r10d
  char *v63; // r9
  int v64; // eax
  int v65; // edx
  int v66; // r8d
  int v67; // r8d
  __int64 v68; // r9
  unsigned int v69; // eax
  __int64 v70; // rsi
  int v71; // ecx
  char *v72; // rdi
  int v73; // edi
  int v74; // edi
  __int64 v75; // r8
  char *v76; // rsi
  unsigned int v77; // r14d
  __int64 v78; // rcx
  int v79; // eax
  unsigned __int64 v80; // [rsp+8h] [rbp-2B8h]
  const __m128i **v81; // [rsp+10h] [rbp-2B0h]
  int v82; // [rsp+18h] [rbp-2A8h]
  char v83; // [rsp+1Fh] [rbp-2A1h]
  __m128i *v85; // [rsp+30h] [rbp-290h]
  __int64 v86; // [rsp+30h] [rbp-290h]
  __int64 v87; // [rsp+30h] [rbp-290h]
  _QWORD *src; // [rsp+38h] [rbp-288h]
  char *srca; // [rsp+38h] [rbp-288h]
  __m128i v90; // [rsp+40h] [rbp-280h] BYREF
  _BYTE *v91; // [rsp+50h] [rbp-270h] BYREF
  __int64 v92; // [rsp+58h] [rbp-268h]
  _BYTE v93[256]; // [rsp+60h] [rbp-260h] BYREF
  __int64 v94; // [rsp+160h] [rbp-160h] BYREF
  _BYTE *v95; // [rsp+168h] [rbp-158h]
  _BYTE *v96; // [rsp+170h] [rbp-150h]
  __int64 v97; // [rsp+178h] [rbp-148h]
  int v98; // [rsp+180h] [rbp-140h]
  _BYTE v99[312]; // [rsp+188h] [rbp-138h] BYREF

  v3 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v4 = a1 + 160;
  v5 = *(_DWORD *)(a1 + 184);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 160);
    goto LABEL_109;
  }
  v6 = *(_QWORD *)(a1 + 168);
  LODWORD(v7) = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v8 = (char *)(v6 + 40LL * (unsigned int)v7);
  v9 = *(_QWORD *)v8;
  if ( v3 == *(_QWORD *)v8 )
    goto LABEL_3;
  v62 = 1;
  v63 = 0;
  while ( v9 != -8 )
  {
    if ( v9 == -16 && !v63 )
      v63 = v8;
    v7 = (v5 - 1) & ((_DWORD)v7 + v62);
    v8 = (char *)(v6 + 40 * v7);
    v9 = *(_QWORD *)v8;
    if ( v3 == *(_QWORD *)v8 )
      goto LABEL_3;
    ++v62;
  }
  v64 = *(_DWORD *)(a1 + 176);
  if ( v63 )
    v8 = v63;
  ++*(_QWORD *)(a1 + 160);
  v65 = v64 + 1;
  if ( 4 * (v64 + 1) >= 3 * v5 )
  {
LABEL_109:
    sub_1417EE0(v4, 2 * v5);
    v66 = *(_DWORD *)(a1 + 184);
    if ( v66 )
    {
      v67 = v66 - 1;
      v68 = *(_QWORD *)(a1 + 168);
      v69 = v67 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v8 = (char *)(v68 + 40LL * v69);
      v65 = *(_DWORD *)(a1 + 176) + 1;
      v70 = *(_QWORD *)v8;
      if ( v3 != *(_QWORD *)v8 )
      {
        v71 = 1;
        v72 = 0;
        while ( v70 != -8 )
        {
          if ( !v72 && v70 == -16 )
            v72 = v8;
          v69 = v67 & (v71 + v69);
          v8 = (char *)(v68 + 40LL * v69);
          v70 = *(_QWORD *)v8;
          if ( v3 == *(_QWORD *)v8 )
            goto LABEL_105;
          ++v71;
        }
        if ( v72 )
          v8 = v72;
      }
      goto LABEL_105;
    }
    goto LABEL_139;
  }
  if ( v5 - *(_DWORD *)(a1 + 180) - v65 <= v5 >> 3 )
  {
    sub_1417EE0(v4, v5);
    v73 = *(_DWORD *)(a1 + 184);
    if ( v73 )
    {
      v74 = v73 - 1;
      v75 = *(_QWORD *)(a1 + 168);
      v76 = 0;
      v77 = v74 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v8 = (char *)(v75 + 40LL * v77);
      v78 = *(_QWORD *)v8;
      v65 = *(_DWORD *)(a1 + 176) + 1;
      v79 = 1;
      if ( v3 != *(_QWORD *)v8 )
      {
        while ( v78 != -8 )
        {
          if ( v78 == -16 && !v76 )
            v76 = v8;
          v77 = v74 & (v79 + v77);
          v8 = (char *)(v75 + 40LL * v77);
          v78 = *(_QWORD *)v8;
          if ( v3 == *(_QWORD *)v8 )
            goto LABEL_105;
          ++v79;
        }
        if ( v76 )
          v8 = v76;
      }
      goto LABEL_105;
    }
LABEL_139:
    ++*(_DWORD *)(a1 + 176);
    BUG();
  }
LABEL_105:
  *(_DWORD *)(a1 + 176) = v65;
  if ( *(_QWORD *)v8 != -8 )
    --*(_DWORD *)(a1 + 180);
  *(_QWORD *)v8 = v3;
  *((_QWORD *)v8 + 1) = 0;
  *((_QWORD *)v8 + 2) = 0;
  *((_QWORD *)v8 + 3) = 0;
  v8[32] = 0;
LABEL_3:
  v10 = (__int64 *)*((_QWORD *)v8 + 2);
  v81 = (const __m128i **)(v8 + 8);
  v91 = v93;
  v92 = 0x2000000000LL;
  v11 = (__int64 *)*((_QWORD *)v8 + 1);
  if ( v11 == v10 )
  {
    v58 = sub_1416080(a1 + 296, *(_QWORD *)(v3 + 40));
    v59 = (unsigned int)v92;
    if ( &v58[8 * v57] != v58 )
    {
      srca = v8;
      v60 = &v58[8 * v57];
      do
      {
        v61 = *(_QWORD *)v58;
        if ( HIDWORD(v92) <= (unsigned int)v59 )
        {
          sub_16CD150(&v91, v93, 0, 8);
          v59 = (unsigned int)v92;
        }
        v58 += 8;
        *(_QWORD *)&v91[8 * v59] = v61;
        v59 = (unsigned int)(v92 + 1);
        LODWORD(v92) = v92 + 1;
      }
      while ( v60 != v58 );
      v8 = srca;
    }
  }
  else
  {
    if ( !v8[32] )
      return v81;
    v12 = v10;
    do
    {
      while ( (v11[1] & 7) != 0 )
      {
        v11 += 2;
        if ( v12 == v11 )
          goto LABEL_11;
      }
      v13 = *v11;
      v14 = (unsigned int)v92;
      if ( (unsigned int)v92 >= HIDWORD(v92) )
      {
        v87 = *v11;
        sub_16CD150(&v91, v93, 0, 8);
        v14 = (unsigned int)v92;
        v13 = v87;
      }
      v11 += 2;
      *(_QWORD *)&v91[8 * v14] = v13;
      LODWORD(v92) = v92 + 1;
    }
    while ( v12 != v11 );
LABEL_11:
    v15 = (__m128i *)*((_QWORD *)v8 + 2);
    v16 = (__m128i *)*((_QWORD *)v8 + 1);
    if ( v15 != v16 )
    {
      v85 = (__m128i *)*((_QWORD *)v8 + 2);
      v17 = (char *)v15 - (char *)v16;
      _BitScanReverse64(&v18, v15 - v16);
      sub_1411980(v16, v85, 2LL * (int)(63 - (v18 ^ 0x3F)));
      if ( v17 <= 256 )
      {
        sub_1411BF0((unsigned __int64 *)v16, (unsigned __int64 *)v85);
      }
      else
      {
        sub_1411BF0((unsigned __int64 *)v16, (unsigned __int64 *)&v16[16]);
        for ( i = v16 + 16; i != v85; v22[1] = v21 )
        {
          v20 = i->m128i_i64[0];
          v21 = i->m128i_i64[1];
          v22 = (__int64 *)i;
          for ( j = i - 1; v20 < j->m128i_i64[0]; j[2] = v24 )
          {
            v24 = _mm_loadu_si128(j);
            v22 = (__int64 *)j--;
          }
          ++i;
          *v22 = v20;
        }
      }
    }
  }
  v25 = *(_BYTE *)(v3 + 16);
  v26 = 0;
  if ( v25 > 0x17u )
  {
    if ( v25 == 78 )
    {
      v26 = v3 | 4;
    }
    else if ( v25 == 29 )
    {
      v26 = v3;
    }
  }
  v27 = sub_134CC90(*(_QWORD *)(a1 + 256), v26);
  v28 = v99;
  v97 = 32;
  v96 = v99;
  v98 = 0;
  v95 = v99;
  v83 = ((v27 >> 1) ^ 1) & 1;
  v30 = *((_QWORD *)v8 + 2);
  v94 = 0;
  v29 = (v30 - *((_QWORD *)v8 + 1)) >> 4;
  LODWORD(v30) = v92;
  if ( (_DWORD)v92 )
  {
    src = v8;
    v31 = v99;
    v86 = 2LL * (unsigned int)v29;
    while ( 1 )
    {
      v32 = *(_QWORD **)&v91[8 * (unsigned int)v30 - 8];
      LODWORD(v92) = v30 - 1;
      if ( v28 != v31 )
        goto LABEL_24;
      v50 = &v28[8 * HIDWORD(v97)];
      if ( v28 != (_BYTE *)v50 )
      {
        v51 = v28;
        v52 = 0;
        while ( v32 != (_QWORD *)*v51 )
        {
          if ( *v51 == -2 )
            v52 = v51;
          if ( v50 == ++v51 )
          {
            if ( !v52 )
              goto LABEL_73;
            *v52 = v32;
            --v98;
            ++v94;
            goto LABEL_25;
          }
        }
LABEL_63:
        LODWORD(v30) = v92;
        goto LABEL_40;
      }
LABEL_73:
      if ( HIDWORD(v97) < (unsigned int)v97 )
      {
        ++HIDWORD(v97);
        *v50 = v32;
        ++v94;
      }
      else
      {
LABEL_24:
        sub_16CCBA0(&v94, v32);
        v31 = v96;
        v28 = v95;
        if ( !v33 )
          goto LABEL_63;
      }
LABEL_25:
      v90 = (__m128i)(unsigned __int64)v32;
      v34 = (_QWORD *)src[1];
      v35 = sub_1411920(v34, (__int64)&v34[v86], (unsigned __int64 *)&v90);
      if ( v35 != v34 )
      {
        if ( v32 == (_QWORD *)*(v35 - 2) )
        {
          v34 = v35 - 2;
          if ( v36 != v35 - 2 )
            goto LABEL_47;
          goto LABEL_30;
        }
        v34 = v35;
      }
      if ( v34 != v36 && v32 == (_QWORD *)*v34 )
      {
LABEL_47:
        v43 = v34[1];
        if ( (v43 & 7) != 0 )
          goto LABEL_63;
        v44 = v43 & 0xFFFFFFFFFFFFFFF8LL;
        v45 = v32 + 5;
        if ( v44 )
        {
          sub_1411E70(a1 + 224, v44, v3);
          v45 = (_QWORD *)(v44 + 24);
        }
        if ( (_QWORD *)v32[6] != v45 )
        {
          v46 = sub_1412B20(a1, a2, v83, v45, (__int64)v32);
          v34[1] = v46;
          v47 = v46 & 7;
          goto LABEL_52;
        }
        v37 = *(_QWORD *)(v32[7] + 80LL);
        if ( !v37 )
        {
LABEL_34:
          v34[1] = 0x2000000000000003LL;
LABEL_35:
          v38 = sub_1416080(a1 + 296, (__int64)v32);
          v40 = &v38[8 * v39];
          v30 = (unsigned int)v92;
          if ( v38 != v40 )
          {
            do
            {
              v41 = *(_QWORD *)v38;
              if ( (unsigned int)v30 >= HIDWORD(v92) )
              {
                sub_16CD150(&v91, v93, 0, 8);
                v30 = (unsigned int)v92;
              }
              v38 += 8;
              *(_QWORD *)&v91[8 * v30] = v41;
              v30 = (unsigned int)(v92 + 1);
              LODWORD(v92) = v92 + 1;
            }
            while ( v40 != v38 );
            goto LABEL_39;
          }
          goto LABEL_56;
        }
LABEL_32:
        if ( v32 == (_QWORD *)(v37 - 24) )
        {
          if ( v34 )
          {
            v34[1] = 0x4000000000000003LL;
            LODWORD(v30) = v92;
            goto LABEL_39;
          }
          v46 = 0x4000000000000003LL;
          v47 = 3;
          goto LABEL_82;
        }
        if ( v34 )
          goto LABEL_34;
        goto LABEL_94;
      }
LABEL_30:
      if ( (_QWORD *)v32[6] != v32 + 5 )
      {
        v46 = sub_1412B20(a1, a2, v83, v32 + 5, (__int64)v32);
        v47 = v46 & 7;
        goto LABEL_82;
      }
      v34 = 0;
      v37 = *(_QWORD *)(v32[7] + 80LL);
      if ( v37 )
        goto LABEL_32;
LABEL_94:
      v46 = 0x2000000000000003LL;
      v47 = 3;
LABEL_82:
      v90.m128i_i64[0] = (__int64)v32;
      v90.m128i_i64[1] = v46;
      v56 = (__m128i *)src[2];
      if ( v56 == (__m128i *)src[3] )
      {
        v82 = v47;
        v80 = v46;
        sub_1414B00(v81, v56, &v90);
        v47 = v82;
        v46 = v80;
      }
      else
      {
        if ( v56 )
        {
          *v56 = _mm_loadu_si128(&v90);
          v56 = (__m128i *)src[2];
        }
        src[2] = v56 + 1;
      }
LABEL_52:
      if ( v47 == 3 )
      {
        if ( v46 >> 61 != 1 )
        {
          LODWORD(v30) = v92;
          goto LABEL_39;
        }
        goto LABEL_35;
      }
      v90.m128i_i64[0] = v46 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v46 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v48 = sub_1417C70(a1 + 224, v90.m128i_i64);
        v49 = (unsigned __int64 *)v48[2];
        if ( (unsigned __int64 *)v48[3] != v49 )
          goto LABEL_55;
        v53 = &v49[*((unsigned int *)v48 + 9)];
        v54 = *((_DWORD *)v48 + 9);
        if ( v49 != v53 )
        {
          v55 = 0;
          while ( v3 != *v49 )
          {
            if ( *v49 == -2 )
              v55 = v49;
            if ( v53 == ++v49 )
            {
              if ( !v55 )
                goto LABEL_95;
              *v55 = v3;
              ++v48[1];
              LODWORD(v30) = v92;
              --*((_DWORD *)v48 + 10);
              goto LABEL_39;
            }
          }
          goto LABEL_56;
        }
LABEL_95:
        if ( v54 < *((_DWORD *)v48 + 8) )
        {
          *((_DWORD *)v48 + 9) = v54 + 1;
          *v53 = v3;
          ++v48[1];
        }
        else
        {
LABEL_55:
          sub_16CCBA0(v48 + 1, v3);
        }
      }
LABEL_56:
      LODWORD(v30) = v92;
LABEL_39:
      v31 = v96;
      v28 = v95;
LABEL_40:
      if ( !(_DWORD)v30 )
      {
        if ( v28 != v31 )
          _libc_free((unsigned __int64)v31);
        break;
      }
    }
  }
  if ( v91 != v93 )
    _libc_free((unsigned __int64)v91);
  return v81;
}
