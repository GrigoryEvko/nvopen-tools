// Function: sub_10730D0
// Address: 0x10730d0
//
__int64 __fastcall sub_10730D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 *v6; // r12
  __int64 *v7; // r10
  unsigned int v8; // esi
  __int64 v9; // rcx
  int v10; // r13d
  int v11; // r11d
  __int64 v12; // rdx
  unsigned int v13; // r8d
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rbx
  char v17; // r15
  unsigned int v18; // ecx
  int v19; // eax
  __int64 v20; // rdi
  __int64 v21; // r13
  __int64 *v22; // r15
  __int64 *v23; // r8
  __int64 v24; // rsi
  _QWORD *v25; // r14
  unsigned int v26; // r12d
  unsigned __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // r8
  unsigned int v30; // r14d
  int v31; // r11d
  __int64 v32; // rsi
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 *v36; // r15
  __int64 *v37; // rbx
  __int64 v38; // r15
  char v39; // al
  __int64 *v40; // r9
  __int64 v41; // rsi
  char *v42; // r14
  unsigned int v43; // r12d
  unsigned __int64 v44; // rax
  _QWORD *v45; // rax
  __int64 v46; // rsi
  unsigned int v47; // edx
  __int64 *v48; // rax
  __int64 v49; // r11
  __int64 v50; // r14
  __int64 *v51; // r13
  __int64 *v52; // rbx
  __int64 *v53; // r10
  __int64 v54; // rsi
  char *v55; // r10
  unsigned __int64 v56; // rax
  _QWORD *v57; // rax
  __int64 v58; // rdi
  unsigned int v59; // edx
  __int64 *v60; // rax
  __int64 v61; // r12
  char v62; // r15
  __int64 v63; // rsi
  __int64 v64; // r12
  char v65; // al
  void *v66; // rax
  char v67; // al
  __int64 v68; // rsi
  void *v69; // rax
  const __m128i *v70; // r13
  __m128i *v71; // r12
  unsigned __int64 v72; // rax
  const __m128i *v73; // r13
  __m128i *v74; // r12
  unsigned __int64 v75; // rax
  char **v76; // rax
  int v77; // r8d
  __m128i *j; // rdi
  char *v79; // r11
  char *v80; // rcx
  char *v81; // rax
  int v82; // edx
  __int64 v83; // rsi
  int v84; // r10d
  __int64 v85; // r10
  __int64 *v86; // r12
  __int64 *v87; // r15
  unsigned int v88; // esi
  __int64 v89; // r13
  __int64 v90; // r8
  int v91; // r11d
  _QWORD *v92; // rdx
  unsigned int v93; // edi
  _QWORD *v94; // rax
  __int64 v95; // rcx
  _DWORD *v96; // rcx
  _DWORD *v97; // rdx
  int v98; // esi
  int v99; // eax
  int v101; // eax
  int v102; // ecx
  int v103; // eax
  int v104; // eax
  int v105; // edi
  __int64 v106; // rsi
  unsigned int v107; // eax
  __int64 v108; // r8
  int v109; // r11d
  _QWORD *v110; // r9
  int v111; // eax
  int v112; // eax
  __int64 v113; // rdi
  _QWORD *v114; // r8
  unsigned int v115; // r14d
  int v116; // r9d
  __int64 v117; // rsi
  int v118; // eax
  int v119; // r14d
  __int64 v120; // r9
  __int64 v121; // [rsp+0h] [rbp-B0h]
  __int64 *v123; // [rsp+10h] [rbp-A0h]
  __int64 *v124; // [rsp+10h] [rbp-A0h]
  __int64 v125; // [rsp+10h] [rbp-A0h]
  __int64 v126; // [rsp+18h] [rbp-98h]
  char *v127; // [rsp+18h] [rbp-98h]
  __int64 *i; // [rsp+30h] [rbp-80h]
  __int64 *v132; // [rsp+30h] [rbp-80h]
  unsigned int v133; // [rsp+30h] [rbp-80h]
  __int64 v135; // [rsp+38h] [rbp-78h]
  __int64 v136; // [rsp+38h] [rbp-78h]
  __m128i v137; // [rsp+40h] [rbp-70h] BYREF
  __int64 v138; // [rsp+50h] [rbp-60h]
  char v139; // [rsp+58h] [rbp-58h] BYREF
  __int64 v140; // [rsp+60h] [rbp-50h] BYREF
  __int64 v141; // [rsp+68h] [rbp-48h]
  __int64 v142; // [rsp+70h] [rbp-40h]
  unsigned int v143; // [rsp+78h] [rbp-38h]

  v5 = a1;
  v6 = *(__int64 **)(a2 + 40);
  v7 = &v6[*(unsigned int *)(a2 + 48)];
  v140 = 0;
  v141 = 0;
  v142 = 0;
  v143 = 0;
  if ( v6 != v7 )
  {
    v8 = 0;
    v9 = 0;
    v10 = 1;
    while ( 1 )
    {
      v16 = *v6;
      v17 = v10;
      if ( !v8 )
        break;
      v11 = 1;
      v12 = 0;
      v13 = (v8 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v14 = v9 + 16LL * v13;
      v15 = *(_QWORD *)v14;
      if ( v16 == *(_QWORD *)v14 )
      {
LABEL_4:
        ++v6;
        ++v10;
        *(_BYTE *)(v14 + 8) = v17;
        if ( v7 == v6 )
          goto LABEL_13;
        goto LABEL_5;
      }
      while ( v15 != -4096 )
      {
        if ( v15 == -8192 && !v12 )
          v12 = v14;
        v13 = (v8 - 1) & (v11 + v13);
        v14 = v9 + 16LL * v13;
        v15 = *(_QWORD *)v14;
        if ( v16 == *(_QWORD *)v14 )
          goto LABEL_4;
        ++v11;
      }
      if ( !v12 )
        v12 = v14;
      ++v140;
      v19 = v142 + 1;
      if ( 4 * ((int)v142 + 1) >= 3 * v8 )
        goto LABEL_8;
      if ( v8 - (v19 + HIDWORD(v142)) <= v8 >> 3 )
      {
        v124 = v7;
        sub_1072EF0((__int64)&v140, v8);
        if ( !v143 )
        {
LABEL_197:
          LODWORD(v142) = v142 + 1;
          BUG();
        }
        v29 = 0;
        v30 = (v143 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v7 = v124;
        v31 = 1;
        v19 = v142 + 1;
        v12 = v141 + 16LL * v30;
        v32 = *(_QWORD *)v12;
        if ( v16 != *(_QWORD *)v12 )
        {
          while ( v32 != -4096 )
          {
            if ( !v29 && v32 == -8192 )
              v29 = v12;
            v30 = (v143 - 1) & (v31 + v30);
            v12 = v141 + 16LL * v30;
            v32 = *(_QWORD *)v12;
            if ( v16 == *(_QWORD *)v12 )
              goto LABEL_10;
            ++v31;
          }
          if ( v29 )
            v12 = v29;
        }
      }
LABEL_10:
      LODWORD(v142) = v19;
      if ( *(_QWORD *)v12 != -4096 )
        --HIDWORD(v142);
      ++v6;
      *(_QWORD *)v12 = v16;
      ++v10;
      *(_BYTE *)(v12 + 8) = 0;
      *(_BYTE *)(v12 + 8) = v17;
      if ( v7 == v6 )
      {
LABEL_13:
        v5 = a1;
        goto LABEL_14;
      }
LABEL_5:
      v9 = v141;
      v8 = v143;
    }
    ++v140;
LABEL_8:
    v123 = v7;
    sub_1072EF0((__int64)&v140, 2 * v8);
    if ( !v143 )
      goto LABEL_197;
    v7 = v123;
    v18 = (v143 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
    v19 = v142 + 1;
    v12 = v141 + 16LL * v18;
    v20 = *(_QWORD *)v12;
    if ( v16 != *(_QWORD *)v12 )
    {
      v119 = 1;
      v120 = 0;
      while ( v20 != -4096 )
      {
        if ( v20 == -8192 && !v120 )
          v120 = v12;
        v18 = (v143 - 1) & (v119 + v18);
        v12 = v141 + 16LL * v18;
        v20 = *(_QWORD *)v12;
        if ( v16 == *(_QWORD *)v12 )
          goto LABEL_10;
        ++v119;
      }
      if ( v120 )
        v12 = v120;
    }
    goto LABEL_10;
  }
LABEL_14:
  v21 = v5 + 272;
  v22 = *(__int64 **)(a2 + 56);
  for ( i = &v22[*(unsigned int *)(a2 + 64)]; i != v22; ++v22 )
  {
    v28 = *v22;
    if ( (*(_BYTE *)(*v22 + 8) & 2) == 0 || (*(_BYTE *)(v28 + 9) & 8) != 0 )
    {
      if ( (*(_BYTE *)(*v22 + 8) & 1) != 0 )
      {
        v23 = *(__int64 **)(v28 - 8);
        v24 = *v23;
        v25 = v23 + 3;
        v26 = *v23;
      }
      else
      {
        v26 = 0;
        v24 = 0;
        v25 = 0;
      }
      v27 = sub_C94890(v25, v24);
      sub_C0CA60(v5 + 272, (__int64)v25, (v27 << 32) | v26);
    }
  }
  sub_C0D290(v5 + 272);
  v36 = *(__int64 **)(a2 + 56);
  v132 = &v36[*(unsigned int *)(a2 + 64)];
  if ( v36 != v132 )
  {
    v126 = v5;
    v37 = *(__int64 **)(a2 + 56);
    while ( 1 )
    {
      while ( 1 )
      {
        v38 = *v37;
        v39 = *(_BYTE *)(*v37 + 8);
        if ( (v39 & 2) != 0 && (*(_BYTE *)(v38 + 9) & 8) == 0 )
          goto LABEL_42;
        if ( (v39 & 0x20) == 0 )
          break;
LABEL_46:
        v137.m128i_i64[0] = v38;
        if ( (v39 & 1) != 0 )
        {
          v40 = *(__int64 **)(v38 - 8);
          v41 = *v40;
          v42 = (char *)(v40 + 3);
          v43 = *v40;
        }
        else
        {
          v43 = 0;
          v41 = 0;
          v42 = 0;
        }
        v44 = sub_C94890(v42, v41);
        v137.m128i_i64[1] = sub_C0C3A0(v21, v42, (v44 << 32) | v43);
        v45 = *(_QWORD **)v38;
        if ( !*(_QWORD *)v38 )
        {
          if ( (*(_BYTE *)(v38 + 9) & 0x70) != 0x20
            || *(char *)(v38 + 8) < 0
            || (*(_BYTE *)(v38 + 8) |= 8u, v45 = sub_E807D0(*(_QWORD *)(v38 + 24)), (*(_QWORD *)v38 = v45) == 0) )
          {
            LOBYTE(v138) = 0;
            v68 = *(_QWORD *)(a5 + 8);
            if ( v68 == *(_QWORD *)(a5 + 16) )
            {
              sub_10727E0(a5, (_BYTE *)v68, &v137);
            }
            else
            {
              if ( v68 )
              {
                *(__m128i *)v68 = _mm_loadu_si128(&v137);
                *(_QWORD *)(v68 + 16) = v138;
                v68 = *(_QWORD *)(a5 + 8);
              }
              *(_QWORD *)(a5 + 8) = v68 + 24;
            }
            goto LABEL_42;
          }
        }
        v46 = *(_QWORD *)(a4 + 8);
        v35 = *(_QWORD *)(a4 + 16);
        if ( off_4C5D170 == (_UNKNOWN *)v45 )
        {
          LOBYTE(v138) = 0;
          if ( v46 != v35 )
          {
            if ( !v46 )
              goto LABEL_55;
LABEL_54:
            *(__m128i *)v46 = _mm_loadu_si128(&v137);
            *(_QWORD *)(v46 + 16) = v138;
            v46 = *(_QWORD *)(a4 + 8);
            goto LABEL_55;
          }
LABEL_146:
          sub_10727E0(a4, (_BYTE *)v46, &v137);
          goto LABEL_42;
        }
        v34 = v45[1];
        if ( v143 )
        {
          v47 = (v143 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
          v48 = (__int64 *)(v141 + 16LL * v47);
          v49 = *v48;
          if ( v34 == *v48 )
          {
LABEL_52:
            LOBYTE(v138) = *((_BYTE *)v48 + 8);
            if ( v46 != v35 )
              goto LABEL_53;
            goto LABEL_146;
          }
          v103 = 1;
          while ( v49 != -4096 )
          {
            v33 = (unsigned int)(v103 + 1);
            v47 = (v143 - 1) & (v103 + v47);
            v48 = (__int64 *)(v141 + 16LL * v47);
            v49 = *v48;
            if ( v34 == *v48 )
              goto LABEL_52;
            v103 = v33;
          }
        }
        LOBYTE(v138) = 0;
        if ( v46 == v35 )
          goto LABEL_146;
LABEL_53:
        if ( v46 )
          goto LABEL_54;
LABEL_55:
        ++v37;
        *(_QWORD *)(a4 + 8) = v46 + 24;
        if ( v132 == v37 )
        {
LABEL_56:
          v5 = v126;
          v50 = *(_QWORD *)(a2 + 56);
          if ( v50 == v50 + 8LL * *(unsigned int *)(a2 + 64) )
            goto LABEL_92;
          v125 = v21;
          v51 = *(__int64 **)(a2 + 56);
          v121 = v126;
          v52 = (__int64 *)(v50 + 8LL * *(unsigned int *)(a2 + 64));
          while ( 1 )
          {
            v64 = *v51;
            v65 = *(_BYTE *)(*v51 + 8);
            if ( (v65 & 2) != 0 && (*(_BYTE *)(v64 + 9) & 8) == 0 )
              goto LABEL_69;
            v62 = v65 & 0x20;
            if ( (v65 & 0x20) != 0 )
              goto LABEL_69;
            if ( *(_QWORD *)v64 )
            {
              v137.m128i_i64[0] = *v51;
              if ( (v65 & 1) != 0 )
                goto LABEL_59;
            }
            else
            {
              if ( (*(_BYTE *)(v64 + 9) & 0x70) != 0x20 )
                goto LABEL_69;
              if ( v65 < 0 )
                goto LABEL_69;
              *(_BYTE *)(v64 + 8) |= 8u;
              v66 = sub_E807D0(*(_QWORD *)(v64 + 24));
              *(_QWORD *)v64 = v66;
              if ( !v66 )
                goto LABEL_69;
              v67 = *(_BYTE *)(v64 + 8);
              v137.m128i_i64[0] = v64;
              if ( (v67 & 1) != 0 )
              {
LABEL_59:
                v53 = *(__int64 **)(v64 - 8);
                v54 = *v53;
                v55 = (char *)(v53 + 3);
                v133 = v54;
                goto LABEL_60;
              }
            }
            v133 = 0;
            v54 = 0;
            v55 = 0;
LABEL_60:
            v127 = v55;
            v56 = sub_C94890(v55, v54);
            v137.m128i_i64[1] = sub_C0C3A0(v125, v127, (v56 << 32) | v133);
            v57 = *(_QWORD **)v64;
            if ( *(_QWORD *)v64 )
            {
              if ( v57 == (_QWORD *)off_4C5D170 )
                goto LABEL_140;
            }
            else
            {
              if ( (*(_BYTE *)(v64 + 9) & 0x70) != 0x20 )
              {
                if ( off_4C5D170 )
LABEL_198:
                  BUG();
LABEL_140:
                LOBYTE(v138) = 0;
                v63 = *(_QWORD *)(a3 + 8);
                if ( v63 == *(_QWORD *)(a3 + 16) )
                  goto LABEL_157;
                if ( v63 )
                  goto LABEL_67;
                goto LABEL_68;
              }
              if ( *(char *)(v64 + 8) < 0 )
              {
                if ( !off_4C5D170 )
                  goto LABEL_140;
                goto LABEL_137;
              }
              *(_BYTE *)(v64 + 8) |= 8u;
              v57 = sub_E807D0(*(_QWORD *)(v64 + 24));
              *(_QWORD *)v64 = v57;
              if ( off_4C5D170 == (_UNKNOWN *)v57 )
                goto LABEL_140;
              if ( !v57 )
              {
                if ( (*(_BYTE *)(v64 + 9) & 0x70) != 0x20 )
                  goto LABEL_198;
LABEL_137:
                if ( *(char *)(v64 + 8) < 0 )
                  goto LABEL_198;
                *(_BYTE *)(v64 + 8) |= 8u;
                v57 = sub_E807D0(*(_QWORD *)(v64 + 24));
                *(_QWORD *)v64 = v57;
              }
            }
            v58 = v57[1];
            if ( v143 )
            {
              v59 = (v143 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
              v60 = (__int64 *)(v141 + 16LL * v59);
              v61 = *v60;
              if ( v58 == *v60 )
              {
LABEL_64:
                v62 = *((_BYTE *)v60 + 8);
              }
              else
              {
                v118 = 1;
                while ( v61 != -4096 )
                {
                  v33 = (unsigned int)(v118 + 1);
                  v59 = (v143 - 1) & (v118 + v59);
                  v60 = (__int64 *)(v141 + 16LL * v59);
                  v61 = *v60;
                  if ( v58 == *v60 )
                    goto LABEL_64;
                  v118 = v33;
                }
              }
            }
            LOBYTE(v138) = v62;
            v63 = *(_QWORD *)(a3 + 8);
            if ( v63 == *(_QWORD *)(a3 + 16) )
            {
LABEL_157:
              sub_10727E0(a3, (_BYTE *)v63, &v137);
              goto LABEL_69;
            }
            if ( v63 )
            {
LABEL_67:
              *(__m128i *)v63 = _mm_loadu_si128(&v137);
              *(_QWORD *)(v63 + 16) = v138;
              v63 = *(_QWORD *)(a3 + 8);
            }
LABEL_68:
            *(_QWORD *)(a3 + 8) = v63 + 24;
LABEL_69:
            if ( v52 == ++v51 )
            {
              v5 = v121;
              goto LABEL_92;
            }
          }
        }
      }
      if ( !*(_QWORD *)v38 )
      {
        if ( (*(_BYTE *)(v38 + 9) & 0x70) != 0x20 || v39 < 0 )
          goto LABEL_46;
        *(_BYTE *)(v38 + 8) |= 8u;
        v69 = sub_E807D0(*(_QWORD *)(v38 + 24));
        *(_QWORD *)v38 = v69;
        if ( !v69 )
        {
          v39 = *(_BYTE *)(v38 + 8);
          goto LABEL_46;
        }
      }
LABEL_42:
      if ( v132 == ++v37 )
        goto LABEL_56;
    }
  }
LABEL_92:
  v70 = *(const __m128i **)(a4 + 8);
  v71 = *(__m128i **)a4;
  if ( v70 != *(const __m128i **)a4 )
  {
    _BitScanReverse64(&v72, 0xAAAAAAAAAAAAAAABLL * (((char *)v70 - (char *)v71) >> 3));
    sub_1070800(*(_QWORD *)a4, *(__m128i **)(a4 + 8), 2LL * (int)(63 - (v72 ^ 0x3F)), v33, v34, v35);
    sub_1070BF0(v71, v70);
  }
  v73 = *(const __m128i **)(a5 + 8);
  v74 = *(__m128i **)a5;
  if ( v73 != *(const __m128i **)a5 )
  {
    _BitScanReverse64(&v75, 0xAAAAAAAAAAAAAAABLL * (((char *)v73 - (char *)v74) >> 3));
    sub_1070800(*(_QWORD *)a5, *(__m128i **)(a5 + 8), 2LL * (int)(63 - (v75 ^ 0x3F)), v33, v34, v35);
    sub_1070BF0(v74, v73);
  }
  v76 = (char **)a3;
  v77 = 0;
  v137.m128i_i64[1] = a4;
  v137.m128i_i64[0] = a3;
  v138 = a5;
  for ( j = &v137; ; v76 = (char **)j->m128i_i64[0] )
  {
    v79 = *v76;
    v80 = v76[1];
    if ( v80 != *v76 )
    {
      v81 = *v76;
      v82 = v77;
      do
      {
        v83 = *(_QWORD *)v81;
        v84 = v82;
        v81 += 24;
        ++v82;
        *(_DWORD *)(v83 + 16) = v84;
      }
      while ( v80 != v81 );
      v77 = v77 - 1431655765 * ((unsigned __int64)(v80 - 24 - v79) >> 3) + 1;
    }
    j = (__m128i *)((char *)j + 8);
    if ( &v139 == (char *)j )
      break;
  }
  v85 = v5 + 112;
  v86 = *(__int64 **)(a2 + 40);
  v87 = &v86[*(unsigned int *)(a2 + 48)];
  if ( v87 != v86 )
  {
    while ( 1 )
    {
      v88 = *(_DWORD *)(v5 + 136);
      v89 = *v86;
      if ( !v88 )
        break;
      v90 = *(_QWORD *)(v5 + 120);
      v91 = 1;
      v92 = 0;
      v93 = (v88 - 1) & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
      v94 = (_QWORD *)(v90 + 32LL * v93);
      v95 = *v94;
      if ( v89 == *v94 )
      {
LABEL_106:
        v96 = (_DWORD *)v94[2];
        v97 = (_DWORD *)v94[1];
        if ( v96 != v97 )
        {
          while ( 1 )
          {
            if ( !*(_QWORD *)v97 )
              goto LABEL_109;
            v98 = *(_DWORD *)(*(_QWORD *)v97 + 16LL);
            v99 = v97[3];
            if ( *(_DWORD *)(v5 + 2056) == 1 )
            {
              v97 += 4;
              *(v97 - 1) = v98 | v99 & 0xFF000000 | 0x8000000;
              if ( v96 == v97 )
                break;
            }
            else
            {
              v97[3] = (v98 << 8) | (unsigned __int8)v99 | 0x10;
LABEL_109:
              v97 += 4;
              if ( v96 == v97 )
                break;
            }
          }
        }
        if ( v87 == ++v86 )
          return sub_C7D6A0(v141, 16LL * v143, 8);
      }
      else
      {
        while ( v95 != -4096 )
        {
          if ( v95 == -8192 && !v92 )
            v92 = v94;
          v93 = (v88 - 1) & (v91 + v93);
          v94 = (_QWORD *)(v90 + 32LL * v93);
          v95 = *v94;
          if ( v89 == *v94 )
            goto LABEL_106;
          ++v91;
        }
        if ( !v92 )
          v92 = v94;
        v101 = *(_DWORD *)(v5 + 128);
        ++*(_QWORD *)(v5 + 112);
        v102 = v101 + 1;
        if ( 4 * (v101 + 1) < 3 * v88 )
        {
          if ( v88 - *(_DWORD *)(v5 + 132) - v102 <= v88 >> 3 )
          {
            v136 = v85;
            sub_1072980(v85, v88);
            v111 = *(_DWORD *)(v5 + 136);
            if ( !v111 )
            {
LABEL_199:
              ++*(_DWORD *)(v5 + 128);
              BUG();
            }
            v112 = v111 - 1;
            v113 = *(_QWORD *)(v5 + 120);
            v114 = 0;
            v115 = v112 & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
            v85 = v136;
            v116 = 1;
            v102 = *(_DWORD *)(v5 + 128) + 1;
            v92 = (_QWORD *)(v113 + 32LL * v115);
            v117 = *v92;
            if ( v89 != *v92 )
            {
              while ( v117 != -4096 )
              {
                if ( !v114 && v117 == -8192 )
                  v114 = v92;
                v115 = v112 & (v116 + v115);
                v92 = (_QWORD *)(v113 + 32LL * v115);
                v117 = *v92;
                if ( v89 == *v92 )
                  goto LABEL_125;
                ++v116;
              }
              if ( v114 )
                v92 = v114;
            }
          }
          goto LABEL_125;
        }
LABEL_150:
        v135 = v85;
        sub_1072980(v85, 2 * v88);
        v104 = *(_DWORD *)(v5 + 136);
        if ( !v104 )
          goto LABEL_199;
        v105 = v104 - 1;
        v106 = *(_QWORD *)(v5 + 120);
        v85 = v135;
        v107 = (v104 - 1) & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
        v102 = *(_DWORD *)(v5 + 128) + 1;
        v92 = (_QWORD *)(v106 + 32LL * v107);
        v108 = *v92;
        if ( v89 != *v92 )
        {
          v109 = 1;
          v110 = 0;
          while ( v108 != -4096 )
          {
            if ( !v110 && v108 == -8192 )
              v110 = v92;
            v107 = v105 & (v109 + v107);
            v92 = (_QWORD *)(v106 + 32LL * v107);
            v108 = *v92;
            if ( v89 == *v92 )
              goto LABEL_125;
            ++v109;
          }
          if ( v110 )
            v92 = v110;
        }
LABEL_125:
        *(_DWORD *)(v5 + 128) = v102;
        if ( *v92 != -4096 )
          --*(_DWORD *)(v5 + 132);
        ++v86;
        *v92 = v89;
        v92[1] = 0;
        v92[2] = 0;
        v92[3] = 0;
        if ( v87 == v86 )
          return sub_C7D6A0(v141, 16LL * v143, 8);
      }
    }
    ++*(_QWORD *)(v5 + 112);
    goto LABEL_150;
  }
  return sub_C7D6A0(v141, 16LL * v143, 8);
}
