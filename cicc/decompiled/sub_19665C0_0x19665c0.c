// Function: sub_19665C0
// Address: 0x19665c0
//
__int64 __fastcall sub_19665C0(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        _QWORD *a6,
        __m128 a7,
        __m128i a8,
        __m128i a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 **a15,
        __int64 a16,
        __int64 *a17,
        char a18)
{
  int v19; // eax
  __int64 v20; // rdx
  _QWORD *v21; // rax
  _QWORD *i; // rdx
  __int64 v23; // r12
  __int64 *v24; // r13
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // rdx
  unsigned int v28; // ecx
  __int64 *v29; // rbx
  __int64 v30; // rdi
  __int64 v31; // r14
  __int64 v32; // rax
  unsigned __int64 v33; // r13
  __int64 *v34; // r14
  __int64 *v35; // rbx
  __int64 *k; // r15
  __int64 v37; // rsi
  __int64 *v38; // r13
  __int64 *v39; // rbx
  __int64 v40; // rsi
  __int64 v41; // rbx
  unsigned int v42; // r14d
  double v43; // xmm4_8
  double v44; // xmm5_8
  __int64 v45; // rdi
  __int64 v46; // rax
  __int64 v47; // rsi
  __int64 v48; // rdi
  unsigned int v49; // r9d
  __int64 *v50; // rdx
  __int64 v51; // r10
  _QWORD *v52; // rbx
  _QWORD *v53; // r12
  __int64 v54; // rax
  unsigned __int64 *v55; // rax
  unsigned __int64 *v56; // r13
  unsigned int v58; // ecx
  _QWORD *v59; // rdi
  unsigned int v60; // eax
  int v61; // eax
  unsigned __int64 v62; // rax
  __int64 v63; // rax
  int v64; // ebx
  __int64 v65; // r12
  _QWORD *v66; // rax
  __int64 v67; // rdx
  _QWORD *j; // rdx
  unsigned int v69; // eax
  unsigned int v70; // edx
  _QWORD *v71; // rbx
  _QWORD *n; // r12
  unsigned __int64 v73; // rdi
  int v74; // r8d
  int v75; // r9d
  __int64 v76; // rax
  __int64 v77; // rcx
  __int64 *v78; // r12
  __int64 *v79; // rbx
  __int64 *v80; // rbx
  unsigned int v81; // esi
  __int64 v82; // r9
  __int64 v83; // rcx
  _QWORD *v84; // rax
  _QWORD *v85; // rdx
  __int64 v86; // rdi
  __int64 v87; // rdx
  __int64 v88; // rsi
  __int64 v89; // rdi
  unsigned int v90; // ecx
  __int64 *v91; // rax
  __int64 v92; // r9
  __int64 v93; // rax
  int v94; // r8d
  int v95; // r8d
  __int64 v96; // r9
  int v97; // edx
  unsigned int v98; // ecx
  __int64 v99; // r11
  int v100; // eax
  __int64 *m; // rbx
  __int64 v102; // rax
  int v103; // r8d
  __int64 v104; // r9
  __int64 v105; // rax
  __int64 v106; // rbx
  char v107; // al
  unsigned __int64 *v108; // rax
  __int64 *v109; // r14
  _QWORD *v110; // r10
  int v111; // edx
  __int64 v112; // rcx
  _QWORD *v113; // rax
  __int64 v114; // r8
  __int64 v115; // r12
  _QWORD *v116; // r9
  unsigned int v117; // esi
  unsigned int v118; // ecx
  unsigned int v119; // edx
  unsigned int v120; // edi
  __int64 v121; // r8
  __int64 v122; // rax
  unsigned __int64 *v123; // r13
  unsigned __int64 *v124; // rbx
  unsigned __int64 v125; // rdi
  unsigned __int64 *v126; // rbx
  unsigned __int64 *v127; // r13
  unsigned __int64 v128; // rdi
  int v129; // edx
  _QWORD *v130; // rax
  int v131; // r8d
  _QWORD *v132; // rdi
  int v133; // ecx
  int v134; // r8d
  int v135; // r8d
  __int64 v136; // r9
  int v137; // esi
  unsigned int v138; // ebx
  _QWORD *v139; // rcx
  __int64 v140; // rdi
  __int64 *v141; // rbx
  __int64 *v142; // rbx
  __int64 *v143; // rbx
  _QWORD *v144; // rsi
  int v145; // edi
  int v146; // r11d
  int v147; // r10d
  __int64 v148; // rax
  int v149; // esi
  unsigned int v150; // edx
  _QWORD *v151; // rcx
  int v152; // edi
  int v153; // esi
  unsigned int v154; // edx
  int v155; // edi
  int v156; // edi
  _QWORD *v157; // rsi
  __int64 v158; // [rsp+30h] [rbp-2B0h]
  unsigned __int64 v159; // [rsp+38h] [rbp-2A8h]
  __int64 v160; // [rsp+48h] [rbp-298h]
  unsigned __int8 v161; // [rsp+53h] [rbp-28Dh]
  _QWORD *v162; // [rsp+58h] [rbp-288h]
  __int64 *v164; // [rsp+68h] [rbp-278h]
  __int64 v165; // [rsp+68h] [rbp-278h]
  __int64 v166; // [rsp+70h] [rbp-270h]
  __int64 *v168; // [rsp+78h] [rbp-268h]
  __int64 v169; // [rsp+78h] [rbp-268h]
  __int64 *v171; // [rsp+88h] [rbp-258h]
  __int64 v172; // [rsp+88h] [rbp-258h]
  char v173; // [rsp+88h] [rbp-258h]
  __int16 v174; // [rsp+90h] [rbp-250h] BYREF
  __int64 v175; // [rsp+98h] [rbp-248h]
  _QWORD *v176; // [rsp+A0h] [rbp-240h]
  __int64 v177; // [rsp+A8h] [rbp-238h]
  unsigned int v178; // [rsp+B0h] [rbp-230h]
  __int64 *v179; // [rsp+C0h] [rbp-220h] BYREF
  __int64 v180; // [rsp+C8h] [rbp-218h]
  _BYTE v181[64]; // [rsp+D0h] [rbp-210h] BYREF
  _BYTE *v182; // [rsp+110h] [rbp-1D0h] BYREF
  __int64 v183; // [rsp+118h] [rbp-1C8h]
  _BYTE v184[64]; // [rsp+120h] [rbp-1C0h] BYREF
  __int64 v185; // [rsp+160h] [rbp-180h] BYREF
  __int64 v186; // [rsp+168h] [rbp-178h]
  _QWORD *v187; // [rsp+170h] [rbp-170h] BYREF
  unsigned int v188; // [rsp+178h] [rbp-168h]
  _BYTE *v189; // [rsp+1B0h] [rbp-130h] BYREF
  __int64 v190; // [rsp+1B8h] [rbp-128h]
  _BYTE v191[64]; // [rsp+1C0h] [rbp-120h] BYREF
  __int64 *v192; // [rsp+200h] [rbp-E0h] BYREF
  __int64 v193; // [rsp+208h] [rbp-D8h]
  __int64 v194; // [rsp+210h] [rbp-D0h] BYREF
  int v195; // [rsp+218h] [rbp-C8h]
  __int64 v196; // [rsp+220h] [rbp-C0h]
  __int64 v197; // [rsp+228h] [rbp-B8h]
  __int64 v198; // [rsp+230h] [rbp-B0h]
  int v199; // [rsp+238h] [rbp-A8h]
  __int64 v200; // [rsp+240h] [rbp-A0h]
  __int64 v201; // [rsp+248h] [rbp-98h]
  unsigned __int64 *v202; // [rsp+250h] [rbp-90h]
  __int64 v203; // [rsp+258h] [rbp-88h]
  _BYTE v204[32]; // [rsp+260h] [rbp-80h] BYREF
  unsigned __int64 *v205; // [rsp+280h] [rbp-60h]
  __int64 v206; // [rsp+288h] [rbp-58h]
  _QWORD v207[10]; // [rsp+290h] [rbp-50h] BYREF

  ++*(_QWORD *)(a3 + 8);
  v19 = *(_DWORD *)(a3 + 24);
  v166 = a5;
  v162 = a6;
  *(_BYTE *)(a3 + 1) = 1;
  if ( !v19 )
  {
    if ( !*(_DWORD *)(a3 + 28) )
      goto LABEL_7;
    v20 = *(unsigned int *)(a3 + 32);
    if ( (unsigned int)v20 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a3 + 16));
      *(_QWORD *)(a3 + 16) = 0;
      *(_QWORD *)(a3 + 24) = 0;
      *(_DWORD *)(a3 + 32) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v20 = *(unsigned int *)(a3 + 32);
  v58 = 4 * v19;
  if ( (unsigned int)(4 * v19) < 0x40 )
    v58 = 64;
  if ( (unsigned int)v20 <= v58 )
  {
LABEL_4:
    v21 = *(_QWORD **)(a3 + 16);
    for ( i = &v21[3 * v20]; i != v21; *(v21 - 2) = -8 )
    {
      *v21 = -8;
      v21 += 3;
    }
    *(_QWORD *)(a3 + 24) = 0;
    goto LABEL_7;
  }
  v59 = *(_QWORD **)(a3 + 16);
  v60 = v19 - 1;
  if ( !v60 )
  {
    v65 = 3072;
    v64 = 128;
LABEL_66:
    j___libc_free_0(v59);
    *(_DWORD *)(a3 + 32) = v64;
    v66 = (_QWORD *)sub_22077B0(v65);
    v67 = *(unsigned int *)(a3 + 32);
    *(_QWORD *)(a3 + 24) = 0;
    *(_QWORD *)(a3 + 16) = v66;
    for ( j = &v66[3 * v67]; j != v66; v66 += 3 )
    {
      if ( v66 )
      {
        *v66 = -8;
        v66[1] = -8;
      }
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v60, v60);
  v61 = 1 << (33 - (v60 ^ 0x1F));
  if ( v61 < 64 )
    v61 = 64;
  if ( (_DWORD)v20 != v61 )
  {
    v62 = ((unsigned __int64)(4 * v61 / 3u + 1) >> 1)
        | (4 * v61 / 3u + 1)
        | ((((unsigned __int64)(4 * v61 / 3u + 1) >> 1) | (4 * v61 / 3u + 1)) >> 2);
    v63 = (((v62 | (v62 >> 4)) >> 8) | v62 | (v62 >> 4) | ((((v62 | (v62 >> 4)) >> 8) | v62 | (v62 >> 4)) >> 16)) + 1;
    v64 = v63;
    v65 = 24 * v63;
    goto LABEL_66;
  }
  *(_QWORD *)(a3 + 24) = 0;
  v130 = &v59[3 * v20];
  do
  {
    if ( v59 )
    {
      *v59 = -8;
      v59[1] = -8;
    }
    v59 += 3;
  }
  while ( v130 != v59 );
LABEL_7:
  v23 = 0;
  v192 = &v194;
  v193 = 0x400000000LL;
  v24 = (__int64 *)a2[1];
  v171 = (__int64 *)a2[2];
  if ( v24 != v171 )
  {
    while ( 1 )
    {
      v25 = *(unsigned int *)(a1 + 24);
      v26 = *v24;
      if ( !(_DWORD)v25 )
        goto LABEL_20;
      v27 = *(_QWORD *)(a1 + 8);
      LODWORD(a6) = 1;
      v28 = (v25 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v29 = (__int64 *)(v27 + 16LL * v28);
      v30 = *v29;
      if ( v26 != *v29 )
      {
        while ( v30 != -8 )
        {
          LODWORD(a5) = (_DWORD)a6 + 1;
          v28 = (v25 - 1) & ((_DWORD)a6 + v28);
          v29 = (__int64 *)(v27 + 16LL * v28);
          v30 = *v29;
          if ( v26 == *v29 )
            goto LABEL_14;
          LODWORD(a6) = (_DWORD)a6 + 1;
        }
        goto LABEL_20;
      }
LABEL_14:
      if ( v29 == (__int64 *)(v27 + 16 * v25) )
      {
LABEL_20:
        v32 = (unsigned int)v193;
        if ( (unsigned int)v193 >= HIDWORD(v193) )
        {
          sub_16CD150((__int64)&v192, &v194, 0, 8, a5, (int)a6);
          v32 = (unsigned int)v193;
        }
        ++v24;
        v192[v32] = v26;
        LODWORD(v193) = v193 + 1;
        if ( v171 == v24 )
        {
LABEL_23:
          if ( v23 )
            goto LABEL_24;
          break;
        }
      }
      else
      {
        v31 = v29[1];
        if ( v23 )
        {
          sub_135D320(v23, v29[1]);
          if ( v31 )
          {
            sub_12D5E00(v31);
            j_j___libc_free_0(v31, 72);
          }
        }
        else
        {
          v23 = v29[1];
        }
        *v29 = -16;
        ++v24;
        --*(_DWORD *)(a1 + 16);
        ++*(_DWORD *)(a1 + 20);
        if ( v171 == v24 )
          goto LABEL_23;
      }
    }
  }
  v93 = sub_22077B0(72);
  v23 = v93;
  if ( v93 )
  {
    *(_QWORD *)(v93 + 24) = 0;
    *(_QWORD *)(v93 + 32) = 0;
    *(_QWORD *)v93 = a3;
    *(_QWORD *)(v93 + 16) = v93 + 8;
    *(_QWORD *)(v93 + 40) = 0;
    *(_QWORD *)(v93 + 8) = (v93 + 8) | 4;
    *(_DWORD *)(v93 + 48) = 0;
    *(_DWORD *)(v93 + 56) = 0;
    *(_QWORD *)(v93 + 64) = 0;
  }
LABEL_24:
  v33 = (unsigned __int64)v192;
  if ( v192 != &v192[(unsigned int)v193] )
  {
    v172 = a1;
    v34 = &v192[(unsigned int)v193];
    do
    {
      v35 = *(__int64 **)(*(_QWORD *)v33 + 40LL);
      for ( k = *(__int64 **)(*(_QWORD *)v33 + 32LL); v35 != k; ++k )
      {
        v37 = *k;
        sub_135CE70(v23, v37);
      }
      v33 += 8LL;
    }
    while ( v34 != (__int64 *)v33 );
    a1 = v172;
  }
  v38 = (__int64 *)a2[4];
  v39 = (__int64 *)a2[5];
  while ( v39 != v38 )
  {
    v40 = *v38++;
    sub_135CE70(v23, v40);
  }
  if ( v192 != &v194 )
    _libc_free((unsigned __int64)v192);
  v174 = 0;
  v41 = sub_13FC520((__int64)a2);
  v175 = 0;
  v176 = 0;
  v177 = 0;
  v178 = 0;
  sub_1436EA0((__int64)&v174, (__int64)a2);
  v42 = sub_13FC370((__int64)a2);
  if ( (_BYTE)v42 )
  {
    v86 = 0;
    v87 = *(unsigned int *)(v166 + 48);
    if ( (_DWORD)v87 )
    {
      v88 = *(_QWORD *)(v166 + 32);
      v89 = *(_QWORD *)a2[4];
      v90 = (v87 - 1) & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
      v91 = (__int64 *)(v88 + 16LL * v90);
      v92 = *v91;
      if ( v89 == *v91 )
      {
LABEL_100:
        if ( v91 != (__int64 *)(v88 + 16 * v87) )
        {
          v86 = v91[1];
          goto LABEL_102;
        }
      }
      else
      {
        v100 = 1;
        while ( v92 != -8 )
        {
          v147 = v100 + 1;
          v148 = ((_DWORD)v87 - 1) & (v90 + v100);
          v90 = v148;
          v91 = (__int64 *)(v88 + 16 * v148);
          v92 = *v91;
          if ( v89 == *v91 )
            goto LABEL_100;
          v100 = v147;
        }
      }
      v86 = 0;
    }
LABEL_102:
    v42 = sub_1965DD0(
            v86,
            a3,
            a4,
            v166,
            (__int64)v162,
            a15,
            a7,
            a8,
            a9,
            a10,
            v43,
            v44,
            a13,
            a14,
            (__int64)a2,
            v23,
            (__int64)&v174,
            a17);
  }
  if ( !v41 )
    goto LABEL_41;
  v45 = 0;
  v46 = *(unsigned int *)(v166 + 48);
  if ( (_DWORD)v46 )
  {
    v47 = *(_QWORD *)(v166 + 32);
    v48 = *(_QWORD *)a2[4];
    v49 = (v46 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
    v50 = (__int64 *)(v47 + 16LL * v49);
    v51 = *v50;
    if ( v48 == *v50 )
    {
LABEL_38:
      if ( v50 != (__int64 *)(v47 + 16 * v46) )
      {
        v45 = v50[1];
        goto LABEL_40;
      }
    }
    else
    {
      v129 = 1;
      while ( v51 != -8 )
      {
        v146 = v129 + 1;
        v49 = (v46 - 1) & (v129 + v49);
        v50 = (__int64 *)(v47 + 16LL * v49);
        v51 = *v50;
        if ( v48 == *v50 )
          goto LABEL_38;
        v129 = v146;
      }
    }
    v45 = 0;
  }
LABEL_40:
  v42 |= sub_1961260(
           v45,
           a3,
           a4,
           v166,
           (__int64)v162,
           (__int64)a2,
           a7,
           *(double *)a8.m128i_i64,
           *(double *)a9.m128i_i64,
           a10,
           v43,
           v44,
           a13,
           a14,
           v23,
           (char *)&v174,
           a17,
           *(_BYTE *)(a1 + 32));
  v173 = byte_4FB0620;
  if ( byte_4FB0620 || !(unsigned __int8)sub_13FC370((__int64)a2) )
    goto LABEL_41;
  v179 = (__int64 *)v181;
  v180 = 0x800000000LL;
  sub_13FA0E0((__int64)a2, (__int64)&v179);
  v159 = (unsigned int)v180;
  v76 = 8LL * (unsigned int)v180;
  v164 = v179;
  v168 = &v179[(unsigned __int64)v76 / 8];
  v77 = v76 >> 3;
  if ( v76 >> 5 )
  {
    v160 = v23;
    v78 = v179;
    v79 = &v179[4 * (v76 >> 5)];
    while ( 1 )
    {
      if ( *(_BYTE *)(sub_157EBA0(*v78) + 16) == 34 )
      {
        v80 = v78;
        v23 = v160;
        goto LABEL_92;
      }
      if ( *(_BYTE *)(sub_157EBA0(v78[1]) + 16) == 34 )
      {
        v142 = v78;
        v23 = v160;
        v80 = v142 + 1;
        goto LABEL_92;
      }
      if ( *(_BYTE *)(sub_157EBA0(v78[2]) + 16) == 34 )
      {
        v141 = v78;
        v23 = v160;
        v80 = v141 + 2;
        goto LABEL_92;
      }
      if ( *(_BYTE *)(sub_157EBA0(v78[3]) + 16) == 34 )
        break;
      v78 += 4;
      if ( v78 == v79 )
      {
        v80 = v78;
        v23 = v160;
        v77 = v168 - v80;
        goto LABEL_117;
      }
    }
    v143 = v78;
    v23 = v160;
    v80 = v143 + 3;
    goto LABEL_92;
  }
  v80 = v179;
LABEL_117:
  if ( v77 == 2 )
    goto LABEL_199;
  if ( v77 == 3 )
  {
    if ( *(_BYTE *)(sub_157EBA0(*v80) + 16) == 34 )
      goto LABEL_92;
    ++v80;
LABEL_199:
    if ( *(_BYTE *)(sub_157EBA0(*v80) + 16) == 34 )
      goto LABEL_92;
    ++v80;
    goto LABEL_120;
  }
  if ( v77 != 1 )
    goto LABEL_121;
LABEL_120:
  if ( *(_BYTE *)(sub_157EBA0(*v80) + 16) != 34 )
    goto LABEL_121;
LABEL_92:
  if ( v168 == v80 )
  {
LABEL_121:
    v182 = v184;
    v183 = 0x800000000LL;
    if ( v159 > 8 )
    {
      sub_16CD150((__int64)&v182, v184, v159, 8, v74, v75);
      v164 = v179;
      v168 = &v179[(unsigned int)v180];
    }
    for ( m = v164; v168 != m; LODWORD(v183) = v183 + 1 )
    {
      v102 = sub_157EE30(*m);
      v104 = v102 - 24;
      if ( !v102 )
        v104 = 0;
      v105 = (unsigned int)v183;
      if ( (unsigned int)v183 >= HIDWORD(v183) )
      {
        v165 = v104;
        sub_16CD150((__int64)&v182, v184, 0, 8, v103, v104);
        v105 = (unsigned int)v183;
        v104 = v165;
      }
      ++m;
      *(_QWORD *)&v182[8 * v105] = v104;
    }
    v192 = 0;
    v202 = (unsigned __int64 *)v204;
    v203 = 0x400000000LL;
    v205 = v207;
    v193 = 0;
    v194 = 0;
    v195 = 0;
    v196 = 0;
    v197 = 0;
    v198 = 0;
    v199 = 0;
    v200 = 0;
    v201 = 0;
    v206 = 0;
    v207[0] = 0;
    v207[1] = 1;
    v106 = *(_QWORD *)(v23 + 16);
    v169 = v23 + 8;
    if ( v106 == v23 + 8 )
      goto LABEL_164;
    v158 = v23;
    v161 = v42;
    while ( 1 )
    {
      if ( !*(_QWORD *)(v106 + 32) )
      {
        v107 = *(_BYTE *)(v106 + 67);
        if ( (((unsigned __int8)v107 >> 4) & 2) != 0
          && (v107 & 0x40) == 0
          && v107 >= 0
          && sub_13FC1A0((__int64)a2, **(_QWORD **)(v106 + 16)) )
        {
          break;
        }
      }
LABEL_131:
      v106 = *(_QWORD *)(v106 + 8);
      if ( v169 == v106 )
      {
        v23 = v158;
        v42 = v161;
        if ( v173 )
          sub_1AE5120(a2, v166, a4, a16);
        v123 = v202;
        LOBYTE(v42) = v173 | v161;
        v124 = &v202[(unsigned int)v203];
        if ( v202 != v124 )
        {
          do
          {
            v125 = *v123++;
            _libc_free(v125);
          }
          while ( v124 != v123 );
        }
        v126 = v205;
        v127 = &v205[2 * (unsigned int)v206];
        if ( v205 != v127 )
        {
          do
          {
            v128 = *v126;
            v126 += 2;
            _libc_free(v128);
          }
          while ( v126 != v127 );
          v127 = v205;
        }
        if ( v127 != v207 )
          _libc_free((unsigned __int64)v127);
LABEL_164:
        if ( v202 != (unsigned __int64 *)v204 )
          _libc_free((unsigned __int64)v202);
        j___libc_free_0(v197);
        j___libc_free_0(v193);
        if ( v182 != v184 )
          _libc_free((unsigned __int64)v182);
        v164 = v179;
        goto LABEL_93;
      }
    }
    v108 = (unsigned __int64 *)&v187;
    v185 = 0;
    v186 = 1;
    do
      *v108++ = -8;
    while ( v108 != (unsigned __int64 *)&v189 );
    v189 = v191;
    v190 = 0x800000000LL;
    v109 = *(__int64 **)(v106 + 16);
    if ( !v109 )
    {
LABEL_178:
      v173 |= sub_195F380(
                (__int64)&v185,
                (__int64)&v179,
                (__int64)&v182,
                (__int64)&v192,
                a4,
                v166,
                v162,
                (__int64)a2,
                v158,
                (char *)&v174,
                a17);
      if ( v189 != v191 )
        _libc_free((unsigned __int64)v189);
      if ( (v186 & 1) == 0 )
        j___libc_free_0(v187);
      goto LABEL_131;
    }
    while ( 1 )
    {
      v115 = *v109;
      LODWORD(v116) = v186 & 1;
      if ( (v186 & 1) != 0 )
      {
        v110 = &v187;
        v111 = 7;
      }
      else
      {
        v117 = v188;
        v110 = v187;
        v111 = v188 - 1;
        if ( !v188 )
        {
          v118 = v186;
          ++v185;
          v113 = 0;
          v119 = ((unsigned int)v186 >> 1) + 1;
          goto LABEL_147;
        }
      }
      LODWORD(v112) = v111 & (((unsigned int)v115 >> 9) ^ ((unsigned int)v115 >> 4));
      v113 = &v110[(unsigned int)v112];
      v114 = *v113;
      if ( v115 != *v113 )
        break;
LABEL_143:
      v109 = (__int64 *)v109[2];
      if ( !v109 )
        goto LABEL_178;
    }
    v144 = 0;
    v145 = 1;
    while ( v114 != -8 )
    {
      if ( v114 == -16 && !v144 )
        v144 = v113;
      v112 = v111 & (unsigned int)(v112 + v145);
      v113 = &v110[v112];
      v114 = *v113;
      if ( v115 == *v113 )
        goto LABEL_143;
      ++v145;
    }
    v118 = v186;
    if ( v144 )
      v113 = v144;
    ++v185;
    v119 = ((unsigned int)v186 >> 1) + 1;
    if ( (_BYTE)v116 )
    {
      v120 = 24;
      v117 = 8;
    }
    else
    {
      v117 = v188;
LABEL_147:
      v120 = 3 * v117;
    }
    LODWORD(v121) = 4 * v119;
    if ( 4 * v119 >= v120 )
    {
      sub_184F0F0((__int64)&v185, 2 * v117);
      if ( (v186 & 1) != 0 )
      {
        v116 = &v187;
        v153 = 7;
      }
      else
      {
        v116 = v187;
        if ( !v188 )
        {
LABEL_266:
          LODWORD(v186) = (2 * ((unsigned int)v186 >> 1) + 2) | v186 & 1;
          BUG();
        }
        v153 = v188 - 1;
      }
      v118 = v186;
      v154 = v153 & (((unsigned int)v115 >> 9) ^ ((unsigned int)v115 >> 4));
      v113 = &v116[v154];
      v121 = *v113;
      if ( v115 == *v113 )
        goto LABEL_150;
      v151 = 0;
      v155 = 1;
      while ( v121 != -8 )
      {
        if ( v121 == -16 && !v151 )
          v151 = v113;
        v154 = v153 & (v155 + v154);
        v113 = &v116[v154];
        v121 = *v113;
        if ( v115 == *v113 )
          goto LABEL_236;
        ++v155;
      }
    }
    else
    {
      if ( v117 - HIDWORD(v186) - v119 > v117 >> 3 )
        goto LABEL_150;
      sub_184F0F0((__int64)&v185, v117);
      if ( (v186 & 1) != 0 )
      {
        v116 = &v187;
        v149 = 7;
      }
      else
      {
        v116 = v187;
        if ( !v188 )
          goto LABEL_266;
        v149 = v188 - 1;
      }
      v118 = v186;
      v150 = v149 & (((unsigned int)v115 >> 9) ^ ((unsigned int)v115 >> 4));
      v113 = &v116[v150];
      v121 = *v113;
      if ( v115 == *v113 )
        goto LABEL_150;
      v151 = 0;
      v152 = 1;
      while ( v121 != -8 )
      {
        if ( v121 == -16 && !v151 )
          v151 = v113;
        v150 = v149 & (v152 + v150);
        v113 = &v116[v150];
        v121 = *v113;
        if ( v115 == *v113 )
          goto LABEL_236;
        ++v152;
      }
    }
    if ( v151 )
      v113 = v151;
LABEL_236:
    v118 = v186;
LABEL_150:
    LODWORD(v186) = (2 * (v118 >> 1) + 2) | v118 & 1;
    if ( *v113 != -8 )
      --HIDWORD(v186);
    *v113 = v115;
    v122 = (unsigned int)v190;
    if ( (unsigned int)v190 >= HIDWORD(v190) )
    {
      sub_16CD150((__int64)&v189, v191, 0, 8, v121, (int)v116);
      v122 = (unsigned int)v190;
    }
    *(_QWORD *)&v189[8 * v122] = v115;
    LODWORD(v190) = v190 + 1;
    goto LABEL_143;
  }
LABEL_93:
  if ( v164 != (__int64 *)v181 )
    _libc_free((unsigned __int64)v164);
LABEL_41:
  if ( *a2 && !a18 )
  {
    v81 = *(_DWORD *)(a1 + 24);
    if ( v81 )
    {
      v82 = *(_QWORD *)(a1 + 8);
      LODWORD(v83) = (v81 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v84 = (_QWORD *)(v82 + 16LL * (unsigned int)v83);
      v85 = (_QWORD *)*v84;
      if ( (_QWORD *)*v84 == a2 )
      {
LABEL_97:
        v84[1] = v23;
        goto LABEL_45;
      }
      v131 = 1;
      v132 = 0;
      while ( v85 != (_QWORD *)-8LL )
      {
        if ( !v132 && v85 == (_QWORD *)-16LL )
          v132 = v84;
        v83 = (v81 - 1) & ((_DWORD)v83 + v131);
        v84 = (_QWORD *)(v82 + 16 * v83);
        v85 = (_QWORD *)*v84;
        if ( (_QWORD *)*v84 == a2 )
          goto LABEL_97;
        ++v131;
      }
      v133 = *(_DWORD *)(a1 + 16);
      if ( v132 )
        v84 = v132;
      ++*(_QWORD *)a1;
      v97 = v133 + 1;
      if ( 4 * (v133 + 1) < 3 * v81 )
      {
        if ( v81 - *(_DWORD *)(a1 + 20) - v97 > v81 >> 3 )
        {
LABEL_110:
          *(_DWORD *)(a1 + 16) = v97;
          if ( *v84 != -8 )
            --*(_DWORD *)(a1 + 20);
          v84[1] = 0;
          *v84 = a2;
          goto LABEL_97;
        }
        sub_19622F0(a1, v81);
        v134 = *(_DWORD *)(a1 + 24);
        if ( v134 )
        {
          v135 = v134 - 1;
          v136 = *(_QWORD *)(a1 + 8);
          v137 = 1;
          v138 = v135 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v97 = *(_DWORD *)(a1 + 16) + 1;
          v139 = 0;
          v84 = (_QWORD *)(v136 + 16LL * v138);
          v140 = *v84;
          if ( (_QWORD *)*v84 != a2 )
          {
            while ( v140 != -8 )
            {
              if ( !v139 && v140 == -16 )
                v139 = v84;
              v138 = v135 & (v137 + v138);
              v84 = (_QWORD *)(v136 + 16LL * v138);
              v140 = *v84;
              if ( (_QWORD *)*v84 == a2 )
                goto LABEL_110;
              ++v137;
            }
            if ( v139 )
              v84 = v139;
          }
          goto LABEL_110;
        }
LABEL_267:
        ++*(_DWORD *)(a1 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    sub_19622F0(a1, 2 * v81);
    v94 = *(_DWORD *)(a1 + 24);
    if ( v94 )
    {
      v95 = v94 - 1;
      v96 = *(_QWORD *)(a1 + 8);
      v97 = *(_DWORD *)(a1 + 16) + 1;
      v98 = v95 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v84 = (_QWORD *)(v96 + 16LL * v98);
      v99 = *v84;
      if ( (_QWORD *)*v84 != a2 )
      {
        v156 = 1;
        v157 = 0;
        while ( v99 != -8 )
        {
          if ( v99 == -16 && !v157 )
            v157 = v84;
          v98 = v95 & (v156 + v98);
          v84 = (_QWORD *)(v96 + 16LL * v98);
          v99 = *v84;
          if ( (_QWORD *)*v84 == a2 )
            goto LABEL_110;
          ++v156;
        }
        if ( v157 )
          v84 = v157;
      }
      goto LABEL_110;
    }
    goto LABEL_267;
  }
  if ( v23 )
  {
    sub_12D5E00(v23);
    j_j___libc_free_0(v23, 72);
  }
LABEL_45:
  if ( a16 )
  {
    if ( (_BYTE)v42 )
    {
      ++*(_QWORD *)(a16 + 656);
      if ( *(_QWORD *)(a16 + 672) )
      {
        v69 = 4 * *(_DWORD *)(a16 + 672);
        v70 = *(_DWORD *)(a16 + 680);
        if ( v69 < 0x40 )
          v69 = 64;
        if ( v69 < v70 )
        {
          sub_195EB50(a16 + 656);
        }
        else
        {
          v71 = *(_QWORD **)(a16 + 664);
          for ( n = &v71[5 * v70]; n != v71; v71 += 5 )
          {
            if ( *v71 != -8 )
            {
              if ( *v71 != -16 )
              {
                v73 = v71[1];
                if ( (_QWORD *)v73 != v71 + 3 )
                  _libc_free(v73);
              }
              *v71 = -8;
            }
          }
          *(_QWORD *)(a16 + 672) = 0;
        }
      }
    }
  }
  if ( v178 )
  {
    v52 = v176;
    v53 = &v176[2 * v178];
    do
    {
      if ( *v52 != -8 && *v52 != -16 )
      {
        v54 = v52[1];
        if ( (v54 & 4) != 0 )
        {
          v55 = (unsigned __int64 *)(v54 & 0xFFFFFFFFFFFFFFF8LL);
          v56 = v55;
          if ( v55 )
          {
            if ( (unsigned __int64 *)*v55 != v55 + 2 )
              _libc_free(*v55);
            j_j___libc_free_0(v56, 48);
          }
        }
      }
      v52 += 2;
    }
    while ( v53 != v52 );
  }
  j___libc_free_0(v176);
  return v42;
}
