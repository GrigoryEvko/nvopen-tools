// Function: sub_6BBC40
// Address: 0x6bbc40
//
__int64 __fastcall sub_6BBC40(const __m128i *a1, __int64 a2, _DWORD *a3, __int64 a4, __int64 *a5)
{
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdi
  char *v15; // rbx
  char *v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdx
  int v21; // eax
  char *v22; // r14
  __int64 v23; // rsi
  _DWORD *v25; // r12
  __int64 *v26; // r12
  __int64 n; // r15
  __int64 v28; // rdi
  __int64 v29; // r14
  __int64 v30; // rdx
  unsigned __int64 ii; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rdx
  __int64 v35; // rax
  unsigned __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // rcx
  int v39; // eax
  __int64 v40; // rax
  char *v41; // r12
  __int64 v42; // r8
  __int64 v43; // r8
  __int64 *v44; // r14
  __int64 i; // rax
  __int64 v46; // rbx
  __int64 *v47; // rsi
  __int64 v48; // rdx
  __int64 v49; // rbx
  __int64 v50; // rax
  char v51; // al
  unsigned int v52; // r12d
  __int64 j; // rax
  __int64 v54; // rdi
  __int64 v55; // rdx
  __int64 k; // rax
  __int64 m; // rdi
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // rdi
  int v62; // eax
  __int64 v63; // rbx
  __int64 v64; // rdx
  __int64 v65; // r8
  __int64 v66; // r9
  __int64 v67; // rax
  __int64 v68; // r12
  char *v69; // r12
  unsigned __int64 v70; // rbx
  char *v71; // rax
  char *v72; // rax
  char *v73; // rax
  __int64 v74; // r12
  _QWORD *v75; // rax
  __int64 v76; // r12
  __int64 v77; // rsi
  __int64 v78; // rdi
  int v79; // eax
  __int64 v80; // r8
  __int64 v81; // r9
  __int64 v82; // rcx
  __int64 v83; // rdx
  int v84; // eax
  unsigned int v85; // eax
  __int64 v86; // rdx
  __int64 v87; // rcx
  __int64 v88; // r8
  __int64 v89; // r9
  __int64 v90; // r8
  unsigned __int64 v91; // rax
  __int64 v92; // rdx
  __int64 v93; // rcx
  __int64 v94; // r8
  __int64 v95; // r9
  int v96; // ebx
  int v97; // eax
  __int64 v98; // rdx
  __int64 v99; // r12
  int v100; // eax
  __int64 v101; // rdx
  int v102; // r11d
  __int64 v103; // rax
  __int64 v104; // r14
  __m128i v105; // xmm1
  __m128i v106; // xmm2
  __m128i v107; // xmm3
  __m128i v108; // xmm4
  __m128i v109; // xmm5
  __m128i v110; // xmm6
  __m128i v111; // xmm7
  __m128i v112; // xmm0
  __int8 v113; // dl
  int v114; // r14d
  int v115; // eax
  const char *v116; // rsi
  size_t v117; // rax
  __int64 v118; // r8
  __int64 v119; // r9
  __int64 v120; // rdx
  __int64 v121; // rcx
  __int64 v122; // rdx
  __int64 v123; // rdx
  __int64 v124; // rcx
  __int64 v125; // r8
  __int64 v126; // r9
  __int64 v127; // rdx
  int v128; // r14d
  __int64 v129; // rdx
  __int64 v130; // rcx
  __int64 v131; // r8
  __int64 v132; // r9
  int v133; // eax
  int v134; // eax
  int v135; // edx
  __m128i v136; // xmm2
  __m128i v137; // xmm3
  __m128i v138; // xmm4
  __m128i v139; // xmm5
  __m128i v140; // xmm6
  __m128i v141; // xmm7
  __m128i v142; // xmm0
  __m128i v143; // xmm1
  __m128i v144; // xmm2
  __m128i v145; // xmm3
  __m128i v146; // xmm4
  __m128i v147; // xmm5
  size_t v148; // rax
  char v149; // dl
  char *v150; // rax
  int v151; // eax
  int v152; // eax
  __int64 v153; // [rsp+8h] [rbp-1F8h]
  unsigned __int64 v154; // [rsp+10h] [rbp-1F0h]
  __int64 v155; // [rsp+10h] [rbp-1F0h]
  __int64 v156; // [rsp+18h] [rbp-1E8h]
  unsigned int v157; // [rsp+20h] [rbp-1E0h]
  __int64 v158; // [rsp+20h] [rbp-1E0h]
  int v159; // [rsp+20h] [rbp-1E0h]
  int v160; // [rsp+20h] [rbp-1E0h]
  __int64 v161; // [rsp+28h] [rbp-1D8h]
  __int64 v162; // [rsp+30h] [rbp-1D0h]
  int v163; // [rsp+3Ch] [rbp-1C4h]
  unsigned __int16 v165; // [rsp+48h] [rbp-1B8h]
  __int64 v166; // [rsp+48h] [rbp-1B8h]
  char *v168; // [rsp+58h] [rbp-1A8h] BYREF
  __int64 v169; // [rsp+60h] [rbp-1A0h] BYREF
  __int64 v170; // [rsp+68h] [rbp-198h] BYREF
  char dest[32]; // [rsp+70h] [rbp-190h] BYREF
  __m128i v172; // [rsp+90h] [rbp-170h]
  __m128i v173; // [rsp+A0h] [rbp-160h]
  _OWORD v174[5]; // [rsp+B0h] [rbp-150h] BYREF
  __m128i v175; // [rsp+100h] [rbp-100h]
  __m128i v176; // [rsp+110h] [rbp-F0h]
  __m128i v177; // [rsp+120h] [rbp-E0h]
  __m128i v178; // [rsp+130h] [rbp-D0h]
  __m128i v179; // [rsp+140h] [rbp-C0h]
  __m128i v180; // [rsp+150h] [rbp-B0h]
  __m128i v181; // [rsp+160h] [rbp-A0h]
  __m128i v182; // [rsp+170h] [rbp-90h]
  __m128i v183; // [rsp+180h] [rbp-80h]
  __m128i v184; // [rsp+190h] [rbp-70h]
  __m128i v185; // [rsp+1A0h] [rbp-60h]
  __m128i v186; // [rsp+1B0h] [rbp-50h]
  __m128i v187; // [rsp+1C0h] [rbp-40h]

  v8 = *(_QWORD *)(a4 + 48);
  *a5 = 0;
  v168 = (char *)a2;
  v161 = v8;
  v9 = sub_6EB5C0(a1);
  v14 = *(unsigned int *)(a4 + 72);
  v162 = v9;
  if ( !(_DWORD)v14 )
  {
    v15 = v168;
    if ( v168 )
      goto LABEL_4;
LABEL_13:
    if ( (unsigned int)sub_6E5430(v14, a2, v10, v11, v12, v13) )
      sub_6851C0(0x66Du, a3);
    goto LABEL_15;
  }
  if ( a1[1].m128i_i8[0] != 1 )
  {
    v162 = 0;
    if ( (unsigned int)sub_6E5430(v14, a2, v10, v11, v12, v13) )
      sub_6851C0(0xEB8u, a3);
    return v162;
  }
  v15 = v168;
  if ( !v168 )
  {
    if ( (unsigned int)sub_6E5430(v14, a2, v10, v11, v12, v13) )
    {
      a2 = (__int64)a3;
      v14 = 3769;
      sub_6851C0(0xEB9u, a3);
    }
    goto LABEL_13;
  }
LABEL_4:
  v165 = *(_WORD *)(v9 + 176);
  v16 = v15;
  v170 = *(_QWORD *)sub_6E1A20(v15);
  v20 = *(unsigned int *)(a4 + 32);
  if ( (int)v20 <= 0 )
  {
    v22 = (char *)&v168;
  }
  else
  {
    v21 = 0;
    while ( 1 )
    {
      ++v21;
      v22 = v16;
      v16 = *(char **)v16;
      if ( v21 == (_DWORD)v20 )
        break;
      if ( !v16 )
      {
        v23 = *(unsigned int *)(a4 + 72);
        if ( !(_DWORD)v23 || (v20 = (unsigned int)(v20 - 1), (_DWORD)v20 != v21) )
        {
          if ( (unsigned int)sub_6E5430(0, v23, v20, a4, v18, v19) )
            sub_6851C0(0xA5u, a3);
LABEL_15:
          sub_6E6840(a1);
          return v162;
        }
        do
        {
          v41 = v15;
          v15 = *(char **)v15;
        }
        while ( v15 );
        a2 = 4;
        sub_6E7080(dest, 4);
        v16 = dest;
        *(_QWORD *)v41 = sub_6E3060(dest);
        goto LABEL_21;
      }
    }
    if ( !v16 )
      goto LABEL_21;
  }
  a2 = 1647;
  v25 = (_DWORD *)sub_6E1A20(v16);
  if ( (unsigned int)sub_6E53E0(5, 1647, v25) )
  {
    a2 = (__int64)v25;
    sub_684B30(0x66Fu, v25);
  }
  v16 = *(char **)v22;
  sub_6E1990(*(_QWORD *)v22);
  *(_QWORD *)v22 = 0;
LABEL_21:
  if ( v165 == 25222 )
  {
    v29 = 0;
    v156 = 0;
    n = sub_72CBE0(v16, a2, v20, v17, v18, v19);
  }
  else
  {
    v26 = (__int64 *)v168;
    sub_6E65B0(v168);
    if ( *(_DWORD *)(a4 + 36) == 1 )
      v156 = *(_QWORD *)(v26[3] + 8);
    else
      v156 = *(_QWORD *)(*(_QWORD *)(*v26 + 24) + 8LL);
    n = v156;
    if ( *(_BYTE *)(v156 + 140) == 12 )
    {
      do
        n = *(_QWORD *)(n + 160);
      while ( *(_BYTE *)(n + 140) == 12 );
    }
    else
    {
      n = v156;
    }
    v28 = n;
    v163 = sub_8D2E30(n);
    if ( !v163 )
    {
      v157 = 1;
      if ( !(unsigned int)sub_8D3D40(n) )
      {
        while ( 1 )
        {
          v51 = *(_BYTE *)(n + 140);
          if ( v51 != 12 )
            break;
          n = *(_QWORD *)(n + 160);
        }
        if ( v51 )
        {
          v52 = *(_DWORD *)(a4 + 56) == 0 ? 1645 : 852;
          if ( (unsigned int)sub_6E5430(v28, a2, v30, ii, v42, v33) )
            sub_6851C0(v52, &v170);
        }
        goto LABEL_15;
      }
LABEL_54:
      v166 = 0;
      v43 = *(unsigned int *)(a4 + 56);
      if ( (_DWORD)v43 )
      {
        v169 = sub_724DC0(v28, a2, v30, ii, v43, v33);
        v63 = sub_727560();
        sub_72BAF0(v169, *(_QWORD *)(n + 128), unk_4F06A51);
        sub_6E6A50(v169, dest);
        v166 = sub_6F7150(dest, dest, v64);
        *a5 = v166;
        sub_724E30(&v169);
        *(_QWORD *)(a1[9].m128i_i64[0] + 64) = v63;
        if ( !*(_DWORD *)(a4 + 72) )
          *(_BYTE *)(v63 + 32) = 10;
      }
      v44 = 0;
      if ( !v157 )
      {
        for ( i = *(_QWORD *)(v162 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        v44 = **(__int64 ***)(i + 168);
        if ( *(_DWORD *)(a4 + 56) )
          v44 = (__int64 *)*v44;
      }
      if ( !v26 )
      {
LABEL_82:
        if ( !*(_QWORD *)(a4 + 8) )
          *(_QWORD *)(a4 + 8) = n;
        return v162;
      }
      while ( 1 )
      {
        sub_6E65B0(v26);
        v46 = v26[3];
        v47 = 0;
        sub_6F69D0(v46 + 8, 0);
        if ( v157 )
          goto LABEL_69;
        if ( !(unsigned int)sub_8D2E30(*(_QWORD *)(v46 + 8)) && !(unsigned int)sub_8D2660(*(_QWORD *)(v46 + 8))
          || !(unsigned int)sub_8D2780(v44[1]) )
        {
          goto LABEL_65;
        }
        for ( j = *(_QWORD *)(v46 + 8); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        v54 = v44[1];
        v55 = *(_QWORD *)(j + 128);
        for ( k = v54; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
          ;
        if ( v55 == *(_QWORD *)(k + 128) )
        {
          v47 = (__int64 *)(v46 + 8);
          sub_6FC3F0(v54, v46 + 8, 1);
        }
        else
        {
LABEL_65:
          v48 = *(unsigned int *)(a4 + 56);
          if ( (_DWORD)v48 && *((_DWORD *)v44 + 9) > 1u && (unsigned int)sub_8D2E30(v44[1]) )
          {
            for ( m = *(_QWORD *)(v46 + 8); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
              ;
            if ( (unsigned int)sub_8DBE70(m) )
            {
LABEL_68:
              v163 = 1;
              goto LABEL_69;
            }
            if ( !(unsigned int)sub_8D2EB0(m) || (v61 = sub_8D46C0(m), v62 = sub_8D23B0(v61), v48 = v61, v62) )
            {
              sub_69A8C0(852, (_DWORD *)(v46 + 76), v48, v58, v59, v60);
              return v162;
            }
            while ( *(_BYTE *)(v48 + 140) == 12 )
              v48 = *(_QWORD *)(v48 + 160);
            if ( *(_QWORD *)(v48 + 128) != *(_QWORD *)(n + 128) )
            {
              if ( (unsigned int)sub_6E5430(v61, 0, v48, v58, v59, v60) )
                sub_6858F0(0xA7u, (_DWORD *)(v46 + 76), v156, *(_QWORD *)(v46 + 8));
              return v162;
            }
          }
        }
        if ( v163 )
          goto LABEL_68;
        v47 = v44;
        sub_843D70(v46 + 8, v44, 0, 167);
LABEL_69:
        v49 = sub_6F7150(v46 + 8, v47, v48);
        if ( *a5 )
          *(_QWORD *)(v166 + 16) = v49;
        else
          *a5 = v49;
        if ( !v157 )
          v44 = (__int64 *)*v44;
        v50 = *v26;
        if ( !*v26 )
          goto LABEL_82;
        if ( *(_BYTE *)(v50 + 8) == 3 )
        {
          v50 = sub_6BBB10(v26);
          if ( !v50 )
            goto LABEL_82;
        }
        v166 = v49;
        v26 = (__int64 *)v50;
      }
    }
    v29 = sub_8D46C0(n);
    for ( n = v29; *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
      ;
  }
  v28 = n;
  v163 = sub_8D3D40(n);
  if ( v163 )
  {
    v163 = 0;
    v26 = (__int64 *)v168;
    v157 = 1;
    goto LABEL_54;
  }
  v34 = *(unsigned __int8 *)(n + 140);
  if ( (_BYTE)v34 == 12 )
  {
    v35 = n;
    do
    {
      v35 = *(_QWORD *)(v35 + 160);
      v34 = *(unsigned __int8 *)(v35 + 140);
    }
    while ( (_BYTE)v34 == 12 );
  }
  if ( !(_BYTE)v34 )
  {
    sub_6E6000(n, a2, v34, ii, v32, v33);
    v157 = 1;
    goto LABEL_107;
  }
  v36 = v165;
  if ( (unsigned __int16)(v165 - 25154) <= 0x3Fu )
  {
    v34 = 0x8400000000000081LL;
    LOWORD(v36) = v165 - 25154;
    if ( _bittest64(&v34, v36) )
    {
      v28 = n;
      if ( !(unsigned int)sub_8D2780(n) )
      {
        if ( (unsigned int)sub_6E5430(n, a2, v34, v37, v32, v33) )
          sub_6851C0(0xEBBu, &v170);
        return v162;
      }
    }
  }
  v38 = a4;
  v39 = *(_DWORD *)(a4 + 72);
  if ( v39 )
  {
    if ( v165 != 25222 )
    {
      v38 = *(unsigned int *)(a4 + 56);
      if ( (_DWORD)v38 )
      {
LABEL_40:
        if ( !unk_4F04C50 || (v40 = *(_QWORD *)(unk_4F04C50 + 32LL)) == 0 || (*(_BYTE *)(v40 + 198) & 0x10) == 0 )
        {
          if ( (unsigned int)sub_6E5430(v28, a2, v34, v38, v32, v33) )
            sub_6851C0(0xEA9u, &v170);
          return v162;
        }
        goto LABEL_182;
      }
      v28 = n;
      if ( !(unsigned int)sub_8D2960(n) )
      {
        v28 = n;
        if ( !(unsigned int)sub_8D2E30(n) )
        {
          v28 = n;
          if ( !(unsigned int)sub_8D2AC0(n) )
          {
            sub_69A8C0(3745, &v170, v34, v38, v32, v33);
            return v162;
          }
        }
      }
      v39 = *(_DWORD *)(a4 + 72);
    }
    if ( !v39 )
      goto LABEL_182;
    goto LABEL_40;
  }
LABEL_182:
  if ( ((unsigned __int16)(v165 - 25191) <= 1u || ((v165 - 25183) & 0xFFFA) == 0) && (unsigned int)sub_8D2AC0(n) )
  {
    if ( (unsigned int)sub_6E5430(n, a2, v86, v87, v88, v89) )
      sub_6851C0(0xEBAu, &v170);
    return v162;
  }
  if ( v161
    && !*(_DWORD *)(a4 + 56)
    && !*(_DWORD *)(a4 + 72)
    && !(unsigned int)sub_8D2930(n)
    && !(unsigned int)sub_8D2E30(n) )
  {
    a2 = (__int64)&v170;
    v28 = 1645;
    sub_69A8C0(1645, &v170, v123, v124, v125, v126);
    v163 = 0;
    v157 = 1;
    goto LABEL_107;
  }
  v28 = n;
  if ( (unsigned int)sub_8DBE70(n) )
  {
    v157 = 1;
    v26 = (__int64 *)v168;
    goto LABEL_54;
  }
  v163 = *(_DWORD *)(a4 + 56);
  if ( v163 )
  {
    v28 = n;
    if ( !(unsigned int)sub_8D25A0(n) )
    {
      sub_69A8C0(852, &v170, v122, ii, v65, v66);
      return v162;
    }
    v157 = *(_DWORD *)(a4 + 72);
    if ( !v157 )
    {
LABEL_281:
      v163 = 1;
      goto LABEL_107;
    }
    if ( v165 != 25222 )
    {
      ii = *(_QWORD *)(n + 128);
      if ( ii > 0x10 || ((0x10116uLL >> ii) & 1) == 0 )
      {
        sub_69A8C0(3767, &v170, v122, ii, v65, v66);
        return v162;
      }
      v157 = 0;
      goto LABEL_281;
    }
    v157 = 0;
    v163 = 1;
LABEL_124:
    v73 = v168;
LABEL_121:
    ii = *((_QWORD *)v73 + 3) + 8LL;
    v154 = ii;
    v74 = *(_QWORD *)(*(_QWORD *)v73 + 24LL);
    v67 = 0;
    v68 = v74 + 8;
LABEL_109:
    v30 = v154;
    *(_QWORD *)&dest[8] = v67;
    v153 = v68;
    v69 = dest;
    v70 = v154;
    if ( v154 )
    {
LABEL_110:
      if ( *(_BYTE *)(v70 + 16) != 2 || *(_BYTE *)(*(_QWORD *)(v70 + 272) + 140LL) != 2 )
      {
        v157 = 1;
        if ( (unsigned int)sub_6E5430(v28, a2, v30, ii, v65, v66) )
          sub_6851C0(0xEABu, &v170);
      }
      v28 = v70 + 320;
      a2 = 1;
      if ( (unsigned int)sub_620EE0((_WORD *)(v70 + 320), 1, &v169) > 5 )
      {
        v157 = 1;
        if ( (unsigned int)sub_6E5430(v28, 1, v30, ii, v65, v66) )
        {
          a2 = (__int64)&v170;
          v28 = 3746;
          sub_6851C0(0xEA2u, &v170);
        }
      }
    }
    while ( 1 )
    {
      v69 += 8;
      if ( &dest[16] == v69 )
        break;
      v70 = *(_QWORD *)v69;
      if ( *(_QWORD *)v69 )
        goto LABEL_110;
    }
    if ( v153 )
    {
      if ( *(_BYTE *)(v153 + 16) != 2 || *(_BYTE *)(*(_QWORD *)(v153 + 272) + 140LL) != 2 )
      {
        v157 = 1;
        if ( (unsigned int)sub_6E5430(v28, a2, v30, ii, v65, v66) )
          sub_6851C0(0xEAAu, &v170);
      }
      v28 = v153 + 320;
      a2 = 1;
      if ( (unsigned int)sub_620EE0((_WORD *)(v153 + 320), 1, &v169) > 4 )
      {
        v157 = 1;
        if ( (unsigned int)sub_6E5430(v28, 1, v30, ii, v65, v66) )
        {
          a2 = (__int64)&v170;
          v28 = 3747;
          sub_6851C0(0xEA3u, &v170);
        }
      }
    }
    if ( *(_DWORD *)(a4 + 72) && unk_4D045E8 <= 0x3Bu && (unsigned int)sub_6E5430(v28, a2, v30, ii, v65, v66) )
    {
      a2 = (__int64)&v170;
      v28 = 3766;
      sub_6851C0(0xEB6u, &v170);
    }
    switch ( v165 )
    {
      case 0x6241u:
      case 0x6242u:
      case 0x6248u:
      case 0x6249u:
      case 0x6286u:
        v77 = 1;
        v78 = v154 + 320;
        v79 = sub_620EE0((_WORD *)(v154 + 320), 1, &v169);
        v82 = v165;
        v83 = (unsigned int)v165 - 25153;
        if ( (unsigned __int16)(v165 - 25153) <= 1u )
        {
          if ( (unsigned int)(v79 - 3) > 1 )
            goto LABEL_140;
          if ( (unsigned int)sub_6E5430(v78, 1, v83, v165, v80, v81) )
          {
            v77 = (__int64)&v170;
            v78 = 3756;
            sub_6851C0(0xEACu, &v170);
            v157 = 1;
            goto LABEL_140;
          }
        }
        else
        {
          v82 = v165;
          v83 = (unsigned int)v165 - 25160;
          if ( (unsigned __int16)(v165 - 25160) > 1u )
          {
            v84 = unk_4D045E8;
            if ( v165 == 25222 )
            {
LABEL_141:
              if ( (unsigned int)(v84 - 60) <= 9 && (unsigned int)sub_6E53E0(5, 3762, &v170) )
                sub_684B30(0xEB2u, &v170);
              v28 = v153 + 320;
              a2 = 1;
              if ( (unsigned int)sub_620EE0((_WORD *)(v153 + 320), 1, &v169) == 2 && unk_4D045E8 <= 0x59u )
              {
                if ( v165 == 25222 )
                {
                  a2 = 3763;
                  v28 = 5;
                  if ( (unsigned int)sub_6E53E0(5, 3763, &v170) )
                  {
                    a2 = (__int64)&v170;
                    v28 = 3763;
                    sub_684B30(0xEB3u, &v170);
                  }
                }
                else
                {
                  a2 = 3759;
                  v28 = 5;
                  if ( (unsigned int)sub_6E53E0(5, 3759, &v170) )
                  {
                    a2 = (__int64)&v170;
                    v28 = 3759;
                    sub_684B30(0xEAFu, &v170);
                  }
                }
              }
LABEL_147:
              switch ( v165 )
              {
                case 0x624Fu:
                case 0x6250u:
                case 0x6257u:
                case 0x6258u:
                  if ( *(_QWORD *)(n + 128) != 8 )
                    goto LABEL_148;
                  if ( *(_BYTE *)(n + 140) != 2 )
                    goto LABEL_148;
                  v30 = (__int64)byte_4B6DF90;
                  if ( !byte_4B6DF90[*(unsigned __int8 *)(n + 160)] )
                    goto LABEL_148;
                  if ( (unsigned int)sub_6E5430(v28, a2, byte_4B6DF90, ii, v65, v66) )
                    sub_6851C0(0xEA7u, &v170);
                  break;
                case 0x626Bu:
                case 0x626Cu:
                case 0x6273u:
                case 0x6274u:
                  if ( *(_BYTE *)(n + 140) != 3 )
                    goto LABEL_148;
                  if ( (unsigned int)sub_6E5430(v28, a2, v30, ii, v65, v66) )
                    sub_6851C0(0xEA8u, &v170);
                  break;
                case 0x627Bu:
                case 0x627Cu:
                  if ( *(_QWORD *)(n + 128) > 3u )
                    goto LABEL_148;
                  if ( (unsigned int)sub_6E5430(v28, a2, v30, ii, v65, v66) )
                    sub_6851C0(0xEA6u, &v170);
                  break;
                case 0x6280u:
                case 0x6281u:
                  if ( *(_QWORD *)(n + 128) > 1u )
                    goto LABEL_148;
                  if ( (unsigned int)sub_6E5430(v28, a2, v30, ii, v65, v66) )
                    sub_6851C0(0xEA5u, &v170);
                  break;
                default:
LABEL_148:
                  v33 = v157;
                  if ( !v157 )
                  {
                    v26 = (__int64 *)v168;
                    goto LABEL_54;
                  }
                  break;
              }
              return v162;
            }
LABEL_140:
            v84 = unk_4D045E8;
            if ( *(_QWORD *)(n + 128) == 16 && unk_4D045E8 <= 0x45u )
            {
              if ( (unsigned int)sub_6E5430(v78, v77, v83, v82, v80, v81) )
                sub_6851C0(0xEAEu, &v170);
              v157 = 1;
              v84 = unk_4D045E8;
            }
            goto LABEL_141;
          }
          v83 = (unsigned int)(v79 - 1);
          if ( (unsigned int)v83 > 1 && v79 != 4 )
            goto LABEL_140;
          if ( (unsigned int)sub_6E5430(v78, 1, v83, v165, v80, v81) )
          {
            v77 = (__int64)&v170;
            v78 = 3757;
            sub_6851C0(0xEADu, &v170);
            v157 = 1;
            goto LABEL_140;
          }
        }
        v157 = 1;
        goto LABEL_140;
      case 0x624Fu:
      case 0x6250u:
      case 0x6257u:
      case 0x6258u:
      case 0x625Fu:
      case 0x6260u:
      case 0x6263u:
      case 0x6264u:
      case 0x6267u:
      case 0x6268u:
      case 0x626Bu:
      case 0x626Cu:
      case 0x6273u:
      case 0x6274u:
      case 0x627Bu:
      case 0x627Cu:
      case 0x6280u:
      case 0x6281u:
        if ( (unsigned int)sub_620EE0((_WORD *)(v153 + 320), 1, &v169) == 2
          && unk_4D045E8 <= 0x59u
          && (unsigned int)sub_6E53E0(5, 3763, &v170) )
        {
          sub_684B30(0xEB3u, &v170);
        }
        a2 = 1;
        v28 = v154 + 320;
        sub_620EE0((_WORD *)(v154 + 320), 1, &v169);
        v85 = unk_4D045E8;
        v30 = (unsigned int)(unk_4D045E8 - 60);
        if ( (unsigned int)v30 <= 9 )
        {
          a2 = 3762;
          v28 = 5;
          if ( (unsigned int)sub_6E53E0(5, 3762, &v170) )
          {
            a2 = (__int64)&v170;
            v28 = 3762;
            sub_684B30(0xEB2u, &v170);
          }
          v85 = unk_4D045E8;
        }
        if ( v85 <= 0x59 )
        {
          ii = v165;
          v30 = (unsigned int)v165 - 25216;
          if ( (unsigned __int16)(v165 - 25216) <= 1u
            || (v30 = (unsigned int)v165 - 25211, (unsigned __int16)(v165 - 25211) <= 1u) )
          {
            if ( *(_QWORD *)(n + 128) == 16 )
            {
              if ( (unsigned int)sub_6E5430(v28, a2, v30, v165, v65, v66) )
              {
                a2 = (__int64)&v170;
                v28 = 3764;
                sub_6851C0(0xEB4u, &v170);
              }
              v157 = 1;
              v85 = unk_4D045E8;
            }
            if ( v85 <= 0x45 && (unsigned __int16)(v165 - 25216) <= 1u && *(_QWORD *)(n + 128) == 2 )
            {
              v157 = 1;
              if ( (unsigned int)sub_6E5430(v28, a2, v30, ii, v65, v66) )
              {
                a2 = (__int64)&v170;
                v28 = 3765;
                sub_6851C0(0xEB5u, &v170);
              }
            }
          }
        }
        goto LABEL_147;
      default:
        goto LABEL_147;
    }
  }
  if ( v161 )
  {
    if ( v165 != 25222 )
    {
      v91 = *(_QWORD *)(n + 128);
      if ( v91 > 0x10 || (v30 = 65814, !_bittest64(&v30, v91)) )
      {
        a2 = (__int64)&v170;
        v28 = 1646;
        sub_69A8C0(1646, &v170, v30, ii, v90, v33);
        v157 = 1;
        goto LABEL_107;
      }
    }
  }
  v155 = sub_727560();
  if ( !*(_DWORD *)(a4 + 40) )
  {
    if ( (*(_BYTE *)(v162 + 89) & 8) != 0 )
      v116 = *(const char **)(v162 + 24);
    else
      v116 = *(const char **)(v162 + 8);
    strcpy(dest, v116);
    switch ( v165 )
    {
      case 0x6250u:
        strcpy(dest, "__nv_atomic_fetch_add");
        goto LABEL_258;
      case 0x6258u:
        strcpy(dest, "__nv_atomic_fetch_sub");
        goto LABEL_258;
      case 0x6260u:
        strcpy(dest, "__nv_atomic_fetch_and");
        goto LABEL_258;
      case 0x6264u:
        strcpy(dest, "__nv_atomic_fetch_xor");
        goto LABEL_258;
      case 0x6268u:
        strcpy(dest, "__nv_atomic_fetch_or");
        goto LABEL_258;
      case 0x626Cu:
        strcpy(dest, "__nv_atomic_fetch_max");
        goto LABEL_258;
      case 0x6274u:
        strcpy(dest, "__nv_atomic_fetch_min");
LABEL_258:
        v117 = strlen(dest);
LABEL_259:
        v121 = *(_QWORD *)(n + 128);
        if ( (unsigned __int64)(v121 - 4) > 4 )
        {
          sub_69A8C0(3748, &v170, v121 - 4, v121, v118, v119);
          return v162;
        }
        break;
      default:
        v117 = strlen(dest);
        v120 = (unsigned __int16)(v165 - 25167);
        switch ( v165 )
        {
          case 0x624Fu:
          case 0x6250u:
          case 0x6257u:
          case 0x6258u:
          case 0x625Fu:
          case 0x6260u:
          case 0x6263u:
          case 0x6264u:
          case 0x6267u:
          case 0x6268u:
          case 0x626Bu:
          case 0x626Cu:
          case 0x6273u:
            goto LABEL_259;
          case 0x627Cu:
            if ( *(_QWORD *)(n + 128) > 3u )
              goto LABEL_253;
            sub_69A8C0(3750, &v170, v120, v165, v118, v119);
            return v162;
          case 0x6281u:
            if ( *(_QWORD *)(n + 128) <= 1u )
            {
              sub_69A8C0(3749, &v170, v120, v165, v118, v119);
              return v162;
            }
LABEL_253:
            if ( *(_DWORD *)(a4 + 68) )
            {
              dest[v117 - 2] = 0;
              v117 -= 2LL;
            }
            sprintf(&dest[v117], "_%u", *(_DWORD *)(n + 128));
            break;
          default:
            goto LABEL_252;
        }
        goto LABEL_256;
    }
LABEL_252:
    switch ( v165 )
    {
      case 0x624Fu:
      case 0x6250u:
      case 0x6257u:
      case 0x6258u:
      case 0x626Bu:
      case 0x626Cu:
      case 0x6273u:
      case 0x6274u:
        sprintf(&dest[v117], "_%u", *(_DWORD *)(n + 128));
        v148 = strlen(dest);
        v149 = *(_BYTE *)(n + 140);
        switch ( v149 )
        {
          case 2:
            v150 = &dest[v148];
            if ( byte_4B6DF90[*(unsigned __int8 *)(n + 160)] )
            {
              strcpy(v150, "_s");
              goto LABEL_256;
            }
            break;
          case 3:
            strcpy(&dest[v148], "_f");
            goto LABEL_256;
          case 6:
            v150 = &dest[v148];
            break;
          default:
LABEL_256:
            v104 = sub_68A160(dest);
            goto LABEL_228;
        }
        strcpy(v150, "_u");
        goto LABEL_256;
      case 0x6286u:
        goto LABEL_256;
      default:
        goto LABEL_253;
    }
  }
  if ( !*(_DWORD *)(a4 + 44) )
  {
    v102 = 0;
    v101 = 0;
    v99 = 0;
    v96 = 0;
    goto LABEL_223;
  }
  if ( (*(_BYTE *)(v29 + 140) & 0xFB) == 8 )
  {
    v28 = v29;
    a2 = dword_4F077C4 != 2;
    if ( (sub_8D4C10(v29, a2) & 8) != 0 )
    {
      a2 = 0;
      v96 = sub_72D2E0(v29, 0);
      v97 = sub_8D2780(n);
      v98 = n;
      if ( !v97 )
        v98 = sub_72BA30(unk_4F06A60);
      v28 = 5;
      v158 = v98;
      v99 = n;
      v100 = sub_72BA30(5);
      v101 = v158;
      v102 = v100;
LABEL_223:
      if ( v165 <= 0x75u )
      {
        if ( v165 > 0x65u )
        {
          switch ( v165 )
          {
            case 'f':
            case 'g':
              v159 = v102;
              v114 = sub_72D2E0(v99, 0);
              v115 = sub_72C390();
              v103 = sub_732700(v115, v96, v114, v99, v159, v159, 0, 0);
              goto LABEL_227;
            case 'h':
              v103 = sub_732700(v99, v96, v99, v102, 0, 0, 0, 0);
              goto LABEL_227;
            case 'i':
            case 'j':
            case 'n':
            case 'o':
            case 'p':
              v103 = sub_732700(v99, v96, v101, v102, 0, 0, 0, 0);
              goto LABEL_227;
            case 'q':
              v152 = sub_72CBE0(v28, a2, v101, v93, v94, v95);
              v103 = sub_732700(v152, v96, v99, 0, 0, 0, 0, 0);
              goto LABEL_227;
            case 's':
              v103 = sub_732700(v99, v96, v102, 0, 0, 0, 0, 0);
              goto LABEL_227;
            case 'u':
              v160 = v102;
              v151 = sub_72CBE0(v28, a2, v101, v93, v94, v95);
              v103 = sub_732700(v151, v96, v99, v160, 0, 0, 0, 0);
              goto LABEL_227;
            default:
              break;
          }
        }
LABEL_237:
        sub_721090(v28);
      }
      if ( v165 == 3495 )
      {
        v135 = 0;
      }
      else
      {
        if ( v165 > 0xDA7u )
        {
          if ( v165 == 15582 )
          {
            v128 = sub_72D2E0(n, 0);
            v133 = sub_72CBE0(n, 0, v129, v130, v131, v132);
            v103 = sub_732700(v133, n, v128, 0, 0, 0, 0, 0);
LABEL_227:
            v104 = sub_68A000(v162, v103);
LABEL_228:
            v105 = _mm_loadu_si128(a1 + 1);
            v106 = _mm_loadu_si128(a1 + 2);
            v107 = _mm_loadu_si128(a1 + 3);
            v108 = _mm_loadu_si128(a1 + 4);
            v109 = _mm_loadu_si128(a1 + 5);
            *(__m128i *)dest = _mm_loadu_si128(a1);
            v110 = _mm_loadu_si128(a1 + 6);
            v111 = _mm_loadu_si128(a1 + 7);
            *(__m128i *)&dest[16] = v105;
            v112 = _mm_loadu_si128(a1 + 8);
            v113 = a1[1].m128i_i8[0];
            v172 = v106;
            v173 = v107;
            v174[0] = v108;
            v174[1] = v109;
            v174[2] = v110;
            v174[3] = v111;
            v174[4] = v112;
            if ( v113 == 2 )
            {
              v136 = _mm_loadu_si128(a1 + 10);
              v137 = _mm_loadu_si128(a1 + 11);
              v138 = _mm_loadu_si128(a1 + 12);
              v139 = _mm_loadu_si128(a1 + 13);
              v175 = _mm_loadu_si128(a1 + 9);
              v140 = _mm_loadu_si128(a1 + 14);
              v141 = _mm_loadu_si128(a1 + 15);
              v176 = v136;
              v142 = _mm_loadu_si128(a1 + 16);
              v143 = _mm_loadu_si128(a1 + 17);
              v177 = v137;
              v144 = _mm_loadu_si128(a1 + 18);
              v145 = _mm_loadu_si128(a1 + 19);
              v178 = v138;
              v146 = _mm_loadu_si128(a1 + 20);
              v179 = v139;
              v147 = _mm_loadu_si128(a1 + 21);
              v180 = v140;
              v181 = v141;
              v182 = v142;
              v183 = v143;
              v184 = v144;
              v185 = v145;
              v186 = v146;
              v187 = v147;
            }
            else if ( v113 == 5 || v113 == 1 )
            {
              v175.m128i_i64[0] = a1[9].m128i_i64[0];
            }
            v28 = v104;
            a2 = (a1[1].m128i_i8[2] & 0x40) != 0;
            sub_6EAB60(v104, a2, 0, (unsigned int)v174 + 4, (unsigned int)v174 + 12, a1[5].m128i_i64[1], (__int64)a1);
            if ( a1[1].m128i_i8[0] )
            {
              v127 = a1->m128i_i64[0];
              for ( ii = *(unsigned __int8 *)(a1->m128i_i64[0] + 140);
                    (_BYTE)ii == 12;
                    ii = *(unsigned __int8 *)(v127 + 140) )
              {
                v127 = *(_QWORD *)(v127 + 160);
              }
              v157 = 1;
              if ( (_BYTE)ii )
              {
                *(_QWORD *)(a1[9].m128i_i64[0] + 64) = v155;
                if ( !*(_DWORD *)(a4 + 72) )
                {
                  if ( *(_DWORD *)(a4 + 68) )
                  {
                    *(_BYTE *)(v155 + 32) = 9;
                  }
                  else if ( v161 )
                  {
                    *(_BYTE *)(v155 + 32) = 8;
                  }
                }
                a2 = 0;
                v28 = (__int64)a1;
                sub_6F5FA0(a1, 0, 0, 1, v65, v66);
                v157 = 0;
                v162 = *(_QWORD *)(v104 + 88);
              }
            }
            else
            {
              v157 = 1;
            }
LABEL_107:
            switch ( v165 )
            {
              case 0x6241u:
              case 0x6248u:
              case 0x6249u:
              case 0x624Fu:
              case 0x6250u:
              case 0x6257u:
              case 0x6258u:
              case 0x625Fu:
              case 0x6260u:
              case 0x6263u:
              case 0x6264u:
              case 0x6267u:
              case 0x6268u:
              case 0x626Bu:
              case 0x626Cu:
              case 0x6273u:
              case 0x6274u:
              case 0x627Cu:
                v71 = v168;
                goto LABEL_119;
              case 0x6242u:
                v72 = v168;
                goto LABEL_120;
              case 0x627Bu:
                v71 = *(char **)v168;
LABEL_119:
                v72 = *(char **)v71;
LABEL_120:
                v73 = *(char **)v72;
                goto LABEL_121;
              case 0x6280u:
              case 0x6281u:
                v75 = ****(_QWORD *****)v168;
                ii = v75[3] + 8LL;
                v154 = ii;
                v76 = *(_QWORD *)(*(_QWORD *)*v75 + 24LL);
                v67 = *(_QWORD *)(*v75 + 24LL) + 8LL;
                v68 = v76 + 8;
                goto LABEL_109;
              case 0x6286u:
                goto LABEL_124;
              default:
                v154 = 0;
                v67 = 0;
                v68 = 0;
                goto LABEL_109;
            }
          }
          if ( v165 > 0x3CDEu || v165 != 3496 && v165 != 15581 )
            goto LABEL_237;
LABEL_288:
          v134 = sub_72D2E0(v29, 0);
          v103 = sub_732700(v29, v134, 0, 0, 0, 0, 0, 0);
          goto LABEL_227;
        }
        if ( v165 != 3484 )
        {
          if ( v165 == 3489 )
            goto LABEL_288;
          if ( v165 != 3431 )
            goto LABEL_237;
        }
        v135 = sub_72BA30(6);
      }
      v103 = sub_732700(v156, v156, v135, 0, 0, 0, 0, 0);
      *(_QWORD *)(a4 + 8) = v156;
      goto LABEL_227;
    }
  }
  if ( (unsigned int)sub_6E5430(v28, a2, v92, v93, v94, v95) )
    sub_6851C0(0xB38u, &v170);
  return v162;
}
