// Function: sub_17879B0
// Address: 0x17879b0
//
__int64 __fastcall sub_17879B0(
        __m128i *a1,
        __int64 a2,
        double a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __m128i v11; // xmm2
  __m128 v12; // xmm0
  unsigned __int8 *v13; // rdi
  unsigned __int8 *v14; // rsi
  unsigned __int8 *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rbx
  __int64 v18; // r12
  __int64 v19; // r13
  _QWORD *v20; // rax
  double v21; // xmm4_8
  double v22; // xmm5_8
  __int64 v24; // rbx
  __int64 v25; // r13
  _QWORD *v26; // rax
  _BYTE *v27; // rcx
  unsigned __int8 v28; // al
  unsigned int v29; // ebx
  bool v30; // al
  __int64 **v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // rcx
  double v35; // xmm4_8
  double v36; // xmm5_8
  __int64 *v37; // rdi
  __int64 v38; // rax
  __int64 v39; // rsi
  __int64 v40; // r12
  __int64 v41; // r14
  __int64 v42; // rdx
  __int64 v43; // rcx
  char v44; // al
  unsigned int v45; // r9d
  __int64 v46; // rsi
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  unsigned int v49; // edx
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // rax
  __int64 v58; // rax
  unsigned int v59; // ebx
  __int64 *v60; // rax
  _BYTE *v61; // rsi
  __int64 *v62; // rax
  int v63; // eax
  int v64; // ebx
  __int64 v65; // rsi
  bool v66; // al
  _BYTE *v67; // r9
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // rdi
  __int64 v71; // rax
  __int64 v72; // rdx
  __int64 v73; // rcx
  __int64 v74; // rdx
  __int64 v75; // rbx
  __int64 v76; // rax
  __int64 v77; // rcx
  __int64 v78; // rcx
  __int64 **v79; // rdi
  __int64 v80; // rdx
  __int64 v81; // rcx
  int v82; // eax
  int v83; // eax
  __int64 v84; // rdx
  unsigned int v85; // r14d
  __int64 v86; // rax
  char v87; // dl
  unsigned int v88; // ebx
  __int64 v89; // rdx
  __int64 v90; // rcx
  bool v91; // bl
  __int64 v92; // rdx
  __int64 v93; // rcx
  __int64 v94; // rdx
  __int64 v95; // rcx
  _BYTE *v96; // rbx
  _BYTE *v97; // r12
  __int64 *v98; // r14
  double v99; // xmm4_8
  double v100; // xmm5_8
  char v101; // al
  __int64 v102; // r11
  __int64 v103; // rsi
  __int64 v104; // rdi
  unsigned __int8 *v105; // r10
  unsigned __int8 v106; // al
  __int64 v107; // rsi
  __int64 v108; // rdi
  __int64 *v109; // rax
  char v110; // al
  __int16 v111; // ax
  __int64 v112; // rax
  __int64 **v113; // rdi
  __int64 v114; // rax
  __int64 v115; // rdi
  __int64 *v116; // rax
  __int64 v117; // rdi
  __int64 *v118; // rax
  __int64 **v119; // rdi
  __int64 v120; // rax
  bool v121; // zf
  __int64 v122; // rdi
  __int64 *v123; // rax
  __int64 v124; // rdx
  __int64 v125; // r10
  __int64 v126; // rsi
  __int64 v127; // rdi
  unsigned __int8 *v128; // rax
  unsigned __int8 *v129; // rax
  __int64 *v130; // r10
  __int64 *v131; // r9
  __int64 v132; // rax
  __int64 v133; // rdx
  __int64 v134; // r10
  __int64 v135; // rsi
  __int64 v136; // rdi
  unsigned __int8 *v137; // rax
  __int64 v138; // rax
  __int64 v139; // r12
  _QWORD *v140; // rsi
  _QWORD *v141; // rdx
  __int64 v142; // rax
  __int64 v143; // rax
  __int64 v144; // rdi
  __int64 v145; // rdx
  __int64 v146; // rsi
  unsigned __int8 *v147; // rax
  __int64 **v148; // r15
  __int64 v149; // r14
  char v150; // al
  __int64 v151; // rax
  __int64 *v152; // r12
  __int64 *v153; // rsi
  __int64 *v154; // rdx
  __int64 v155; // rax
  __int64 v156; // rax
  __int64 v157; // rdi
  __int64 v158; // rdx
  __int64 v159; // rsi
  unsigned __int8 *v160; // rax
  __int64 **v161; // r15
  __int64 v162; // r14
  __int64 v163; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v164; // [rsp+18h] [rbp-C8h]
  __int64 v166; // [rsp+28h] [rbp-B8h]
  __int64 v167; // [rsp+28h] [rbp-B8h]
  _BYTE *v168; // [rsp+28h] [rbp-B8h]
  __int64 v169; // [rsp+30h] [rbp-B0h]
  __int64 v170; // [rsp+30h] [rbp-B0h]
  __int64 v171; // [rsp+30h] [rbp-B0h]
  unsigned int v172; // [rsp+30h] [rbp-B0h]
  int v173; // [rsp+30h] [rbp-B0h]
  __int64 v174; // [rsp+30h] [rbp-B0h]
  __int64 v175; // [rsp+30h] [rbp-B0h]
  __int64 v176; // [rsp+30h] [rbp-B0h]
  __int64 *v177; // [rsp+30h] [rbp-B0h]
  __int64 *v178; // [rsp+30h] [rbp-B0h]
  __int64 *v179; // [rsp+40h] [rbp-A0h] BYREF
  __int64 *v180; // [rsp+48h] [rbp-98h] BYREF
  __int64 v181; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v182; // [rsp+58h] [rbp-88h]
  const char *v183; // [rsp+60h] [rbp-80h] BYREF
  __int64 **v184; // [rsp+68h] [rbp-78h]
  __int16 v185; // [rsp+70h] [rbp-70h]
  __m128 v186; // [rsp+80h] [rbp-60h] BYREF
  __m128i v187; // [rsp+90h] [rbp-50h]
  __int64 v188; // [rsp+A0h] [rbp-40h]

  v11 = _mm_loadu_si128(a1 + 168);
  v12 = (__m128)_mm_loadu_si128(a1 + 167);
  v13 = *(unsigned __int8 **)(a2 - 48);
  v188 = a2;
  v14 = *(unsigned __int8 **)(a2 - 24);
  v186 = v12;
  v187 = v11;
  v15 = sub_13E06F0(v13, v14, &v186);
  if ( v15 )
  {
    v17 = *(_QWORD *)(a2 + 8);
    if ( v17 )
    {
      v18 = (__int64)v15;
      v19 = a1->m128i_i64[0];
      do
      {
        v20 = sub_1648700(v17);
        sub_170B990(v19, (__int64)v20);
        v17 = *(_QWORD *)(v17 + 8);
      }
      while ( v17 );
LABEL_5:
      if ( a2 == v18 )
        v18 = sub_1599EF0(*(__int64 ***)a2);
      sub_164D160(a2, v18, v12, *(double *)a4.m128i_i64, *(double *)v11.m128i_i64, a6, v21, v22, a9, a10);
      return a2;
    }
    return 0;
  }
  v169 = a2;
  if ( (unsigned __int8)sub_170D400(a1, a2, v16, (__m128i)v12, *(double *)a4.m128i_i64, v11) )
    return v169;
  v169 = (__int64)sub_1707490(
                    (__int64)a1,
                    (unsigned __int8 *)a2,
                    *(double *)v12.m128_u64,
                    *(double *)a4.m128i_i64,
                    *(double *)v11.m128i_i64);
  if ( v169 )
    return v169;
  v18 = (__int64)sub_1708300(a1, (unsigned __int8 *)a2, (__m128i)v12, a4, v11);
  if ( v18 )
  {
    v24 = *(_QWORD *)(a2 + 8);
    if ( v24 )
    {
      v25 = a1->m128i_i64[0];
      do
      {
        v26 = sub_1648700(v24);
        sub_170B990(v25, (__int64)v26);
        v24 = *(_QWORD *)(v24 + 8);
      }
      while ( v24 );
      goto LABEL_5;
    }
    return 0;
  }
  v27 = *(_BYTE **)(a2 - 24);
  v166 = *(_QWORD *)(a2 - 48);
  v28 = v27[16];
  v164 = (unsigned __int64)v27;
  if ( v28 == 13 )
  {
    v29 = *((_DWORD *)v27 + 8);
    if ( v29 <= 0x40 )
      v30 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v29) == *((_QWORD *)v27 + 3);
    else
      v30 = v29 == (unsigned int)sub_16A58F0((__int64)(v27 + 24));
    goto LABEL_20;
  }
  if ( *(_BYTE *)(*(_QWORD *)v27 + 8LL) != 16 || v28 > 0x10u )
    goto LABEL_24;
  v58 = sub_15A1020(v27, a2, *(_QWORD *)v27, (__int64)v27);
  if ( v58 && *(_BYTE *)(v58 + 16) == 13 )
  {
    v59 = *(_DWORD *)(v58 + 32);
    if ( v59 <= 0x40 )
      v30 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v59) == *(_QWORD *)(v58 + 24);
    else
      v30 = v59 == (unsigned int)sub_16A58F0(v58 + 24);
LABEL_20:
    if ( v30 )
      goto LABEL_21;
    goto LABEL_24;
  }
  v85 = 0;
  v173 = *(_QWORD *)(*(_QWORD *)v164 + 32LL);
  if ( !v173 )
  {
LABEL_21:
    v183 = sub_1649960(a2);
    v184 = v31;
    v187.m128i_i16[0] = 261;
    v186.m128_u64[0] = (unsigned __int64)&v183;
    v169 = sub_15FB530((__int64 *)v166, (__int64)&v186, 0, v32);
    if ( !sub_15F2380(a2) )
      return v169;
    goto LABEL_22;
  }
  while ( 1 )
  {
    v86 = sub_15A0A60(v164, v85);
    if ( !v86 )
      break;
    v87 = *(_BYTE *)(v86 + 16);
    if ( v87 != 9 )
    {
      if ( v87 != 13 )
        break;
      v88 = *(_DWORD *)(v86 + 32);
      if ( v88 <= 0x40 )
      {
        if ( *(_QWORD *)(v86 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v88) )
          break;
      }
      else if ( v88 != (unsigned int)sub_16A58F0(v86 + 24) )
      {
        break;
      }
    }
    if ( v173 == ++v85 )
      goto LABEL_21;
  }
LABEL_24:
  v186.m128_u64[0] = (unsigned __int64)&v180;
  v186.m128_u64[1] = (unsigned __int64)&v183;
  v187.m128i_i64[0] = (__int64)&v181;
  if ( !(unsigned __int8)sub_1781480(&v186, a2) )
    goto LABEL_31;
  v37 = (__int64 *)v181;
  if ( *(_BYTE *)(v181 + 16) != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v181 + 8LL) == 16 )
    {
      v71 = sub_15A1020((_BYTE *)v181, a2, v33, v34);
      if ( v71 )
      {
        if ( *(_BYTE *)(v71 + 16) == 13 )
        {
          v37 = (__int64 *)v181;
          goto LABEL_26;
        }
      }
    }
LABEL_31:
    v44 = *(_BYTE *)(a2 + 16);
    if ( v44 == 39 )
    {
      v60 = *(__int64 **)(a2 - 48);
      if ( !v60 )
        goto LABEL_34;
      v61 = *(_BYTE **)(a2 - 24);
      v180 = *(__int64 **)(a2 - 48);
      if ( v61[16] > 0x10u )
        goto LABEL_34;
    }
    else
    {
      if ( v44 != 5 )
        goto LABEL_34;
      if ( *(_WORD *)(a2 + 18) != 15 )
        goto LABEL_34;
      v84 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      v60 = *(__int64 **)(a2 - 24 * v84);
      if ( !v60 )
        goto LABEL_34;
      v180 = *(__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      v34 = 1 - v84;
      v33 = 3 * (1 - v84);
      v61 = *(_BYTE **)(a2 + 8 * v33);
      if ( !v61 )
        goto LABEL_34;
    }
    v181 = (__int64)v61;
    v62 = (__int64 *)sub_177F600(*v60, v61, v33, v34);
    if ( v62 )
    {
      v170 = (__int64)v62;
      v63 = sub_1643030(*v62);
      v187.m128i_i16[0] = 257;
      v64 = v63;
      v65 = (__int64)v180;
      v167 = v170;
      v169 = sub_15FB440(23, v180, v170, (__int64)&v186, 0);
      v66 = sub_15F2370(a2);
      v67 = (_BYTE *)v167;
      if ( v66 )
      {
        v65 = 1;
        sub_15F2310(v169, 1);
        v67 = (_BYTE *)v167;
      }
      v168 = v67;
      if ( !sub_15F2380(a2) )
        return v169;
      v70 = (__int64)(v168 + 24);
      if ( v168[16] != 13 )
      {
        if ( *(_BYTE *)(*(_QWORD *)v168 + 8LL) != 16 )
          return v169;
        v112 = sub_15A1020(v168, v65, v68, v69);
        if ( !v112 || *(_BYTE *)(v112 + 16) != 13 )
          return v169;
        v70 = v112 + 24;
      }
      if ( sub_13A38F0(v70, (_QWORD *)(unsigned int)(v64 - 1)) )
        return v169;
LABEL_22:
      sub_15F2330(v169, 1);
      return v169;
    }
LABEL_34:
    if ( *(_BYTE *)(v164 + 16) != 13 )
      goto LABEL_51;
    v45 = *(_DWORD *)(v164 + 32);
    v46 = *(_QWORD *)(v164 + 24);
    v47 = 1LL << ((unsigned __int8)v45 - 1);
    if ( v45 > 0x40 )
    {
      if ( (*(_QWORD *)(v46 + 8LL * ((v45 - 1) >> 6)) & v47) == 0 )
      {
        v182 = *(_DWORD *)(v164 + 32);
        sub_16A4FD0((__int64)&v181, (const void **)(v164 + 24));
        v45 = v182;
LABEL_40:
        v49 = *(_DWORD *)(v164 + 32);
        v47 = 1LL << ((unsigned __int8)v49 - 1);
        if ( v49 > 0x40 )
        {
          v50 = *(_QWORD *)(*(_QWORD *)(v164 + 24) + 8LL * ((v49 - 1) >> 6));
LABEL_42:
          if ( (v47 & v50) == 0 )
            goto LABEL_48;
          if ( v45 > 0x40 )
          {
            v172 = v45;
            v82 = sub_16A5940((__int64)&v181);
            v45 = v172;
            if ( v82 != 1 )
            {
LABEL_49:
              if ( v181 )
                j_j___libc_free_0_0(v181);
LABEL_51:
              v169 = sub_1713A90(
                       a1->m128i_i64,
                       (_BYTE *)a2,
                       v12,
                       *(double *)a4.m128i_i64,
                       *(double *)v11.m128i_i64,
                       a6,
                       v35,
                       v36,
                       a9,
                       a10);
              if ( v169 )
                return v169;
              if ( *(_BYTE *)(v164 + 16) <= 0x10u )
              {
                v54 = *(_QWORD *)(v166 + 8);
                if ( v54 )
                {
                  if ( !*(_QWORD *)(v54 + 8) )
                  {
                    v101 = *(_BYTE *)(v166 + 16);
                    if ( v101 == 35 )
                    {
                      v102 = *(_QWORD *)(v166 - 48);
                      if ( !v102 )
                        goto LABEL_55;
                      v103 = *(_QWORD *)(v166 - 24);
                      if ( *(_BYTE *)(v103 + 16) > 0x10u )
                        goto LABEL_55;
                    }
                    else
                    {
                      if ( v101 != 5 )
                        goto LABEL_55;
                      if ( *(_WORD *)(v166 + 18) != 11 )
                        goto LABEL_55;
                      v53 = v166;
                      v52 = *(_DWORD *)(v166 + 20) & 0xFFFFFFF;
                      v102 = *(_QWORD *)(v166 - 24 * v52);
                      if ( !v102 )
                        goto LABEL_55;
                      v103 = *(_QWORD *)(v166 + 24 * (1 - v52));
                      if ( !v103 )
                        goto LABEL_55;
                    }
                    v174 = v102;
                    v104 = a1->m128i_i64[1];
                    v187.m128i_i16[0] = 257;
                    v105 = sub_171D160(
                             v104,
                             v103,
                             v164,
                             (__int64 *)&v186,
                             0,
                             0,
                             *(double *)v12.m128_u64,
                             *(double *)a4.m128i_i64,
                             *(double *)v11.m128i_i64);
                    v106 = v105[16];
                    if ( v106 != 39 && (v106 != 5 || *((_WORD *)v105 + 9) != 15) )
                    {
                      v107 = v174;
                      v175 = (__int64)v105;
                      v108 = a1->m128i_i64[1];
                      v187.m128i_i16[0] = 257;
                      v185 = 257;
                      v109 = (__int64 *)sub_171D160(
                                          v108,
                                          v107,
                                          v164,
                                          (__int64 *)&v183,
                                          0,
                                          0,
                                          *(double *)v12.m128_u64,
                                          *(double *)a4.m128i_i64,
                                          *(double *)v11.m128i_i64);
                      return sub_15FB440(11, v109, v175, (__int64)&v186, 0);
                    }
                  }
                }
              }
LABEL_55:
              v186.m128_u64[1] = (unsigned __int64)&v179;
              if ( (unsigned __int8)sub_171ECC0((__int64)&v186, v166, v52, v53) && *(_BYTE *)(v164 + 16) <= 0x10u )
              {
                v187.m128i_i16[0] = 257;
                v57 = sub_15A2B90(
                        (__int64 *)v164,
                        0,
                        0,
                        v56,
                        *(double *)v12.m128_u64,
                        *(double *)a4.m128i_i64,
                        *(double *)v11.m128i_i64);
                return sub_15FB440(15, v179, v57, (__int64)&v186, 0);
              }
              v184 = &v179;
              if ( (unsigned __int8)sub_171ECC0((__int64)&v183, v166, v55, v56) )
              {
                v186.m128_u64[1] = (unsigned __int64)&v180;
                if ( (unsigned __int8)sub_171ECC0((__int64)&v186, v164, v72, v73) )
                {
                  v187.m128i_i16[0] = 257;
                  v169 = sub_15FB440(15, v179, (__int64)v180, (__int64)&v186, 0);
                  if ( !sub_15F2380(a2) || (*(_BYTE *)(v166 + 17) & 4) == 0 || (*(_BYTE *)(v164 + 17) & 4) == 0 )
                    return v169;
                  goto LABEL_22;
                }
              }
              if ( (unsigned __int8)(*(_BYTE *)(v166 + 16) - 41) > 1u )
              {
                v83 = *(unsigned __int8 *)(v164 + 16);
                if ( (unsigned __int8)v83 <= 0x17u || (unsigned int)(v83 - 35) > 0x11 )
                {
                  sub_1705480(
                    *(double *)v12.m128_u64,
                    *(double *)a4.m128i_i64,
                    *(double *)v11.m128i_i64,
                    (__int64)a1,
                    v166,
                    v72,
                    v73);
                  goto LABEL_90;
                }
                v74 = v164;
                v75 = v166;
              }
              else
              {
                v74 = v166;
                v75 = v164;
              }
              v171 = v74;
              v76 = sub_1705480(
                      *(double *)v12.m128_u64,
                      *(double *)a4.m128i_i64,
                      *(double *)v11.m128i_i64,
                      (__int64)a1,
                      v75,
                      v74,
                      v73);
              v77 = *(_QWORD *)(v171 + 8);
              if ( v77 && !*(_QWORD *)(v77 + 8) && ((v78 = *(_QWORD *)(v171 - 24)) != 0 && v78 == v75 || v76 == v78) )
              {
                v163 = *(_QWORD *)(v171 - 24);
                if ( (unsigned __int8)(*(_BYTE *)(v171 + 16) - 41) <= 1u )
                {
                  v98 = *(__int64 **)(v171 - 48);
                  if ( sub_15F23D0(v171) )
                  {
                    if ( v75 == v163 )
                    {
                      return sub_170E100(
                               a1->m128i_i64,
                               a2,
                               (__int64)v98,
                               v12,
                               *(double *)a4.m128i_i64,
                               *(double *)v11.m128i_i64,
                               a6,
                               v99,
                               v100,
                               a9,
                               a10);
                    }
                    else
                    {
                      v187.m128i_i16[0] = 257;
                      return sub_15FB530(v98, (__int64)&v186, 0, v163);
                    }
                  }
                  else
                  {
                    v121 = *(_BYTE *)(v171 + 16) == 41;
                    v122 = a1->m128i_i64[1];
                    v187.m128i_i16[0] = 257;
                    v123 = (__int64 *)sub_17066B0(
                                        v122,
                                        (unsigned int)!v121 + 20,
                                        (__int64)v98,
                                        v163,
                                        (__int64 *)&v186,
                                        0,
                                        *(double *)v12.m128_u64,
                                        *(double *)a4.m128i_i64,
                                        *(double *)v11.m128i_i64);
                    v187.m128i_i16[0] = 257;
                    if ( v75 == v163 )
                      return sub_15FB440(13, v98, (__int64)v123, (__int64)&v186, 0);
                    else
                      return sub_15FB440(13, v123, (__int64)v98, (__int64)&v186, 0);
                  }
                }
              }
LABEL_90:
              v79 = *(__int64 ***)a2;
              if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
                v79 = (__int64 **)*v79[2];
              if ( sub_1642F90((__int64)v79, 1) )
              {
                v187.m128i_i16[0] = 257;
                return sub_15FB440(26, (__int64 *)v166, v164, (__int64)&v186, 0);
              }
              v186.m128_u64[1] = (unsigned __int64)&v183;
              if ( (unsigned __int8)sub_175D710((__int64)&v186, v166, v80, v81) )
              {
                v187.m128i_i16[0] = 257;
                v169 = sub_15FB440(23, (__int64 *)v164, (__int64)v183, (__int64)&v186, 0);
                v91 = (*(_BYTE *)(v166 + 17) & 4) != 0;
              }
              else
              {
                v186.m128_u64[1] = (unsigned __int64)&v183;
                if ( !(unsigned __int8)sub_175D710((__int64)&v186, v164, v89, v90) )
                  goto LABEL_116;
                v187.m128i_i16[0] = 257;
                v169 = sub_15FB440(23, (__int64 *)v166, (__int64)v183, (__int64)&v186, 0);
                v91 = (*(_BYTE *)(v164 + 17) & 4) != 0;
              }
              if ( v169 )
              {
                if ( sub_15F2370(a2) )
                  sub_15F2310(v169, 1);
                if ( sub_15F2380(a2) && v91 )
                  goto LABEL_22;
                return v169;
              }
LABEL_116:
              v186.m128_u64[0] = (unsigned __int64)&v179;
              if ( (unsigned __int8)sub_1781630(&v186, v166) && sub_17287D0(*v179, 1) )
              {
                v119 = *(__int64 ***)a2;
                v187.m128i_i16[0] = 257;
                v120 = sub_15A0680((__int64)v119, 0, 0);
                return sub_14EDD70((__int64)v179, (_QWORD *)v164, v120, (__int64)&v186, 0, 0);
              }
              v186.m128_u64[0] = (unsigned __int64)&v179;
              if ( (unsigned __int8)sub_1781630(&v186, v164) && sub_17287D0(*v179, 1) )
              {
                v113 = *(__int64 ***)a2;
                v187.m128i_i16[0] = 257;
                v114 = sub_15A0680((__int64)v113, 0, 0);
                return sub_14EDD70((__int64)v179, (_QWORD *)v166, v114, (__int64)&v186, 0, 0);
              }
              v186.m128_u64[1] = (unsigned __int64)&v181;
              v186.m128_u64[0] = (unsigned __int64)&v179;
              if ( (unsigned __int8)sub_171EB90(&v186, v166, v92, v93) )
              {
                v96 = (_BYTE *)v181;
                if ( sub_13A38F0(v181, (_QWORD *)(unsigned int)(*(_DWORD *)(v181 + 8) - 1)) )
                {
                  v187.m128i_i16[0] = 257;
                  v115 = a1->m128i_i64[1];
                  v185 = 257;
                  v116 = sub_17807C0(
                           v115,
                           (__int64)v179,
                           (__int64)v96,
                           (__int64 *)&v183,
                           0,
                           *(double *)v12.m128_u64,
                           *(double *)a4.m128i_i64,
                           *(double *)v11.m128i_i64);
                  return sub_15FB440(26, v116, v164, (__int64)&v186, 0);
                }
              }
              v186.m128_u64[1] = (unsigned __int64)&v181;
              v186.m128_u64[0] = (unsigned __int64)&v179;
              if ( (unsigned __int8)sub_171EB90(&v186, v164, v94, v95) )
              {
                v97 = (_BYTE *)v181;
                if ( sub_13A38F0(v181, (_QWORD *)(unsigned int)(*(_DWORD *)(v181 + 8) - 1)) )
                {
                  v187.m128i_i16[0] = 257;
                  v117 = a1->m128i_i64[1];
                  v185 = 257;
                  v118 = sub_17807C0(
                           v117,
                           (__int64)v179,
                           (__int64)v97,
                           (__int64 *)&v183,
                           0,
                           *(double *)v12.m128_u64,
                           *(double *)a4.m128i_i64,
                           *(double *)v11.m128i_i64);
                  return sub_15FB440(26, v118, v166, (__int64)&v186, 0);
                }
              }
              if ( *(_BYTE *)(v166 + 16) != 62 )
              {
LABEL_125:
                if ( *(_BYTE *)(v166 + 16) != 61 )
                  goto LABEL_126;
                if ( *(_BYTE *)(v164 + 16) == 13 )
                {
                  v138 = *(_QWORD *)(v166 + 8);
                  if ( !v138 || *(_QWORD *)(v138 + 8) )
                    goto LABEL_126;
                  v139 = sub_15A43B0(v164, **(__int64 ****)(v166 - 24), 0);
                  if ( v164 == sub_15A3CB0(v139, *(__int64 ***)a2, 0)
                    && (unsigned int)sub_17803D0(a1->m128i_i64, *(_QWORD **)(v166 - 24), v139, a2) == 2 )
                  {
                    v145 = v139;
                    v144 = a1->m128i_i64[1];
                    v187.m128i_i16[0] = 259;
                    v186.m128_u64[0] = (unsigned __int64)"mulconv";
                    v146 = *(_QWORD *)(v166 - 24);
LABEL_208:
                    v147 = sub_171D160(
                             v144,
                             v146,
                             v145,
                             (__int64 *)&v186,
                             1,
                             0,
                             *(double *)v12.m128_u64,
                             *(double *)a4.m128i_i64,
                             *(double *)v11.m128i_i64);
                    v148 = *(__int64 ***)a2;
                    v187.m128i_i16[0] = 257;
                    v149 = (__int64)v147;
                    v169 = (__int64)sub_1648A60(56, 1u);
                    if ( v169 )
                      sub_15FC690(v169, v149, (__int64)v148, (__int64)&v186, 0);
                    return v169;
                  }
                }
                goto LABEL_200;
              }
              v150 = *(_BYTE *)(v164 + 16);
              if ( v150 == 13 )
              {
                v151 = *(_QWORD *)(v166 + 8);
                if ( !v151 || *(_QWORD *)(v151 + 8) )
                  goto LABEL_126;
                v152 = (__int64 *)sub_15A43B0(v164, **(__int64 ****)(v166 - 24), 0);
                if ( v164 == sub_15A4460((unsigned __int64)v152, *(__int64 ***)a2, 0)
                  && (unsigned int)sub_1780400(a1->m128i_i64, *(__int64 **)(v166 - 24), v152, a2) == 2 )
                {
                  v158 = (__int64)v152;
                  v157 = a1->m128i_i64[1];
                  v187.m128i_i16[0] = 259;
                  v186.m128_u64[0] = (unsigned __int64)"mulconv";
                  v159 = *(_QWORD *)(v166 - 24);
                  goto LABEL_233;
                }
                if ( *(_BYTE *)(v164 + 16) != 62 )
                  goto LABEL_125;
                v153 = *(__int64 **)(v166 - 24);
                v154 = *(__int64 **)(v164 - 24);
                if ( *v154 != *v153 )
                {
LABEL_221:
                  if ( *(_BYTE *)(v166 + 16) != 61 )
                    goto LABEL_126;
LABEL_200:
                  if ( *(_BYTE *)(v164 + 16) == 61 )
                  {
                    v140 = *(_QWORD **)(v166 - 24);
                    v141 = *(_QWORD **)(v164 - 24);
                    if ( *v141 == *v140
                      && ((v142 = *(_QWORD *)(v166 + 8)) != 0 && !*(_QWORD *)(v142 + 8)
                       || (v143 = *(_QWORD *)(v164 + 8)) != 0 && !*(_QWORD *)(v143 + 8))
                      && (unsigned int)sub_17803D0(a1->m128i_i64, v140, (__int64)v141, a2) == 2 )
                    {
                      v187.m128i_i16[0] = 259;
                      v144 = a1->m128i_i64[1];
                      v186.m128_u64[0] = (unsigned __int64)"mulconv";
                      v145 = *(_QWORD *)(v164 - 24);
                      v146 = *(_QWORD *)(v166 - 24);
                      goto LABEL_208;
                    }
                  }
LABEL_126:
                  if ( sub_15F2380(a2)
                    || (unsigned int)sub_1780400(a1->m128i_i64, (__int64 *)v166, (__int64 *)v164, a2) != 2 )
                  {
                    if ( sub_15F2370(a2) || (unsigned int)sub_17803D0(a1->m128i_i64, (_QWORD *)v166, v164, a2) != 2 )
                      return 0;
                  }
                  else
                  {
                    sub_15F2330(a2, 1);
                    if ( sub_15F2370(a2) || (unsigned int)sub_17803D0(a1->m128i_i64, (_QWORD *)v166, v164, a2) != 2 )
                      return a2;
                  }
                  sub_15F2310(a2, 1);
                  return a2;
                }
              }
              else
              {
                if ( v150 != 62 )
                  goto LABEL_126;
                v153 = *(__int64 **)(v166 - 24);
                v154 = *(__int64 **)(v164 - 24);
                if ( *v153 != *v154 )
                  goto LABEL_126;
              }
              v155 = *(_QWORD *)(v166 + 8);
              if ( v155 && !*(_QWORD *)(v155 + 8) || (v156 = *(_QWORD *)(v164 + 8)) != 0 && !*(_QWORD *)(v156 + 8) )
              {
                if ( (unsigned int)sub_1780400(a1->m128i_i64, v153, v154, a2) != 2 )
                  goto LABEL_125;
                v187.m128i_i16[0] = 259;
                v157 = a1->m128i_i64[1];
                v186.m128_u64[0] = (unsigned __int64)"mulconv";
                v158 = *(_QWORD *)(v164 - 24);
                v159 = *(_QWORD *)(v166 - 24);
LABEL_233:
                v160 = sub_171D160(
                         v157,
                         v159,
                         v158,
                         (__int64 *)&v186,
                         0,
                         1,
                         *(double *)v12.m128_u64,
                         *(double *)a4.m128i_i64,
                         *(double *)v11.m128i_i64);
                v161 = *(__int64 ***)a2;
                v187.m128i_i16[0] = 257;
                v162 = (__int64)v160;
                v169 = (__int64)sub_1648A60(56, 1u);
                if ( v169 )
                  sub_15FC810(v169, v162, (__int64)v161, (__int64)&v186, 0);
                return v169;
              }
              goto LABEL_221;
            }
          }
          else if ( !v181 || (v181 & (v181 - 1)) != 0 )
          {
            goto LABEL_51;
          }
          v51 = *(_QWORD *)(v166 + 8);
          if ( !v51 || *(_QWORD *)(v51 + 8) )
          {
LABEL_48:
            if ( v45 <= 0x40 )
              goto LABEL_51;
            goto LABEL_49;
          }
          v110 = *(_BYTE *)(v166 + 16);
          if ( v110 == 37 )
          {
            v134 = *(_QWORD *)(v166 - 48);
            if ( !v134 )
              goto LABEL_158;
            v135 = *(_QWORD *)(v166 - 24);
            if ( !v135 )
              goto LABEL_158;
          }
          else
          {
            if ( v110 != 5 )
            {
              if ( v110 == 35 )
              {
                v125 = *(_QWORD *)(v166 - 48);
                if ( v125 )
                {
                  v126 = *(_QWORD *)(v166 - 24);
                  if ( *(_BYTE *)(v126 + 16) == 13 )
                    goto LABEL_181;
                }
              }
              goto LABEL_158;
            }
            v111 = *(_WORD *)(v166 + 18);
            if ( v111 != 13 )
            {
              if ( v111 == 11 )
              {
                v124 = *(_DWORD *)(v166 + 20) & 0xFFFFFFF;
                v125 = *(_QWORD *)(v166 - 24 * v124);
                if ( v125 )
                {
                  v126 = *(_QWORD *)(v166 + 24 * (1 - v124));
                  if ( *(_BYTE *)(v126 + 16) == 13 )
                  {
LABEL_181:
                    v176 = v125;
                    v127 = a1->m128i_i64[1];
                    v186.m128_u64[0] = (unsigned __int64)"subc";
                    v187.m128i_i16[0] = 259;
                    v185 = 257;
                    v128 = sub_171CBD0(
                             v127,
                             v126,
                             (__int64 *)&v183,
                             0,
                             0,
                             *(double *)v12.m128_u64,
                             *(double *)a4.m128i_i64,
                             *(double *)v11.m128i_i64);
                    v129 = sub_171D0D0(
                             v127,
                             (__int64)v128,
                             v176,
                             (__int64 *)&v186,
                             0,
                             0,
                             *(double *)v12.m128_u64,
                             *(double *)a4.m128i_i64,
                             *(double *)v11.m128i_i64);
                    v130 = (__int64 *)v176;
                    v131 = (__int64 *)v129;
                    goto LABEL_182;
                  }
                }
              }
LABEL_158:
              v45 = v182;
              goto LABEL_48;
            }
            v133 = *(_DWORD *)(v166 + 20) & 0xFFFFFFF;
            v134 = *(_QWORD *)(v166 - 24 * v133);
            if ( !v134 )
              goto LABEL_158;
            v135 = *(_QWORD *)(v166 + 24 * (1 - v133));
            if ( !v135 )
              goto LABEL_158;
          }
          v178 = (__int64 *)v134;
          v136 = a1->m128i_i64[1];
          v186.m128_u64[0] = (unsigned __int64)"suba";
          v187.m128i_i16[0] = 259;
          v137 = sub_171D0D0(
                   v136,
                   v135,
                   v134,
                   (__int64 *)&v186,
                   0,
                   0,
                   *(double *)v12.m128_u64,
                   *(double *)a4.m128i_i64,
                   *(double *)v11.m128i_i64);
          v130 = v178;
          v131 = (__int64 *)v137;
LABEL_182:
          v177 = v131;
          if ( v131 )
          {
            v187.m128i_i16[0] = 257;
            v132 = sub_15A1070(*v130, (__int64)&v181);
            v169 = sub_15FB440(15, v177, v132, (__int64)&v186, 0);
            if ( v182 > 0x40 && v181 )
              j_j___libc_free_0_0(v181);
            return v169;
          }
          goto LABEL_158;
        }
LABEL_75:
        v50 = *(_QWORD *)(v164 + 24);
        goto LABEL_42;
      }
      v186.m128_i32[2] = *(_DWORD *)(v164 + 32);
      sub_16A4FD0((__int64)&v186, (const void **)(v164 + 24));
      LOBYTE(v45) = v186.m128_i8[8];
      if ( v186.m128_i32[2] > 0x40u )
      {
        sub_16A8F40((__int64 *)&v186);
        goto LABEL_39;
      }
      v48 = v186.m128_u64[0];
    }
    else
    {
      v48 = *(_QWORD *)(v164 + 24);
      if ( (v47 & v46) == 0 )
      {
        v182 = *(_DWORD *)(v164 + 32);
        v181 = v46;
        goto LABEL_75;
      }
      v186.m128_i32[2] = *(_DWORD *)(v164 + 32);
    }
    v186.m128_u64[0] = ~v48 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v45);
LABEL_39:
    sub_16A7400((__int64)&v186);
    v45 = v186.m128_u32[2];
    v182 = v186.m128_u32[2];
    v181 = v186.m128_u64[0];
    goto LABEL_40;
  }
LABEL_26:
  v38 = sub_15A2D50(
          v37,
          (__int64)v183,
          0,
          0,
          *(double *)v12.m128_u64,
          *(double *)a4.m128i_i64,
          *(double *)v11.m128i_i64);
  v39 = (__int64)v180;
  v187.m128i_i16[0] = 257;
  v40 = *(_QWORD *)(a2 - 48);
  v41 = v38;
  v169 = sub_15FB440(15, v180, v38, (__int64)&v186, 0);
  if ( sub_15F2370(a2) && sub_15F2370(v40) )
  {
    v39 = 1;
    sub_15F2310(v169, 1);
  }
  if ( sub_15F2380(a2) && sub_15F2380(v40) && (unsigned __int8)sub_1596730(v41, v39, v42, v43) )
    goto LABEL_22;
  return v169;
}
