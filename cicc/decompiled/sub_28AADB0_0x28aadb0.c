// Function: sub_28AADB0
// Address: 0x28aadb0
//
__int64 __fastcall sub_28AADB0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 a6,
        char a7,
        char a8,
        _QWORD **a9,
        __int64 a10)
{
  unsigned int v11; // r11d
  __int64 v16; // r12
  __m128i v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // r9
  unsigned __int8 v24; // r10
  __int64 v25; // rax
  int v26; // edx
  __int64 v27; // rsi
  int v28; // edx
  unsigned int v29; // ecx
  __int64 *v30; // rax
  __int64 v31; // r8
  __int64 v32; // r8
  unsigned int v33; // ecx
  __int64 *v34; // rax
  __int64 v35; // rsi
  unsigned __int8 v36; // r10
  _QWORD *v37; // rax
  __int64 v38; // rsi
  unsigned __int8 *v39; // rax
  __int64 v40; // rax
  unsigned int v41; // eax
  unsigned __int8 v42; // r10
  __int64 v43; // rsi
  __int64 v44; // r9
  unsigned __int8 v45; // r10
  unsigned __int8 v46; // r11
  unsigned __int64 v47; // rax
  __int64 v48; // r8
  __int64 v49; // rax
  signed __int64 v50; // rdx
  __int64 v51; // rax
  unsigned __int64 v52; // rax
  __m128i *v53; // rdi
  __int64 v54; // rdx
  char *v55; // rdx
  char v56; // cl
  __int64 v57; // r8
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // rax
  __int64 v61; // rcx
  int v62; // eax
  int v63; // eax
  unsigned __int8 *v64; // rax
  unsigned __int8 v65; // r11
  unsigned __int8 v66; // r10
  unsigned __int8 **v67; // rdx
  unsigned __int8 **v68; // rcx
  __int64 v69; // rdx
  unsigned __int8 **v70; // r12
  unsigned __int8 **v71; // rdx
  unsigned __int8 **v72; // rcx
  __int64 v73; // rsi
  unsigned __int64 v74; // rax
  __int64 v75; // r13
  _QWORD *v76; // rdi
  __int64 v77; // r14
  __int64 v78; // rax
  __int64 v79; // rdx
  unsigned __int64 v80; // rax
  char v81; // al
  unsigned __int8 v82; // r11
  unsigned __int8 v83; // r10
  unsigned __int8 **v84; // r13
  __int16 v85; // ax
  __int16 v86; // ax
  __int16 v87; // ax
  bool v88; // al
  unsigned __int8 **v89; // rdx
  __int64 v90; // rax
  unsigned __int8 *v91; // rax
  __int16 v92; // ax
  int v93; // edi
  int v94; // edi
  __int64 v95; // rax
  __int64 v96; // rcx
  unsigned __int8 **v97; // rsi
  __m128i *v98; // rdi
  _QWORD **v99; // rax
  __int64 v100; // r8
  _QWORD *v101; // rdi
  __int64 v102; // rsi
  char v103; // al
  unsigned __int8 v104; // r11
  unsigned __int8 v105; // r10
  unsigned int v106; // r12d
  char v107; // al
  __int64 v108; // rbx
  __int64 v109; // r14
  __int64 *v110; // rax
  __int64 v111; // rcx
  __int64 v112; // r8
  __int64 v113; // r9
  __int64 v114; // r12
  unsigned __int8 v115; // r11
  unsigned __int8 v116; // r15
  unsigned __int8 *v117; // rax
  unsigned __int8 *v118; // rax
  __int16 v119; // ax
  __int16 v120; // ax
  char v121; // al
  __int64 v122; // [rsp+0h] [rbp-190h]
  int v123; // [rsp+8h] [rbp-188h]
  __int16 v124; // [rsp+Ch] [rbp-184h]
  unsigned __int8 **v125; // [rsp+10h] [rbp-180h]
  __int64 v126; // [rsp+10h] [rbp-180h]
  unsigned __int8 **v127; // [rsp+20h] [rbp-170h]
  __int64 v128; // [rsp+20h] [rbp-170h]
  unsigned __int8 v129; // [rsp+28h] [rbp-168h]
  unsigned __int8 v130; // [rsp+28h] [rbp-168h]
  unsigned __int8 v131; // [rsp+28h] [rbp-168h]
  __int64 v132; // [rsp+28h] [rbp-168h]
  __int64 *v133; // [rsp+30h] [rbp-160h]
  unsigned __int8 v134; // [rsp+30h] [rbp-160h]
  __int64 v135; // [rsp+30h] [rbp-160h]
  __int64 v136; // [rsp+30h] [rbp-160h]
  unsigned __int8 v137; // [rsp+30h] [rbp-160h]
  unsigned __int8 v138; // [rsp+30h] [rbp-160h]
  __int64 v139; // [rsp+30h] [rbp-160h]
  char v140; // [rsp+38h] [rbp-158h]
  unsigned __int8 v141; // [rsp+38h] [rbp-158h]
  unsigned __int8 v142; // [rsp+38h] [rbp-158h]
  unsigned __int8 v143; // [rsp+38h] [rbp-158h]
  __int64 v144; // [rsp+38h] [rbp-158h]
  unsigned __int8 v145; // [rsp+38h] [rbp-158h]
  __int64 v146; // [rsp+38h] [rbp-158h]
  unsigned __int8 v147; // [rsp+38h] [rbp-158h]
  __int64 v148; // [rsp+38h] [rbp-158h]
  unsigned __int8 v149; // [rsp+38h] [rbp-158h]
  __int64 v150; // [rsp+38h] [rbp-158h]
  unsigned __int8 v151; // [rsp+38h] [rbp-158h]
  unsigned __int8 v152; // [rsp+38h] [rbp-158h]
  _QWORD *v153; // [rsp+38h] [rbp-158h]
  unsigned __int8 v154; // [rsp+38h] [rbp-158h]
  unsigned __int8 v155; // [rsp+38h] [rbp-158h]
  __int64 v156; // [rsp+40h] [rbp-150h]
  unsigned __int8 v157; // [rsp+40h] [rbp-150h]
  unsigned __int8 v158; // [rsp+40h] [rbp-150h]
  unsigned __int8 v159; // [rsp+40h] [rbp-150h]
  unsigned __int8 v160; // [rsp+40h] [rbp-150h]
  char v161; // [rsp+40h] [rbp-150h]
  unsigned __int8 v162; // [rsp+40h] [rbp-150h]
  unsigned __int8 v163; // [rsp+40h] [rbp-150h]
  __int64 v164; // [rsp+48h] [rbp-148h]
  unsigned __int8 v165; // [rsp+48h] [rbp-148h]
  int v166; // [rsp+48h] [rbp-148h]
  unsigned __int8 v167; // [rsp+48h] [rbp-148h]
  unsigned __int8 v168; // [rsp+48h] [rbp-148h]
  unsigned __int8 *v169; // [rsp+48h] [rbp-148h]
  unsigned __int8 v170; // [rsp+48h] [rbp-148h]
  unsigned __int8 v171; // [rsp+48h] [rbp-148h]
  unsigned __int8 **v172; // [rsp+48h] [rbp-148h]
  __int64 v173; // [rsp+48h] [rbp-148h]
  unsigned __int8 **v174; // [rsp+48h] [rbp-148h]
  unsigned __int8 **v175; // [rsp+48h] [rbp-148h]
  char v176; // [rsp+48h] [rbp-148h]
  __int64 v178; // [rsp+50h] [rbp-140h]
  __int64 v179; // [rsp+58h] [rbp-138h]
  unsigned __int8 *v180; // [rsp+58h] [rbp-138h]
  __int64 v181; // [rsp+58h] [rbp-138h]
  unsigned __int8 v182; // [rsp+58h] [rbp-138h]
  unsigned int v183; // [rsp+58h] [rbp-138h]
  char v184; // [rsp+67h] [rbp-129h] BYREF
  _QWORD *v185; // [rsp+68h] [rbp-128h] BYREF
  __m128i v186; // [rsp+70h] [rbp-120h] BYREF
  __int128 v187; // [rsp+80h] [rbp-110h]
  __int128 v188; // [rsp+90h] [rbp-100h]
  unsigned __int8 *v189[6]; // [rsp+A0h] [rbp-F0h] BYREF
  __m128i v190; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v191; // [rsp+E0h] [rbp-B0h]
  __int64 v192; // [rsp+E8h] [rbp-A8h]
  __int64 v193; // [rsp+F0h] [rbp-A0h]
  __int64 v194; // [rsp+F8h] [rbp-98h]
  char v195; // [rsp+100h] [rbp-90h]
  __m128i v196; // [rsp+110h] [rbp-80h] BYREF
  __m128i v197; // [rsp+120h] [rbp-70h] BYREF
  __m128i v198[6]; // [rsp+130h] [rbp-60h] BYREF

  if ( a8 )
    return 0;
  if ( *(_BYTE *)a5 != 60 )
    return 0;
  v179 = *(_QWORD *)(a5 - 32);
  if ( *(_BYTE *)v179 != 17 )
    return 0;
  v16 = a2;
  v164 = sub_B43CC0(a2);
  v156 = *(_QWORD *)(a5 + 72);
  v140 = sub_AE5020(v164, v156);
  v17.m128i_i64[0] = sub_9208B0(v164, v156);
  v196 = v17;
  if ( v17.m128i_i8[8] )
    return 0;
  v18 = *(_QWORD *)(v179 + 24);
  if ( *(_DWORD *)(v179 + 32) > 0x40u )
    v18 = **(_QWORD **)(v179 + 24);
  v196.m128i_i8[8] = 0;
  v196.m128i_i64[0] = v18 * ((((unsigned __int64)(v17.m128i_i64[0] + 7) >> 3) + (1LL << v140) - 1) >> v140 << v140);
  v180 = (unsigned __int8 *)sub_CA1930(&v196);
  if ( sub_CA1930(&a7) < (unsigned __int64)v180 )
    return 0;
  if ( !*(_QWORD *)(a10 + 16) )
    sub_4263D6(&a7, v18, v19);
  v141 = a6;
  v20 = (*(__int64 (**)(void))(a10 + 24))();
  v178 = v20;
  v21 = v20;
  if ( !v20 )
    return 0;
  v22 = *(_QWORD *)(v20 - 32);
  if ( v22 )
  {
    if ( !*(_BYTE *)v22
      && *(_QWORD *)(v21 + 80) == *(_QWORD *)(v22 + 24)
      && (*(_BYTE *)(v22 + 33) & 0x20) != 0
      && *(_DWORD *)(v22 + 36) == 211 )
    {
      return 0;
    }
  }
  if ( *(_QWORD *)(v178 + 40) != *(_QWORD *)(a3 + 40) )
    return 0;
  if ( *(_BYTE *)a3 == 62 )
  {
    sub_D66840(&v196, (_BYTE *)a3);
    v24 = v141;
    v186 = _mm_loadu_si128(&v196);
    v187 = (__int128)_mm_loadu_si128(&v197);
    v188 = (__int128)_mm_loadu_si128(v198);
  }
  else
  {
    sub_D67210(&v186, a3);
    v24 = v141;
  }
  v185 = 0;
  v25 = a1[5];
  v26 = *(_DWORD *)(v25 + 56);
  v27 = *(_QWORD *)(v25 + 40);
  if ( v26 )
  {
    v28 = v26 - 1;
    v29 = v28 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v30 = (__int64 *)(v27 + 16LL * v29);
    v31 = *v30;
    if ( a3 == *v30 )
    {
LABEL_20:
      v32 = v30[1];
    }
    else
    {
      v63 = 1;
      while ( v31 != -4096 )
      {
        v94 = v63 + 1;
        v95 = v28 & (v29 + v63);
        v29 = v95;
        v30 = (__int64 *)(v27 + 16 * v95);
        v31 = *v30;
        if ( a3 == *v30 )
          goto LABEL_20;
        v63 = v94;
      }
      v32 = 0;
    }
    v33 = v28 & (((unsigned int)v178 >> 9) ^ ((unsigned int)v178 >> 4));
    v34 = (__int64 *)(v27 + 16LL * v33);
    v23 = *v34;
    if ( v178 == *v34 )
    {
LABEL_22:
      v35 = v34[1];
    }
    else
    {
      v62 = 1;
      while ( v23 != -4096 )
      {
        v93 = v62 + 1;
        v33 = v28 & (v62 + v33);
        v34 = (__int64 *)(v27 + 16LL * v33);
        v23 = *v34;
        if ( v178 == *v34 )
          goto LABEL_22;
        v62 = v93;
      }
      v35 = 0;
    }
  }
  else
  {
    v32 = 0;
    v35 = 0;
  }
  v142 = v24;
  if ( sub_28A9CB0(a9, v35, v32, &v185, v32, v23, *(_OWORD *)&v186, v187, v188) )
    return 0;
  v36 = v142;
  if ( v185 )
  {
    v37 = (*((_BYTE *)v185 + 7) & 0x40) != 0 ? (_QWORD *)*(v185 - 1) : &v185[-4 * (*((_DWORD *)v185 + 1) & 0x7FFFFFF)];
    v38 = v37[4];
    if ( *(_BYTE *)v38 > 0x1Cu && *(_QWORD *)(v38 + 40) == *(_QWORD *)(v178 + 40) )
    {
      v88 = sub_B445A0(v178, v38);
      v36 = v142;
      if ( v88 )
        return 0;
    }
  }
  v143 = v36;
  v39 = sub_98ACB0((unsigned __int8 *)a4, 6u);
  v11 = sub_CF7600(v39, &v184);
  if ( !(_BYTE)v11 )
    return v11;
  v129 = v143;
  v133 = (__int64 *)a1[3];
  v144 = a1[2];
  v40 = sub_CA1930(&a7);
  v196.m128i_i32[2] = 64;
  v196.m128i_i64[0] = v40;
  LOBYTE(v41) = sub_D30550((unsigned __int8 *)a4, 0, (unsigned __int64 *)&v196, v164, v178, v144, v133, 0);
  v11 = v41;
  if ( !(_BYTE)v41 )
  {
    if ( v196.m128i_i32[2] <= 0x40u || !v196.m128i_i64[0] )
      return v11;
    j_j___libc_free_0_0(v196.m128i_u64[0]);
    return 0;
  }
  v42 = v129;
  if ( v196.m128i_i32[2] > 0x40u && v196.m128i_i64[0] )
  {
    v176 = v41;
    j_j___libc_free_0_0(v196.m128i_u64[0]);
    LOBYTE(v11) = v176;
    v42 = v129;
  }
  v43 = v178;
  v145 = v42;
  v165 = v11;
  if ( sub_28A9550((unsigned __int8 *)a4, v178, a3) )
    return 0;
  v45 = v145;
  v46 = v165;
  _BitScanReverse64(&v47, 1LL << *(_WORD *)(a5 + 2));
  LOWORD(v47) = v47 ^ 0x3F;
  v124 = 63 - v47;
  if ( v145 < (unsigned __int8)(63 - v47) && *(_BYTE *)a4 != 60 )
    return 0;
  v48 = *(_QWORD *)(a5 + 16);
  v196.m128i_i64[0] = (__int64)&v197;
  v196.m128i_i64[1] = 0x800000000LL;
  if ( !v48 )
    goto LABEL_79;
  v49 = v48;
  v50 = 0;
  do
  {
    v49 = *(_QWORD *)(v49 + 8);
    ++v50;
  }
  while ( v49 );
  if ( v50 > 8 )
  {
    v43 = (__int64)&v197;
    v130 = v145;
    v134 = v165;
    v146 = v48;
    v166 = v50;
    sub_C8D5F0((__int64)&v196, &v197, v50, 8u, v48, v44);
    LODWORD(v50) = v166;
    v48 = v146;
    v46 = v134;
    v45 = v130;
  }
  v51 = v196.m128i_i64[0] + 8LL * v196.m128i_u32[2];
  do
  {
    v51 += 8;
    *(_QWORD *)(v51 - 8) = *(_QWORD *)(v48 + 24);
    v48 = *(_QWORD *)(v48 + 8);
  }
  while ( v48 );
  v196.m128i_i32[2] += v50;
  LODWORD(v52) = v196.m128i_i32[2];
  if ( !v196.m128i_i32[2] )
    goto LABEL_79;
  v167 = v46;
  v147 = v45;
  do
  {
    v53 = (__m128i *)v196.m128i_i64[0];
    v54 = (unsigned int)v52;
    v52 = (unsigned int)(v52 - 1);
    v55 = *(char **)(v196.m128i_i64[0] + 8 * v54 - 8);
    v196.m128i_i32[2] = v52;
    v56 = *v55;
    if ( (unsigned __int8)*v55 > 0x1Cu )
    {
      if ( v56 == 79 )
      {
        v57 = *((_QWORD *)v55 + 2);
        v43 = v196.m128i_u32[3];
        if ( v57 )
        {
          v58 = *((_QWORD *)v55 + 2);
          v59 = 0;
          do
          {
            v58 = *(_QWORD *)(v58 + 8);
            ++v59;
          }
          while ( v58 );
          v44 = v59;
          if ( v196.m128i_u32[3] < v59 + v52 )
          {
            v43 = (__int64)&v197;
            v139 = v59;
            v132 = v57;
            sub_C8D5F0((__int64)&v196, &v197, v59 + v52, 8u, v57, v59);
            v44 = v139;
            v57 = v132;
            v60 = v196.m128i_i64[0] + 8LL * v196.m128i_u32[2];
          }
          else
          {
            v60 = v196.m128i_i64[0] + 8 * v52;
          }
          do
          {
            v60 += 8;
            *(_QWORD *)(v60 - 8) = *(_QWORD *)(v57 + 24);
            v57 = *(_QWORD *)(v57 + 8);
          }
          while ( v57 );
        }
        else
        {
          if ( v196.m128i_u32[3] < v52 )
          {
            v43 = (__int64)&v197;
            sub_C8D5F0((__int64)&v196, &v197, v52, 8u, 0, v44);
          }
          v44 = 0;
        }
        v196.m128i_i32[2] += v44;
        LODWORD(v52) = v196.m128i_i32[2];
        continue;
      }
      if ( v56 == 85 )
      {
        v61 = *((_QWORD *)v55 - 4);
        if ( v61 )
        {
          if ( !*(_BYTE *)v61 )
          {
            v43 = *((_QWORD *)v55 + 10);
            if ( *(_QWORD *)(v61 + 24) == v43
              && (*(_BYTE *)(v61 + 33) & 0x20) != 0
              && (unsigned int)(*(_DWORD *)(v61 + 36) - 210) <= 1 )
            {
              continue;
            }
          }
        }
      }
    }
    if ( v55 != (char *)v178 && (char *)v16 != v55 )
      goto LABEL_115;
  }
  while ( (_DWORD)v52 );
  v46 = v167;
  v45 = v147;
LABEL_79:
  v157 = v45;
  v168 = v46;
  v64 = sub_A7E990((unsigned __int8 *)v178);
  v65 = v168;
  v66 = v157;
  v125 = v67;
  v68 = (unsigned __int8 **)v64;
  v69 = ((char *)v67 - (char *)v64) >> 7;
  if ( v69 <= 0 )
    goto LABEL_131;
  v148 = v16;
  v70 = (unsigned __int8 **)v64;
  v127 = (unsigned __int8 **)&v64[128 * v69];
  v135 = a3;
  do
  {
    if ( (unsigned __int8 *)a5 == sub_BD3990(*v70, v43) )
    {
      v43 = ((__int64)v70 - (v178 - 32LL * (*(_DWORD *)(v178 + 4) & 0x7FFFFFF))) >> 5;
      if ( sub_B49EE0((unsigned __int8 *)v178, v43) )
      {
        v71 = v70;
        v65 = v168;
        v16 = v148;
        a3 = v135;
        v72 = v71;
        v66 = v157;
        goto LABEL_88;
      }
    }
    if ( (unsigned __int8 *)a5 == sub_BD3990(v70[4], v43) )
    {
      v84 = v70 + 4;
      v86 = sub_B49EE0(
              (unsigned __int8 *)v178,
              ((__int64)v70 - (v178 - 32LL * (*(_DWORD *)(v178 + 4) & 0x7FFFFFF)) + 32) >> 5);
      v43 = HIBYTE(v86);
      if ( v86 )
        goto LABEL_120;
    }
    if ( (unsigned __int8 *)a5 == sub_BD3990(v70[8], v43)
      && (v84 = v70 + 8,
          v85 = sub_B49EE0(
                  (unsigned __int8 *)v178,
                  ((__int64)v70 - (v178 - 32LL * (*(_DWORD *)(v178 + 4) & 0x7FFFFFF)) + 64) >> 5),
          v43 = HIBYTE(v85),
          v85)
      || (unsigned __int8 *)a5 == sub_BD3990(v70[12], v43)
      && (v84 = v70 + 12,
          v87 = sub_B49EE0(
                  (unsigned __int8 *)v178,
                  ((__int64)v70 - (v178 - 32LL * (*(_DWORD *)(v178 + 4) & 0x7FFFFFF)) + 96) >> 5),
          v43 = HIBYTE(v87),
          v87) )
    {
LABEL_120:
      v72 = v84;
      v65 = v168;
      v16 = v148;
      a3 = v135;
      v66 = v157;
      goto LABEL_88;
    }
    v70 += 16;
  }
  while ( v70 != v127 );
  v89 = v70;
  v65 = v168;
  v16 = v148;
  a3 = v135;
  v68 = v89;
  v66 = v157;
LABEL_131:
  v90 = (char *)v125 - (char *)v68;
  if ( (char *)v125 - (char *)v68 == 64 )
    goto LABEL_178;
  if ( v90 == 96 )
  {
    v154 = v66;
    v162 = v65;
    v174 = v68;
    v117 = sub_BD3990(*v68, v43);
    v72 = v174;
    v65 = v162;
    v66 = v154;
    if ( (unsigned __int8 *)a5 == v117 )
    {
      v43 = ((__int64)v174 - (v178 - 32LL * (*(_DWORD *)(v178 + 4) & 0x7FFFFFF))) >> 5;
      v119 = sub_B49EE0((unsigned __int8 *)v178, v43);
      v72 = v174;
      v65 = v162;
      v66 = v154;
      if ( v119 )
      {
LABEL_88:
        if ( v125 == v72 )
          goto LABEL_112;
        v149 = v66;
        v158 = v65;
        v169 = sub_98ACB0((unsigned __int8 *)a4, 6u);
        if ( !(unsigned __int8)sub_CF70D0(v169) )
          goto LABEL_114;
        v73 = 1;
        v170 = sub_D13FF0((__int64)v169, 1, v178, a1[3], 1, 0, 0);
        v65 = v158;
        v66 = v149;
        if ( v170 )
          goto LABEL_114;
        v74 = 0xBFFFFFFFFFFFFFFELL;
        if ( (unsigned __int64)v180 <= 0x3FFFFFFFFFFFFFFBLL )
          v74 = (unsigned __int64)v180;
        v126 = v74;
        v128 = *(_QWORD *)(v178 + 40) + 48LL;
        if ( *(_QWORD *)(v178 + 32) == v128 )
          goto LABEL_112;
        v150 = a3;
        v75 = *(_QWORD *)(v178 + 32);
        v136 = a4;
        v131 = v66;
        while ( 1 )
        {
          if ( !v75 )
            BUG();
          v77 = v75 - 24;
          if ( *(_BYTE *)(v75 - 24) != 85 )
            goto LABEL_95;
          v78 = *(_QWORD *)(v75 - 56);
          if ( v78 )
          {
            if ( !*(_BYTE *)v78
              && *(_QWORD *)(v78 + 24) == *(_QWORD *)(v75 + 56)
              && (*(_BYTE *)(v78 + 33) & 0x20) != 0
              && *(_DWORD *)(v78 + 36) == 210 )
            {
              break;
            }
          }
LABEL_96:
          if ( v16 != v77 )
          {
            v73 = v75 - 24;
            v190.m128i_i64[0] = a5;
            v191 = 0;
            v190.m128i_i64[1] = v126;
            v192 = 0;
            v76 = *a9;
            v193 = 0;
            v194 = 0;
            v195 = 1;
            if ( (unsigned __int8)sub_CF63E0(v76, (unsigned __int8 *)(v75 - 24), &v190, (__int64)(a9 + 1))
              || (unsigned int)*(unsigned __int8 *)(v75 - 24) - 30 <= 0xA )
            {
              goto LABEL_168;
            }
          }
          v75 = *(_QWORD *)(v75 + 8);
          if ( v128 == v75 )
          {
LABEL_111:
            v65 = v158;
            a3 = v150;
            a4 = v136;
            v66 = v131;
            goto LABEL_112;
          }
        }
        if ( (unsigned __int8 *)a5 == sub_BD3990(
                                        *(unsigned __int8 **)(v77 + 32 * (1LL - (*(_DWORD *)(v75 - 20) & 0x7FFFFFF))),
                                        v73) )
        {
          v79 = *(_QWORD *)(v77 - 32LL * (*(_DWORD *)(v75 - 20) & 0x7FFFFFF));
          if ( *(_DWORD *)(v79 + 32) > 0x40u )
          {
            v122 = *(_QWORD *)(v77 - 32LL * (*(_DWORD *)(v75 - 20) & 0x7FFFFFF));
            v123 = *(_DWORD *)(v79 + 32);
            if ( v123 - (unsigned int)sub_C444A0(v79 + 24) > 0x40 )
              goto LABEL_111;
            v80 = **(_QWORD **)(v122 + 24);
          }
          else
          {
            v80 = *(_QWORD *)(v79 + 24);
          }
          if ( (unsigned __int64)v180 <= v80 )
            goto LABEL_111;
        }
LABEL_95:
        if ( *(_BYTE *)(v75 - 24) == 30 )
          goto LABEL_111;
        goto LABEL_96;
      }
    }
    v68 = v72 + 4;
LABEL_178:
    v155 = v66;
    v163 = v65;
    v175 = v68;
    v118 = sub_BD3990(*v68, v43);
    v72 = v175;
    v65 = v163;
    v66 = v155;
    if ( (unsigned __int8 *)a5 != v118
      || (v43 = ((__int64)v175 - (v178 - 32LL * (*(_DWORD *)(v178 + 4) & 0x7FFFFFF))) >> 5,
          v120 = sub_B49EE0((unsigned __int8 *)v178, v43),
          v72 = v175,
          v65 = v163,
          v66 = v155,
          !v120) )
    {
      v68 = v72 + 4;
      goto LABEL_134;
    }
    goto LABEL_88;
  }
  if ( v90 == 32 )
  {
LABEL_134:
    v151 = v66;
    v160 = v65;
    v172 = v68;
    v91 = sub_BD3990(*v68, v43);
    v65 = v160;
    v66 = v151;
    if ( (unsigned __int8 *)a5 == v91 )
    {
      v92 = sub_B49EE0(
              (unsigned __int8 *)v178,
              ((__int64)v172 - (v178 - 32LL * (*(_DWORD *)(v178 + 4) & 0x7FFFFFF))) >> 5);
      v72 = v172;
      v65 = v160;
      v66 = v151;
      if ( v92 )
        goto LABEL_88;
    }
  }
LABEL_112:
  v159 = v66;
  v171 = v65;
  v81 = sub_B19DB0(a1[3], a4, v178);
  v82 = v171;
  v83 = v159;
  if ( v81 )
  {
    v161 = 0;
  }
  else
  {
    if ( *(_BYTE *)a4 != 63 )
      goto LABEL_114;
    if ( !(unsigned __int8)sub_B4DD90(a4) )
      goto LABEL_114;
    v121 = sub_B19DB0(a1[3], *(_QWORD *)(a4 - 32LL * (*(_DWORD *)(a4 + 4) & 0x7FFFFFF)), v178);
    v82 = v171;
    v83 = v159;
    v161 = v121;
    if ( !v121 )
      goto LABEL_114;
  }
  if ( (unsigned __int64)v180 > 0x3FFFFFFFFFFFFFFBLL )
    v180 = (unsigned __int8 *)0xBFFFFFFFFFFFFFFELL;
  v96 = 12;
  v189[0] = (unsigned __int8 *)a4;
  v97 = v189;
  memset(&v189[2], 0, 32);
  v98 = &v190;
  v189[1] = v180;
  v99 = a9;
  v100 = (__int64)(a9 + 1);
  while ( v96 )
  {
    v98->m128i_i32[0] = *(_DWORD *)v97;
    v97 = (unsigned __int8 **)((char *)v97 + 4);
    v98 = (__m128i *)((char *)v98 + 4);
    --v96;
  }
  v101 = *v99;
  v102 = v178;
  v137 = v83;
  v152 = v82;
  v181 = v100;
  v195 = 1;
  v103 = sub_CF63E0(v101, (unsigned __int8 *)v178, &v190, v100);
  v104 = v152;
  v105 = v137;
  if ( v103 )
  {
    v102 = v178;
    v107 = sub_CF7130((__int64)*a9, (unsigned __int8 *)v178, v189, a1[3], v181);
    v53 = (__m128i *)v196.m128i_i64[0];
    v104 = v152;
    v105 = v137;
    if ( v107 )
      goto LABEL_115;
  }
  if ( *(_QWORD *)(a4 + 8) != *(_QWORD *)(a5 + 8) )
    goto LABEL_114;
  v173 = v16;
  v106 = 0;
  v182 = v104;
  v138 = v105;
  while ( (unsigned int)sub_A17190((unsigned __int8 *)v178) > v106 )
  {
    if ( (unsigned __int8 *)a5 == sub_BD3990(
                                    *(unsigned __int8 **)(v178
                                                        + 32
                                                        * (v106 - (unsigned __int64)(*(_DWORD *)(v178 + 4) & 0x7FFFFFF))),
                                    v102)
      && *(_QWORD *)(*(_QWORD *)(v178 + 32 * (v106 - (unsigned __int64)(*(_DWORD *)(v178 + 4) & 0x7FFFFFF))) + 8LL) != *(_QWORD *)(a5 + 8) )
    {
      goto LABEL_114;
    }
    ++v106;
  }
  v114 = v173;
  v115 = v182;
  v170 = 0;
  v183 = 0;
  v153 = a1;
  v116 = v115;
  while ( (unsigned int)sub_A17190((unsigned __int8 *)v178) > v183 )
  {
    if ( (unsigned __int8 *)a5 == sub_BD3990(
                                    *(unsigned __int8 **)(v178
                                                        + 32
                                                        * (v183 - (unsigned __int64)(*(_DWORD *)(v178 + 4) & 0x7FFFFFF))),
                                    v102) )
    {
      v102 = a4;
      sub_AC2B30(v178 + 32 * (v183 - (unsigned __int64)(*(_DWORD *)(v178 + 4) & 0x7FFFFFF)), a4);
      v170 = v116;
    }
    ++v183;
  }
  if ( !v170 )
  {
LABEL_114:
    v53 = (__m128i *)v196.m128i_i64[0];
LABEL_115:
    v170 = 0;
    goto LABEL_116;
  }
  if ( v138 < (unsigned __int8)v124 )
    *(_WORD *)(a4 + 2) = v124 | *(_WORD *)(a4 + 2) & 0xFFC0;
  if ( v161 )
  {
    if ( *(_BYTE *)a4 != 63 )
      a4 = 0;
    sub_B444E0((_QWORD *)a4, v178 + 24, 0);
  }
  if ( v185 )
  {
    sub_B444E0(v185, v178 + 24, 0);
    v108 = v153[5];
    v109 = sub_28AACA0(v108, v178);
    v110 = (__int64 *)sub_28AACA0(v108, (__int64)v185);
    sub_D75380((__int64 *)v153[6], v110, v109, v111, v112, v113);
  }
  sub_F57040((_BYTE *)v178, v114);
  if ( v114 != a3 )
    sub_F57040((_BYTE *)v178, a3);
LABEL_168:
  v53 = (__m128i *)v196.m128i_i64[0];
LABEL_116:
  if ( v53 != &v197 )
    _libc_free((unsigned __int64)v53);
  return v170;
}
