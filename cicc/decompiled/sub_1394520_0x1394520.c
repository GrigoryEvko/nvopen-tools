// Function: sub_1394520
// Address: 0x1394520
//
__int64 __fastcall sub_1394520(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r14d
  __int64 v6; // rdx
  int v7; // edx
  __m128i *v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r9
  __int32 v12; // r8d
  __int32 v13; // esi
  __int64 *v14; // rbx
  __int64 *v15; // r14
  _QWORD *v16; // r13
  _QWORD *v17; // r12
  __int64 v18; // rdi
  __int64 *v20; // r13
  __int64 *v21; // rcx
  __int64 *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // r13
  __int64 *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // r13
  __int64 **v30; // rax
  __int64 *v31; // r12
  unsigned int v32; // edi
  __int64 *v33; // r14
  unsigned __int64 v34; // rdx
  unsigned __int64 v35; // rax
  int v36; // r13d
  unsigned int v37; // ebx
  int v38; // r10d
  unsigned int jj; // esi
  __int64 v40; // rdx
  unsigned int v41; // esi
  const __m128i *v42; // rsi
  const __m128i *v43; // rdi
  unsigned __int32 v44; // ecx
  unsigned __int32 v45; // eax
  const __m128i *v46; // rdx
  __int32 v47; // edx
  const __m128i *kk; // rax
  __int64 v49; // rsi
  __int64 v50; // rdx
  __int64 v51; // r12
  __int64 *v52; // rbx
  __int64 *v53; // rax
  __int64 v54; // rdx
  unsigned __int32 v55; // edi
  __int64 v56; // rcx
  int v57; // r9d
  unsigned __int64 v58; // rsi
  unsigned __int64 v59; // rsi
  unsigned int i; // eax
  __int64 v61; // rsi
  unsigned int v62; // eax
  __m128i *v63; // rsi
  signed __int64 v64; // r14
  __int64 v65; // r8
  const __m128i *v66; // rdi
  const __m128i *v67; // r9
  unsigned __int32 v68; // esi
  const __m128i *v69; // rax
  unsigned __int32 v70; // eax
  const __m128i *v71; // rdx
  __int32 v72; // edx
  int v73; // r11d
  __int64 v74; // r14
  unsigned int v75; // ebx
  __int64 v76; // rdx
  int v77; // r15d
  __int64 v78; // r8
  __int64 v79; // rsi
  unsigned __int32 v80; // r9d
  unsigned __int64 v81; // rdi
  unsigned __int64 v82; // rdi
  unsigned int k; // eax
  __int64 v84; // rdi
  unsigned int v85; // eax
  __m128i *v86; // rsi
  signed __int64 v87; // rcx
  __int64 v88; // rdi
  const __m128i *v89; // r8
  const __m128i *v90; // r9
  unsigned __int32 v91; // esi
  const __m128i *v92; // rax
  unsigned __int32 v93; // eax
  const __m128i *v94; // rdx
  __int32 v95; // edx
  __int64 v96; // r8
  int v97; // r10d
  unsigned __int64 v98; // rsi
  unsigned __int64 v99; // rsi
  unsigned int n; // eax
  __int64 v101; // rsi
  unsigned int v102; // eax
  const __m128i *v103; // r9
  const __m128i *v104; // rdi
  const __m128i *v105; // r9
  __int64 v106; // rcx
  unsigned __int32 v107; // esi
  const __m128i *v108; // rax
  unsigned __int32 v109; // eax
  const __m128i *v110; // rdx
  __int32 v111; // edx
  const __m128i *v112; // rsi
  unsigned __int32 v113; // ecx
  const __m128i *v114; // rax
  unsigned __int32 ii; // eax
  const __m128i *v116; // rdx
  __int32 v117; // edx
  __m128i *v118; // rsi
  __int64 v119; // rax
  int v120; // r10d
  unsigned __int64 v121; // rsi
  unsigned __int64 v122; // rsi
  unsigned int m; // eax
  __int64 v124; // rsi
  unsigned int v125; // eax
  const __m128i *v126; // rsi
  unsigned __int32 v127; // eax
  unsigned __int32 v128; // edx
  const __m128i *v129; // rcx
  __int32 v130; // ecx
  const __m128i *v131; // rdx
  int v132; // r10d
  unsigned __int64 v133; // rsi
  unsigned __int64 v134; // rsi
  unsigned int j; // eax
  __int64 v136; // rsi
  unsigned int v137; // eax
  signed __int64 v140; // [rsp+20h] [rbp-190h]
  int v141; // [rsp+20h] [rbp-190h]
  __int64 *v142; // [rsp+28h] [rbp-188h]
  unsigned int *v143; // [rsp+30h] [rbp-180h]
  int v144; // [rsp+40h] [rbp-170h]
  unsigned int v145; // [rsp+44h] [rbp-16Ch]
  __int64 *v146; // [rsp+48h] [rbp-168h]
  int v147; // [rsp+48h] [rbp-168h]
  int v148; // [rsp+48h] [rbp-168h]
  __int64 v149; // [rsp+48h] [rbp-168h]
  __int64 v150; // [rsp+50h] [rbp-160h]
  int v151; // [rsp+50h] [rbp-160h]
  unsigned __int32 v152; // [rsp+58h] [rbp-158h]
  __int64 v153; // [rsp+58h] [rbp-158h]
  unsigned int *v154; // [rsp+60h] [rbp-150h]
  __int64 v155; // [rsp+60h] [rbp-150h]
  __int64 *v156; // [rsp+68h] [rbp-148h]
  __int64 *v157; // [rsp+68h] [rbp-148h]
  int v158; // [rsp+70h] [rbp-140h]
  __int64 v159; // [rsp+70h] [rbp-140h]
  __int64 v160; // [rsp+78h] [rbp-138h]
  unsigned __int64 v161; // [rsp+78h] [rbp-138h]
  __m128i *v162; // [rsp+80h] [rbp-130h] BYREF
  __int64 v163; // [rsp+88h] [rbp-128h]
  __int64 v164; // [rsp+90h] [rbp-120h]
  __int64 v165; // [rsp+A0h] [rbp-110h] BYREF
  __int64 v166; // [rsp+A8h] [rbp-108h]
  __int64 v167; // [rsp+B0h] [rbp-100h]
  unsigned __int32 v168; // [rsp+B8h] [rbp-F8h]
  const __m128i *v169; // [rsp+C0h] [rbp-F0h] BYREF
  __m128i *v170; // [rsp+C8h] [rbp-E8h]
  const __m128i *v171; // [rsp+D0h] [rbp-E0h]
  __m128i v172; // [rsp+E0h] [rbp-D0h] BYREF
  __m128i v173; // [rsp+F0h] [rbp-C0h] BYREF
  __m128i *v174; // [rsp+100h] [rbp-B0h]
  __int64 v175; // [rsp+108h] [rbp-A8h]
  __int64 v176; // [rsp+110h] [rbp-A0h]
  __int64 v177[3]; // [rsp+120h] [rbp-90h] BYREF
  __int64 *v178; // [rsp+138h] [rbp-78h]
  __int64 v179; // [rsp+140h] [rbp-70h]
  unsigned int v180; // [rsp+148h] [rbp-68h]
  unsigned __int64 v181[2]; // [rsp+150h] [rbp-60h] BYREF
  _BYTE v182[80]; // [rsp+160h] [rbp-50h] BYREF

  v6 = *(_QWORD *)(a2 + 8);
  v177[0] = a2;
  v181[0] = (unsigned __int64)v182;
  v177[1] = v6;
  v181[1] = 0x400000000LL;
  v177[2] = 0;
  v178 = 0;
  v179 = 0;
  v180 = 0;
  sub_1394350(v177, a3);
  v165 = 0;
  v166 = 0;
  v7 = v179;
  v167 = 0;
  v168 = 0;
  v169 = 0;
  v170 = 0;
  v171 = 0;
  v156 = v178;
  v146 = &v178[4 * v180];
  if ( !(_DWORD)v179 )
    goto LABEL_2;
  if ( v178 != v146 )
  {
    v20 = v178;
    while ( *v20 == -8 || *v20 == -16 )
    {
      v20 += 4;
      if ( v146 == v20 )
        goto LABEL_32;
    }
    if ( v146 == v20 )
    {
      v22 = v178;
      v21 = &v178[4 * v180];
      goto LABEL_36;
    }
    v51 = *v20;
    v145 = v3;
    v52 = &v178[4 * v180];
    if ( !(unsigned __int8)sub_138ECE0(*v20) )
      goto LABEL_80;
    while ( 1 )
    {
LABEL_73:
      v53 = v20 + 4;
      if ( v20 + 4 == v52 )
        goto LABEL_77;
      while ( 1 )
      {
        v51 = *v53;
        v20 = v53;
        if ( *v53 != -16 && v51 != -8 )
          break;
        v53 += 4;
        if ( v52 == v53 )
          goto LABEL_77;
      }
      if ( v53 == v52 )
      {
LABEL_77:
        v3 = v145;
        v156 = v178;
        v7 = v179;
        v146 = &v178[4 * v180];
        goto LABEL_32;
      }
      if ( (unsigned __int8)sub_138ECE0(*v53) )
        continue;
LABEL_80:
      v54 = v168;
      if ( v168 )
      {
        v55 = v168 - 1;
        v56 = v166;
        v57 = 1;
        v58 = ((((unsigned __int64)(((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4)) << 32) - 1) >> 22)
            ^ (((unsigned __int64)(((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4)) << 32) - 1);
        v59 = ((9 * (((v58 - 1 - (v58 << 13)) >> 8) ^ (v58 - 1 - (v58 << 13)))) >> 15)
            ^ (9 * (((v58 - 1 - (v58 << 13)) >> 8) ^ (v58 - 1 - (v58 << 13))));
        for ( i = (v168 - 1) & (((v59 - 1 - (v59 << 27)) >> 31) ^ (v59 - 1 - ((_DWORD)v59 << 27))); ; i = v55 & v62 )
        {
          v61 = v166 + 24LL * i;
          if ( *(_QWORD *)v61 == v51 && !*(_DWORD *)(v61 + 8) )
            break;
          if ( *(_QWORD *)v61 == -8 && *(_DWORD *)(v61 + 8) == -1 )
            goto LABEL_86;
          v62 = v57 + i;
          ++v57;
        }
        if ( v61 != v166 + 24LL * v168 )
          break;
      }
LABEL_86:
      v63 = v170;
      v172.m128i_i64[1] = -1;
      v173.m128i_i64[0] = 0;
      v173.m128i_i32[2] = -1;
      v64 = ((char *)v170 - (char *)v169) >> 5;
      v172.m128i_i32[0] = v64;
      if ( v170 == v171 )
      {
        sub_1390910(&v169, v170, &v172);
      }
      else
      {
        if ( v170 )
        {
          *v170 = _mm_loadu_si128(&v172);
          v63[1] = _mm_loadu_si128(&v173);
          v63 = v170;
        }
        v170 = v63 + 2;
      }
      sub_1392730((__int64)&v165, v51, 0, v64);
      v54 = v168;
      v56 = v166;
      v65 = *(_QWORD *)(v20[1] + 48);
      if ( v168 )
      {
        v55 = v168 - 1;
LABEL_168:
        v132 = 1;
        v133 = ((((unsigned __int64)(((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4)) << 32) - 1) >> 22)
             ^ (((unsigned __int64)(((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4)) << 32) - 1);
        v134 = ((9 * (((v133 - 1 - (v133 << 13)) >> 8) ^ (v133 - 1 - (v133 << 13)))) >> 15)
             ^ (9 * (((v133 - 1 - (v133 << 13)) >> 8) ^ (v133 - 1 - (v133 << 13))));
        for ( j = v55 & (((v134 - 1 - (v134 << 27)) >> 31) ^ (v134 - 1 - ((_DWORD)v134 << 27))); ; j = v55 & v137 )
        {
          v136 = v56 + 24LL * j;
          if ( *(_QWORD *)v136 == v51 && !*(_DWORD *)(v136 + 8) )
            break;
          if ( *(_QWORD *)v136 == -8 && *(_DWORD *)(v136 + 8) == -1 )
            goto LABEL_91;
          v137 = v132 + j;
          ++v132;
        }
        if ( v136 != v56 + 24 * v54 )
          v143 = (unsigned int *)(v136 + 16);
      }
LABEL_91:
      v66 = v169;
      v67 = &v169[2 * *v143];
      v68 = v67[1].m128i_u32[2];
      v69 = v67;
      if ( v68 != -1 )
      {
        v70 = v67[1].m128i_u32[2];
        do
        {
          v71 = &v169[2 * v70];
          v70 = v71[1].m128i_u32[2];
        }
        while ( v70 != -1 );
        v72 = v71->m128i_i32[0];
        while ( 1 )
        {
          v67[1].m128i_i32[2] = v72;
          v69 = &v66[2 * v68];
          v68 = v69[1].m128i_u32[2];
          if ( v68 == -1 )
            break;
          v66 = v169;
          v67 = v69;
        }
      }
      v69[1].m128i_i64[0] |= v65;
      v151 = -1227133513 * ((v20[2] - v20[1]) >> 3) - 1;
      if ( -1227133513 * (unsigned int)((v20[2] - v20[1]) >> 3) != 1 )
      {
        v142 = v52;
        v73 = 0;
        v74 = 56;
        v161 = (unsigned __int64)(((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4)) << 32;
        v75 = 37;
        while ( 1 )
        {
          v76 = v168;
          v159 = (unsigned int)(v73 + 1);
          v77 = v73 + 1;
          v78 = v159;
          if ( !v168 )
            goto LABEL_105;
          v79 = v75;
          v147 = 1;
          v80 = v168 - 1;
          v81 = (v75 | v161) - 1 - ((unsigned __int64)v75 << 32);
          v82 = 9
              * ((((v81 ^ (v81 >> 22)) - 1 - ((v81 ^ (v81 >> 22)) << 13)) >> 8)
               ^ ((v81 ^ (v81 >> 22)) - 1 - ((v81 ^ (v81 >> 22)) << 13)));
          for ( k = (v168 - 1)
                  & ((((v82 ^ (v82 >> 15)) - 1 - ((v82 ^ (v82 >> 15)) << 27)) >> 31)
                   ^ ((v82 ^ (v82 >> 15)) - 1 - (((unsigned int)v82 ^ (unsigned int)(v82 >> 15)) << 27))); ; k = v80 & v85 )
          {
            v84 = v56 + 24LL * k;
            if ( *(_QWORD *)v84 == v51 && v77 == *(_DWORD *)(v84 + 8) )
              break;
            if ( *(_QWORD *)v84 == -8 && *(_DWORD *)(v84 + 8) == -1 )
              goto LABEL_105;
            v85 = v147 + k;
            ++v147;
          }
          if ( v84 == v56 + 24LL * v168 )
          {
LABEL_105:
            v86 = v170;
            v172.m128i_i64[1] = -1;
            v173.m128i_i64[0] = 0;
            v173.m128i_i32[2] = -1;
            v87 = ((char *)v170 - (char *)v169) >> 5;
            v172.m128i_i32[0] = v87;
            if ( v170 == v171 )
            {
              v144 = v73;
              v140 = ((char *)v170 - (char *)v169) >> 5;
              v149 = (unsigned int)(v73 + 1);
              sub_1390910(&v169, v170, &v172);
              v73 = v144;
              LODWORD(v87) = v140;
              v78 = v149;
            }
            else
            {
              if ( v170 )
              {
                *v170 = _mm_loadu_si128(&v172);
                v86[1] = _mm_loadu_si128(&v173);
                v86 = v170;
              }
              v170 = v86 + 2;
            }
            v148 = v73;
            sub_1392730((__int64)&v165, v51, v78, v87);
            v76 = v168;
            v56 = v166;
            v73 = v148;
            v88 = *(_QWORD *)(v20[1] + v74 + 48);
            if ( !v168 )
              goto LABEL_110;
            v79 = v75;
            v80 = v168 - 1;
          }
          else
          {
            v88 = *(_QWORD *)(v20[1] + v74 + 48);
          }
          v120 = 1;
          v121 = (((v79 | v161) - 1 - (v79 << 32)) >> 22) ^ ((v79 | v161) - 1 - (v79 << 32));
          v122 = 9 * (((v121 - 1 - (v121 << 13)) >> 8) ^ (v121 - 1 - (v121 << 13)));
          for ( m = v80
                  & ((((v122 ^ (v122 >> 15)) - 1 - ((v122 ^ (v122 >> 15)) << 27)) >> 31)
                   ^ ((v122 ^ (v122 >> 15)) - 1 - (((unsigned int)v122 ^ (unsigned int)(v122 >> 15)) << 27)));
                ;
                m = v80 & v125 )
          {
            v124 = v56 + 24LL * m;
            if ( *(_QWORD *)v124 == v51 && v77 == *(_DWORD *)(v124 + 8) )
              break;
            if ( *(_QWORD *)v124 == -8 && *(_DWORD *)(v124 + 8) == -1 )
              goto LABEL_110;
            v125 = v120 + m;
            ++v120;
          }
          if ( v124 != v56 + 24 * v76 )
            v154 = (unsigned int *)(v124 + 16);
LABEL_110:
          v89 = v169;
          v90 = &v169[2 * *v154];
          v91 = v90[1].m128i_u32[2];
          v92 = v90;
          if ( v91 != -1 )
          {
            v93 = v90[1].m128i_u32[2];
            do
            {
              v94 = &v169[2 * v93];
              v93 = v94[1].m128i_u32[2];
            }
            while ( v93 != -1 );
            v95 = v94->m128i_i32[0];
            while ( 1 )
            {
              v90[1].m128i_i32[2] = v95;
              v92 = &v89[2 * v91];
              v91 = v92[1].m128i_u32[2];
              if ( v91 == -1 )
                break;
              v89 = v169;
              v90 = v92;
            }
          }
          v92[1].m128i_i64[0] |= v88;
          v96 = v159;
          if ( !v168 )
          {
LABEL_122:
            v103 = v169;
LABEL_123:
            v104 = v103;
            goto LABEL_124;
          }
          v97 = 1;
          v98 = ((((v75 - 37) | v161) - 1 - ((unsigned __int64)(v75 - 37) << 32)) >> 22)
              ^ (((v75 - 37) | v161) - 1 - ((unsigned __int64)(v75 - 37) << 32));
          v99 = ((9 * (((v98 - 1 - (v98 << 13)) >> 8) ^ (v98 - 1 - (v98 << 13)))) >> 15)
              ^ (9 * (((v98 - 1 - (v98 << 13)) >> 8) ^ (v98 - 1 - (v98 << 13))));
          for ( n = (v168 - 1) & (((v99 - 1 - (v99 << 27)) >> 31) ^ (v99 - 1 - ((_DWORD)v99 << 27)));
                ;
                n = (v168 - 1) & v102 )
          {
            v101 = v56 + 24LL * n;
            if ( *(_QWORD *)v101 == v51 && *(_DWORD *)(v101 + 8) == v73 )
              break;
            if ( *(_QWORD *)v101 == -8 && *(_DWORD *)(v101 + 8) == -1 )
              goto LABEL_122;
            v102 = v97 + n;
            ++v97;
          }
          v103 = v169;
          v104 = v169;
          if ( v101 == v56 + 24LL * v168 )
            goto LABEL_123;
          v126 = &v169[2 * *(unsigned int *)(v101 + 16)];
          v127 = v126[1].m128i_u32[2];
          if ( v127 != -1 )
          {
            v128 = v126[1].m128i_u32[2];
            do
            {
              v129 = &v169[2 * v128];
              v128 = v129[1].m128i_u32[2];
            }
            while ( v128 != -1 );
            v130 = v129->m128i_i32[0];
            do
            {
              v126[1].m128i_i32[2] = v130;
              v131 = &v104[2 * v127];
              v104 = v169;
              v127 = v131[1].m128i_u32[2];
              v126 = v131;
            }
            while ( v127 != -1 );
          }
          v152 = v126->m128i_i32[0];
LABEL_124:
          v105 = &v104[2 * v152];
          v106 = 2LL * v152;
          v107 = v105[1].m128i_u32[2];
          v108 = v105;
          if ( v107 != -1 )
          {
            v109 = v105[1].m128i_u32[2];
            do
            {
              v110 = &v104[2 * v109];
              v109 = v110[1].m128i_u32[2];
            }
            while ( v109 != -1 );
            v111 = v110->m128i_i32[0];
            while ( 1 )
            {
              v105[1].m128i_i32[2] = v111;
              v108 = &v104[2 * v107];
              v104 = v169;
              v107 = v108[1].m128i_u32[2];
              if ( v107 == -1 )
                break;
              v105 = v108;
            }
          }
          if ( v108->m128i_i32[3] == -1 )
          {
            v118 = v170;
            v172.m128i_i64[1] = -1;
            v173.m128i_i64[0] = 0;
            v173.m128i_i32[2] = -1;
            v119 = ((char *)v170 - (char *)v104) >> 5;
            v172.m128i_i32[0] = v119;
            if ( v170 == v171 )
            {
              v141 = v119;
              sub_1390910(&v169, v170, &v172);
              v104 = v169;
              LODWORD(v119) = v141;
              v96 = v159;
              v106 = 2LL * v152;
            }
            else
            {
              if ( v170 )
              {
                *v170 = _mm_loadu_si128(&v172);
                v118[1] = _mm_loadu_si128(&v173);
                v118 = v170;
                v104 = v169;
              }
              v170 = v118 + 2;
            }
            v104[v106].m128i_i32[3] = v119;
            v169[2 * (unsigned int)v119].m128i_i32[2] = v152;
            v104 = v169;
          }
          v112 = &v104[v106];
          v113 = v104[v106 + 1].m128i_u32[2];
          v114 = v112;
          if ( v113 != -1 )
          {
            for ( ii = v113; ii != -1; ii = v116[1].m128i_u32[2] )
              v116 = &v104[2 * ii];
            v117 = v116->m128i_i32[0];
            while ( 1 )
            {
              v112[1].m128i_i32[2] = v117;
              v114 = &v104[2 * v113];
              v113 = v114[1].m128i_u32[2];
              if ( v113 == -1 )
                break;
              v104 = v169;
              v112 = v114;
            }
          }
          v75 += 37;
          v74 += 56;
          sub_1392730((__int64)&v165, v51, v96, v114->m128i_u32[3]);
          if ( v77 == v151 )
          {
            v52 = v142;
            goto LABEL_73;
          }
          v56 = v166;
          v73 = v77;
        }
      }
    }
    v65 = *(_QWORD *)(v20[1] + 48);
    goto LABEL_168;
  }
LABEL_32:
  if ( v7 )
  {
    v21 = v146;
    v22 = v156;
    if ( v146 != v156 )
    {
LABEL_36:
      while ( 1 )
      {
        v23 = *v22;
        if ( *v22 != -16 && v23 != -8 )
          break;
        v22 += 4;
        if ( v21 == v22 )
          goto LABEL_2;
      }
      v157 = v22;
      if ( v146 != v22 )
      {
        v24 = *v22;
        if ( !(unsigned __int8)sub_138ECE0(v23) )
          goto LABEL_47;
        while ( 1 )
        {
          v25 = v157 + 4;
          if ( v157 + 4 == v146 )
            break;
          while ( 1 )
          {
            v24 = *v25;
            if ( *v25 != -16 && v24 != -8 )
              break;
            v25 += 4;
            if ( v146 == v25 )
              goto LABEL_2;
          }
          v157 = v25;
          if ( v146 == v25 )
            break;
          if ( !(unsigned __int8)sub_138ECE0(v24) )
          {
LABEL_47:
            v26 = v157[1];
            v27 = 0x6DB6DB6DB6DB6DB7LL * ((v157[2] - v26) >> 3);
            if ( (_DWORD)v27 )
            {
              v160 = v24;
              v153 = (unsigned int)v27;
              v28 = ((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4);
              v29 = 0;
              v150 = v28 << 32;
              while ( 1 )
              {
                v30 = (__int64 **)(v26 + 56 * v29);
                v31 = v30[1];
                if ( v31 != *v30 )
                {
                  v32 = v3;
                  v158 = v29;
                  v33 = *v30;
                  v155 = v29;
                  v34 = ((((unsigned int)(37 * v29) | (unsigned __int64)v150)
                        - 1
                        - ((unsigned __int64)(unsigned int)(37 * v29) << 32)) >> 22)
                      ^ (((unsigned int)(37 * v29) | (unsigned __int64)v150)
                       - 1
                       - ((unsigned __int64)(unsigned int)(37 * v29) << 32));
                  v35 = ((9 * (((v34 - 1 - (v34 << 13)) >> 8) ^ (v34 - 1 - (v34 << 13)))) >> 15)
                      ^ (9 * (((v34 - 1 - (v34 << 13)) >> 8) ^ (v34 - 1 - (v34 << 13))));
                  v36 = ((v35 - 1 - (v35 << 27)) >> 31) ^ (v35 - 1 - ((_DWORD)v35 << 27));
                  v37 = v32;
                  do
                  {
                    if ( v168 )
                    {
                      v38 = 1;
                      for ( jj = v36 & (v168 - 1); ; jj = (v168 - 1) & v41 )
                      {
                        v40 = v166 + 24LL * jj;
                        if ( v160 == *(_QWORD *)v40 && v158 == *(_DWORD *)(v40 + 8) )
                          break;
                        if ( *(_QWORD *)v40 == -8 && *(_DWORD *)(v40 + 8) == -1 )
                          goto LABEL_65;
                        v41 = v38 + jj;
                        ++v38;
                      }
                      if ( v40 != v166 + 24LL * v168 )
                      {
                        v42 = v169;
                        v43 = &v169[2 * *(unsigned int *)(v40 + 16)];
                        v44 = v43[1].m128i_u32[2];
                        if ( v44 != -1 )
                        {
                          v45 = v43[1].m128i_u32[2];
                          do
                          {
                            v46 = &v169[2 * v45];
                            v45 = v46[1].m128i_u32[2];
                          }
                          while ( v45 != -1 );
                          v47 = v46->m128i_i32[0];
                          for ( kk = v43; ; kk = v43 )
                          {
                            kk[1].m128i_i32[2] = v47;
                            v43 = &v42[2 * v44];
                            v44 = v43[1].m128i_u32[2];
                            if ( v44 == -1 )
                              break;
                            v42 = v169;
                          }
                        }
                        v37 = v43->m128i_i32[0];
                      }
                    }
LABEL_65:
                    v49 = *v33;
                    v50 = v33[1];
                    v33 += 3;
                    sub_1392730((__int64)&v165, v49, v50, v37);
                  }
                  while ( v31 != v33 );
                  v29 = v155;
                  v3 = v37;
                }
                if ( v153 == ++v29 )
                  break;
                v26 = v157[1];
              }
            }
          }
        }
      }
    }
  }
LABEL_2:
  v162 = 0;
  v163 = 0;
  v164 = 0;
  sub_1392D30((__int64)&v165, &v162);
  sub_1390AA0((__int64 *)&v162);
  if ( v169 != v170 )
    v170 = (__m128i *)v169;
  v8 = v162;
  v172.m128i_i64[0] = 1;
  v9 = v163;
  v10 = v164;
  v163 = 0;
  v11 = v166;
  v12 = v167;
  v173.m128i_i32[1] = HIDWORD(v167);
  v13 = v168;
  ++v165;
  v164 = 0;
  v162 = 0;
  v166 = 0;
  v167 = 0;
  v168 = 0;
  v172.m128i_i64[1] = v11;
  v173.m128i_i32[0] = v12;
  v173.m128i_i32[2] = v13;
  v174 = v8;
  v175 = v9;
  v176 = v10;
  j___libc_free_0(0);
  if ( v162 )
    j_j___libc_free_0(v162, v164 - (_QWORD)v162);
  sub_1390580(a1, a3, (__int64)v181, (__int64)&v172);
  if ( v174 )
    j_j___libc_free_0(v174, v176 - (_QWORD)v174);
  j___libc_free_0(v172.m128i_i64[1]);
  if ( v169 )
    j_j___libc_free_0(v169, (char *)v171 - (char *)v169);
  j___libc_free_0(v166);
  if ( (_BYTE *)v181[0] != v182 )
    _libc_free(v181[0]);
  if ( v180 )
  {
    v14 = v178;
    v15 = &v178[4 * v180];
    do
    {
      if ( *v14 != -8 && *v14 != -16 )
      {
        v16 = (_QWORD *)v14[2];
        v17 = (_QWORD *)v14[1];
        if ( v16 != v17 )
        {
          do
          {
            v18 = v17[3];
            if ( v18 )
              j_j___libc_free_0(v18, v17[5] - v18);
            if ( *v17 )
              j_j___libc_free_0(*v17, v17[2] - *v17);
            v17 += 7;
          }
          while ( v16 != v17 );
          v17 = (_QWORD *)v14[1];
        }
        if ( v17 )
          j_j___libc_free_0(v17, v14[3] - (_QWORD)v17);
      }
      v14 += 4;
    }
    while ( v15 != v14 );
  }
  j___libc_free_0(v178);
  return a1;
}
