// Function: sub_1BC6D90
// Address: 0x1bc6d90
//
__int64 __fastcall sub_1BC6D90(__m128i *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 *v4; // r12
  __int64 v5; // rbx
  __int32 v6; // ecx
  __int64 v7; // r14
  __int64 v8; // r15
  __int32 v9; // edx
  __int8 v10; // al
  __m128i v11; // xmm0
  __int32 v12; // eax
  __int32 v13; // edx
  unsigned int v14; // r14d
  __int32 v16; // esi
  __int64 v17; // rcx
  __int32 v18; // edx
  __int8 v19; // al
  __m128i v20; // xmm2
  __int32 v21; // esi
  __int64 v22; // rcx
  __int32 v23; // edx
  __int8 v24; // al
  __m128i v25; // xmm3
  __int64 v26; // r15
  int v27; // r8d
  int v28; // r9d
  char v29; // al
  __int32 v30; // eax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rax
  unsigned __int64 *v35; // rbx
  unsigned __int64 *v36; // r15
  bool v37; // cc
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // r15
  __int64 v41; // rbx
  __int64 *v42; // rax
  __int64 v43; // rcx
  __int64 v44; // rax
  unsigned __int64 *v45; // rbx
  unsigned __int64 *v46; // r15
  int v47; // edi
  _QWORD *v48; // rdi
  __int64 v49; // rax
  __int64 *v50; // rdi
  __int64 v51; // rax
  __int64 v52; // r15
  __int64 v53; // r12
  unsigned int v54; // r8d
  int v55; // r9d
  __int64 v56; // r12
  __int64 v57; // r12
  int v58; // r8d
  __int64 v59; // r9
  __m128i v60; // xmm1
  int v61; // edx
  __int32 v62; // r15d
  int v63; // eax
  int v64; // esi
  _DWORD *v65; // r15
  __int64 v66; // rax
  __int64 v67; // rsi
  unsigned int v68; // ecx
  __int64 *v69; // rdx
  __int64 v70; // r9
  __int64 v71; // rdx
  __int64 v72; // rsi
  __int64 *v73; // rcx
  __int64 v74; // rdi
  __int64 v75; // rdx
  __int64 v76; // r15
  __int64 v77; // r8
  unsigned int v78; // eax
  _QWORD *v79; // rdx
  __int64 v80; // r15
  unsigned int v81; // eax
  __int64 *v82; // rdx
  bool v83; // al
  bool v84; // cl
  __int64 v85; // rax
  int v86; // eax
  bool v87; // al
  __int64 v88; // r15
  unsigned int v89; // eax
  __int64 *v90; // rdx
  __int64 v91; // rax
  int v92; // edi
  __int64 v93; // rsi
  unsigned int v94; // r15d
  __int64 *v95; // rdx
  __int64 v96; // rcx
  __int64 v97; // rax
  char *v98; // rdx
  char *v99; // rax
  unsigned int v100; // r11d
  __int64 *v101; // rcx
  __int64 v102; // r9
  char *v103; // rcx
  __int64 v104; // rsi
  __int64 v105; // rdx
  char *v106; // rdx
  unsigned __int64 v107; // rax
  _QWORD *v108; // rcx
  _QWORD *v109; // rsi
  _QWORD *v110; // rdx
  unsigned __int64 v111; // rdi
  __int64 v112; // rax
  _BOOL8 v113; // r15
  __int64 *v114; // rax
  int v115; // ecx
  int v116; // r10d
  __int64 v117; // rcx
  __int32 v118; // eax
  int v119; // ecx
  int v120; // r10d
  __int64 v121; // rdi
  __int64 *v122; // rax
  char v123; // al
  char v124; // al
  int v125; // edx
  int v126; // r11d
  __m128i v127; // xmm4
  int v128; // eax
  int v129; // edx
  int v130; // r8d
  bool v131; // [rsp+3h] [rbp-31Dh]
  int v132; // [rsp+4h] [rbp-31Ch]
  bool v133; // [rsp+8h] [rbp-318h]
  int v134; // [rsp+8h] [rbp-318h]
  int v135; // [rsp+8h] [rbp-318h]
  __int64 v136; // [rsp+8h] [rbp-318h]
  __int64 v137; // [rsp+18h] [rbp-308h]
  __int64 *v138; // [rsp+18h] [rbp-308h]
  __m128i v139; // [rsp+30h] [rbp-2F0h] BYREF
  __int64 v140; // [rsp+40h] [rbp-2E0h]
  int v141; // [rsp+48h] [rbp-2D8h]
  char v142; // [rsp+4Ch] [rbp-2D4h]
  __m128i v143; // [rsp+50h] [rbp-2D0h] BYREF
  __int64 v144; // [rsp+60h] [rbp-2C0h] BYREF
  int v145; // [rsp+68h] [rbp-2B8h]
  char v146; // [rsp+6Ch] [rbp-2B4h]
  __m128i v147; // [rsp+E0h] [rbp-240h] BYREF
  __int64 v148; // [rsp+F0h] [rbp-230h] BYREF
  __int64 v149; // [rsp+F8h] [rbp-228h]

  v3 = a2;
  v4 = (__int64 *)a3;
  v5 = (__int64)a1;
  if ( a3 )
  {
    sub_1BBFB40((__int64)&v147, a3);
    v6 = v147.m128i_i32[0];
    v7 = v147.m128i_i64[1];
    v8 = v148;
    v9 = v149;
    v10 = BYTE4(v149);
  }
  else
  {
    v8 = 0;
    v7 = 0;
    v10 = 0;
    v9 = 0;
    v6 = 0;
  }
  v147.m128i_i32[0] = v6;
  v147.m128i_i64[1] = v7;
  v11 = _mm_load_si128(&v147);
  v148 = v8;
  LODWORD(v149) = v9;
  BYTE4(v149) = v10;
  a1[41].m128i_i64[0] = v8;
  a1[41].m128i_i32[2] = v9;
  a1[41].m128i_i8[12] = v10;
  a1[40] = v11;
  if ( a2 )
  {
    if ( a2 == v7 )
    {
      if ( *(_BYTE *)(v8 + 16) <= 0x17u )
      {
        v19 = 0;
        v18 = 0;
        v17 = 0;
        v7 = 0;
        v16 = 0;
        v4 = 0;
      }
      else
      {
        v4 = (__int64 *)v8;
        sub_1BBFB40((__int64)&v147, v8);
        v16 = v147.m128i_i32[0];
        v7 = v147.m128i_i64[1];
        v17 = v148;
        v18 = v149;
        v19 = BYTE4(v149);
      }
      v147.m128i_i32[0] = v16;
      v3 = 0;
      v147.m128i_i64[1] = v7;
      v20 = _mm_load_si128(&v147);
      v148 = v17;
      LODWORD(v149) = v18;
      BYTE4(v149) = v19;
      a1[41].m128i_i64[0] = v17;
      a1[41].m128i_i32[2] = v18;
      a1[41].m128i_i8[12] = v19;
      a1[40] = v20;
    }
    else if ( a2 == v8 )
    {
      if ( *(_BYTE *)(v7 + 16) <= 0x17u )
      {
        v24 = 0;
        v23 = 0;
        v22 = 0;
        v7 = 0;
        v21 = 0;
        v4 = 0;
      }
      else
      {
        v4 = (__int64 *)v7;
        sub_1BBFB40((__int64)&v147, v7);
        v21 = v147.m128i_i32[0];
        v22 = v148;
        v23 = v149;
        v24 = BYTE4(v149);
        v7 = v147.m128i_i64[1];
      }
      v147.m128i_i32[0] = v21;
      v3 = 0;
      v147.m128i_i64[1] = v7;
      v25 = _mm_load_si128(&v147);
      v148 = v22;
      LODWORD(v149) = v23;
      BYTE4(v149) = v24;
      a1[41].m128i_i64[0] = v22;
      a1[41].m128i_i32[2] = v23;
      a1[41].m128i_i8[12] = v24;
      a1[40] = v25;
    }
  }
  if ( !v7 || !a1[41].m128i_i64[0] )
    return 0;
  v12 = a1[41].m128i_i32[2];
  v13 = a1[40].m128i_i32[0];
  if ( v12 == 1 )
  {
    if ( (unsigned int)(v13 - 11) > 1 || !(unsigned __int8)sub_15F34B0((__int64)v4) )
      return 0;
  }
  else
  {
    if ( (unsigned int)(v13 - 51) > 1 )
      return 0;
    if ( ((v12 - 2) & 0xFFFFFFFD) == 0 )
    {
      switch ( a1[41].m128i_i32[2] )
      {
        case 0:
        case 1:
        case 3:
        case 5:
          goto LABEL_23;
        case 2:
        case 4:
          if ( v13 != 51 )
          {
            v42 = (__int64 *)sub_13CF970((__int64)v4);
            if ( !sub_15F2480(*v42) )
              return 0;
          }
          goto LABEL_23;
      }
    }
    if ( v13 != 51 || ((v12 - 3) & 0xFFFFFFFD) != 0 )
      return 0;
  }
LABEL_23:
  v26 = *v4;
  v14 = sub_1643F10(*v4);
  if ( !(_BYTE)v14 )
    return 0;
  v29 = *(_BYTE *)(v26 + 8);
  if ( (v29 & 0xFD) == 4 )
    return 0;
  if ( v29 == 16 )
    v29 = *(_BYTE *)(**(_QWORD **)(v26 + 16) + 8LL);
  if ( v29 != 11 && (unsigned __int8)(v29 - 1) > 5u )
    return 0;
  a1[42].m128i_i32[0] = 0;
  v147.m128i_i64[0] = (__int64)&v148;
  v147.m128i_i64[1] = 0x2000000000LL;
  v30 = a1[41].m128i_i32[2];
  a1[42].m128i_i64[1] = 0;
  v31 = (unsigned int)(v30 - 2);
  a1[43].m128i_i8[12] = 0;
  a1[43].m128i_i64[0] = 0;
  a1[43].m128i_i32[2] = 0;
  a1[39].m128i_i64[1] = (__int64)v4;
  v148 = (__int64)v4;
  if ( (unsigned int)v31 > 3 )
  {
    v149 = 0;
    v147.m128i_i32[2] = 1;
    if ( v30 != 1 )
    {
      v50 = &v148;
      v49 = 1;
LABEL_62:
      v138 = v4;
      while ( 1 )
      {
        v51 = (__int64)&v50[2 * v49 - 2];
        v52 = *(unsigned int *)(v51 + 8);
        v53 = *(_QWORD *)v51;
        *(_DWORD *)(v51 + 8) = v52 + 1;
        if ( v53
          && (sub_1BBFB40((__int64)&v139, v53), *(_DWORD *)(v5 + 664) == v141)
          && *(_DWORD *)(v5 + 640) == v139.m128i_i32[0] )
        {
          if ( (_DWORD)v52 != (v141 != 1) + 2 )
          {
            if ( (*(_BYTE *)(v53 + 23) & 0x40) != 0 )
              v56 = *(_QWORD *)(v53 - 8);
            else
              v56 = v53 - 24LL * (*(_DWORD *)(v53 + 20) & 0xFFFFFFF);
            v57 = *(_QWORD *)(v56 + 24 * v52);
            if ( v3 != v57 && *(_BYTE *)(v57 + 16) > 0x17u )
            {
              sub_1BBFB40((__int64)&v143, v57);
              v60 = _mm_load_si128(&v143);
              v61 = v145;
              v62 = v143.m128i_i32[0];
              v140 = v144;
              v141 = v145;
              v142 = v146;
              v63 = *(_DWORD *)(v5 + 672);
              v139 = v60;
              if ( !v63 || *(_DWORD *)(v5 + 696) == v145 && v63 == v143.m128i_i32[0] )
              {
                v64 = *(_DWORD *)(v5 + 664);
                if ( v64 != v145 || (v121 = v138[5], *(_DWORD *)(v5 + 640) != v143.m128i_i32[0]) )
                {
                  v83 = *(_QWORD *)(v57 + 40) == v138[5];
                  v84 = 0;
                  goto LABEL_110;
                }
LABEL_174:
                if ( v64 == 1 )
                {
                  v84 = v14;
                  v83 = *(_QWORD *)(v57 + 40) == v121;
LABEL_110:
                  if ( !v83 )
                    goto LABEL_118;
                  v131 = 0;
                  if ( v64 != v145 )
                    goto LABEL_112;
                  v118 = *(_DWORD *)(v5 + 640);
LABEL_168:
                  v131 = v118 == v143.m128i_i32[0];
LABEL_112:
                  if ( v64 == 1 )
                  {
                    v85 = *(_QWORD *)(v57 + 8);
                    if ( v85 )
                      goto LABEL_114;
LABEL_122:
                    if ( (__int64 *)v57 != v138 )
                    {
LABEL_118:
                      sub_1BC6CD0(v5, v147.m128i_i64[0] + 16LL * v147.m128i_u32[2] - 16, v57);
                      v49 = v147.m128i_u32[2];
                      goto LABEL_65;
                    }
                  }
                  else
                  {
                    v132 = v145;
                    v133 = v84;
                    v87 = sub_1648CD0(v57, 2);
                    v84 = v133;
                    v61 = v132;
                    if ( !v87 )
                      goto LABEL_122;
                    if ( v131 )
                    {
                      v85 = *(_QWORD *)(*(_QWORD *)(v57 - 72) + 8LL);
                      if ( !v85 )
                        goto LABEL_122;
LABEL_114:
                      if ( *(_QWORD *)(v85 + 8) )
                        goto LABEL_122;
                    }
                  }
                  if ( v84 )
                  {
                    switch ( v61 )
                    {
                      case 1:
                        v135 = v61;
                        v124 = sub_15F34B0(v57);
                        v61 = v135;
                        if ( v124 )
                          goto LABEL_160;
                        goto LABEL_118;
                      case 2:
                      case 4:
                        if ( v62 != 51 )
                        {
                          v134 = v61;
                          v122 = (__int64 *)sub_13CF970(v57);
                          v123 = sub_15F2480(*v122);
                          v61 = v134;
                          if ( !v123 )
                            goto LABEL_118;
                        }
                        goto LABEL_160;
                      case 3:
                      case 5:
                        goto LABEL_160;
                      default:
                        ++*(_DWORD *)(v5 + 360);
                        BUG();
                    }
                  }
                  v86 = *(_DWORD *)(v5 + 672);
                  if ( !v86 )
                  {
                    v127 = _mm_load_si128(&v139);
                    *(_QWORD *)(v5 + 688) = v140;
                    v128 = v141;
                    *(__m128i *)(v5 + 672) = v127;
                    *(_DWORD *)(v5 + 696) = v128;
                    *(_BYTE *)(v5 + 700) = v142;
LABEL_160:
                    v112 = v147.m128i_u32[2];
                    v113 = (unsigned int)(v61 - 2) <= 3;
                    if ( v147.m128i_i32[2] >= (unsigned __int32)v147.m128i_i32[3] )
                    {
                      sub_16CD150((__int64)&v147, &v148, 0, 16, v58, v59);
                      v112 = v147.m128i_u32[2];
                    }
                    v114 = (__int64 *)(v147.m128i_i64[0] + 16 * v112);
                    *v114 = v57;
                    v114[1] = v113;
                    v49 = (unsigned int)++v147.m128i_i32[2];
                    goto LABEL_65;
                  }
                  if ( *(_DWORD *)(v5 + 696) == v61 && v86 == v62 )
                    goto LABEL_160;
                  goto LABEL_118;
                }
                v59 = *(_QWORD *)(v57 - 72);
                v84 = v59 != 0 && *(_QWORD *)(v57 + 40) == v121;
                if ( v84 && v121 == *(_QWORD *)(v59 + 40) )
                {
                  v118 = v143.m128i_i32[0];
                  goto LABEL_168;
                }
                goto LABEL_118;
              }
              v64 = *(_DWORD *)(v5 + 664);
              if ( v64 == v145 && *(_DWORD *)(v5 + 640) == v143.m128i_i32[0] )
              {
                v121 = v138[5];
                goto LABEL_174;
              }
            }
            v65 = (_DWORD *)(v147.m128i_i64[0] + 16LL * v147.m128i_u32[2] - 16);
            v66 = *(unsigned int *)(v5 + 600);
            if ( (_DWORD)v66 )
            {
              v67 = *(_QWORD *)(v5 + 584);
              v68 = (v66 - 1) & (((unsigned int)*(_QWORD *)v65 >> 9) ^ ((unsigned int)*(_QWORD *)v65 >> 4));
              v69 = (__int64 *)(v67 + 16LL * v68);
              v70 = *v69;
              if ( *(_QWORD *)v65 == *v69 )
              {
LABEL_79:
                if ( v69 != (__int64 *)(v67 + 16 * v66) )
                {
                  *(_QWORD *)sub_1907820(
                               v5 + 576,
                               (unsigned __int64 *)(v147.m128i_i64[0] + 16LL * v147.m128i_u32[2] - 16)) = 0;
                  v65[2] = *(_DWORD *)(*(_QWORD *)v65 + 20LL) & 0xFFFFFFF;
LABEL_81:
                  v49 = v147.m128i_u32[2];
                  goto LABEL_65;
                }
              }
              else
              {
                v125 = 1;
                while ( v70 != -8 )
                {
                  v126 = v125 + 1;
                  v68 = (v66 - 1) & (v125 + v68);
                  v69 = (__int64 *)(v67 + 16LL * v68);
                  v70 = *v69;
                  if ( *(_QWORD *)v65 == *v69 )
                    goto LABEL_79;
                  v125 = v126;
                }
              }
            }
            *(_QWORD *)sub_1907820(v5 + 576, (unsigned __int64 *)(v147.m128i_i64[0] + 16LL * v147.m128i_u32[2] - 16)) = v57;
            goto LABEL_81;
          }
          v71 = *(unsigned int *)(v5 + 600);
          if ( (_DWORD)v71 )
          {
            v55 = v71 - 1;
            v72 = *(_QWORD *)(v5 + 584);
            v54 = (v71 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
            v73 = (__int64 *)(v72 + 16LL * v54);
            v74 = *v73;
            if ( v53 == *v73 )
            {
LABEL_84:
              if ( v73 != (__int64 *)(v72 + 16 * v71) )
              {
                v75 = *(_QWORD *)(v5 + 608) + 16LL * *((unsigned int *)v73 + 2);
                if ( *(_QWORD *)(v5 + 616) != v75 && !*(_QWORD *)(v75 + 8) )
                {
                  v50 = (__int64 *)v147.m128i_i64[0];
                  if ( v147.m128i_u32[2] <= 1uLL )
                  {
                    v14 = 0;
                    goto LABEL_100;
                  }
                  sub_1BC6CD0(v5, v147.m128i_i64[0] + 16 * (v147.m128i_u32[2] - 2LL), v53);
                  v91 = *(unsigned int *)(v5 + 600);
                  if ( (_DWORD)v91 )
                  {
                    v92 = v91 - 1;
                    v93 = *(_QWORD *)(v5 + 584);
                    v94 = (v91 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
                    v95 = (__int64 *)(v93 + 16LL * v94);
                    v96 = *v95;
                    if ( v53 == *v95 )
                    {
LABEL_131:
                      if ( v95 != (__int64 *)(v93 + 16 * v91) )
                      {
                        v97 = *((unsigned int *)v95 + 2);
                        v98 = *(char **)(v5 + 616);
                        v99 = (char *)(*(_QWORD *)(v5 + 608) + 16 * v97);
                        if ( v98 != v99 )
                        {
                          v100 = v92 & (((unsigned int)*(_QWORD *)v99 >> 9) ^ ((unsigned int)*(_QWORD *)v99 >> 4));
                          v101 = (__int64 *)(v93 + 16LL * v100);
                          v102 = *v101;
                          if ( *(_QWORD *)v99 == *v101 )
                          {
LABEL_134:
                            *v101 = -16;
                            v98 = *(char **)(v5 + 616);
                            --*(_DWORD *)(v5 + 592);
                            ++*(_DWORD *)(v5 + 596);
                          }
                          else
                          {
                            v115 = 1;
                            while ( v102 != -8 )
                            {
                              v116 = v115 + 1;
                              v117 = v92 & (v100 + v115);
                              v100 = v117;
                              v101 = (__int64 *)(v93 + 16 * v117);
                              v102 = *v101;
                              if ( *(_QWORD *)v99 == *v101 )
                                goto LABEL_134;
                              v115 = v116;
                            }
                          }
                          v103 = v99 + 16;
                          if ( v99 + 16 != v98 )
                          {
                            v104 = (v98 - v103) >> 4;
                            if ( v98 - v103 > 0 )
                            {
                              do
                              {
                                v105 = *(_QWORD *)v103;
                                v103 += 16;
                                *((_QWORD *)v103 - 4) = v105;
                                *((_QWORD *)v103 - 3) = *((_QWORD *)v103 - 1);
                                --v104;
                              }
                              while ( v104 );
                              v98 = *(char **)(v5 + 616);
                            }
                          }
                          v106 = v98 - 16;
                          *(_QWORD *)(v5 + 616) = v106;
                          if ( v99 != v106 )
                          {
                            v107 = (__int64)&v99[-*(_QWORD *)(v5 + 608)] >> 4;
                            if ( *(_DWORD *)(v5 + 592) )
                            {
                              v108 = *(_QWORD **)(v5 + 584);
                              v109 = &v108[2 * *(unsigned int *)(v5 + 600)];
                              if ( v108 != v109 )
                              {
                                while ( 1 )
                                {
                                  v110 = v108;
                                  if ( *v108 != -16 && *v108 != -8 )
                                    break;
                                  v108 += 2;
                                  if ( v109 == v108 )
                                    goto LABEL_64;
                                }
                                if ( v109 != v108 )
                                {
                                  do
                                  {
                                    v111 = *((unsigned int *)v110 + 2);
                                    if ( v107 < v111 )
                                      *((_DWORD *)v110 + 2) = v111 - 1;
                                    v110 += 2;
                                    if ( v110 == v109 )
                                      break;
                                    while ( *v110 == -8 || *v110 == -16 )
                                    {
                                      v110 += 2;
                                      if ( v109 == v110 )
                                        goto LABEL_64;
                                    }
                                  }
                                  while ( v110 != v109 );
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                    else
                    {
                      v129 = 1;
                      while ( v96 != -8 )
                      {
                        v130 = v129 + 1;
                        v94 = v92 & (v129 + v94);
                        v95 = (__int64 *)(v93 + 16LL * v94);
                        v96 = *v95;
                        if ( v53 == *v95 )
                          goto LABEL_131;
                        v129 = v130;
                      }
                    }
                  }
                  goto LABEL_64;
                }
              }
            }
            else
            {
              v119 = 1;
              while ( v74 != -8 )
              {
                v120 = v119 + 1;
                v54 = v55 & (v119 + v54);
                v73 = (__int64 *)(v72 + 16LL * v54);
                v74 = *v73;
                if ( v53 == *v73 )
                  goto LABEL_84;
                v119 = v120;
              }
            }
          }
          if ( v141 == 1 )
          {
            v88 = *(_QWORD *)v5;
            v89 = *(_DWORD *)(*(_QWORD *)v5 + 8LL);
            if ( v89 >= *(_DWORD *)(*(_QWORD *)v5 + 12LL) )
            {
              sub_16CD150(*(_QWORD *)v5, (const void *)(v88 + 16), 0, 8, v54, v55);
              v89 = *(_DWORD *)(v88 + 8);
            }
            v90 = (__int64 *)(*(_QWORD *)v88 + 8LL * v89);
            if ( v90 )
            {
              *v90 = v53;
              v89 = *(_DWORD *)(v88 + 8);
            }
            *(_DWORD *)(v88 + 8) = v89 + 1;
          }
          else if ( (unsigned int)(v141 - 2) <= 3 )
          {
            v76 = *(_QWORD *)v5;
            v77 = *(_QWORD *)(v53 - 72);
            v78 = *(_DWORD *)(*(_QWORD *)v5 + 8LL);
            if ( v78 >= *(_DWORD *)(*(_QWORD *)v5 + 12LL) )
            {
              v136 = *(_QWORD *)(v53 - 72);
              sub_16CD150(*(_QWORD *)v5, (const void *)(v76 + 16), 0, 8, v77, v55);
              v78 = *(_DWORD *)(v76 + 8);
              v77 = v136;
            }
            v79 = (_QWORD *)(*(_QWORD *)v76 + 8LL * v78);
            if ( v79 )
            {
              *v79 = v77;
              v78 = *(_DWORD *)(v76 + 8);
            }
            *(_DWORD *)(v76 + 8) = v78 + 1;
            v80 = *(_QWORD *)v5;
            v81 = *(_DWORD *)(*(_QWORD *)v5 + 152LL);
            if ( v81 >= *(_DWORD *)(*(_QWORD *)v5 + 156LL) )
            {
              sub_16CD150(v80 + 144, (const void *)(v80 + 160), 0, 8, v77, v55);
              v81 = *(_DWORD *)(v80 + 152);
            }
            v82 = (__int64 *)(*(_QWORD *)(v80 + 144) + 8LL * v81);
            if ( v82 )
            {
              *v82 = v53;
              v81 = *(_DWORD *)(v80 + 152);
            }
            *(_DWORD *)(v80 + 152) = v81 + 1;
          }
        }
        else
        {
          v143.m128i_i64[0] = v53;
          sub_12A9700(v5 + 304, &v143);
        }
LABEL_64:
        v49 = (unsigned int)--v147.m128i_i32[2];
LABEL_65:
        v50 = (__int64 *)v147.m128i_i64[0];
        if ( !(_DWORD)v49 )
        {
          v14 = (unsigned __int8)v14;
          goto LABEL_100;
        }
      }
    }
    v43 = a1->m128i_i64[0];
    v143.m128i_i64[0] = (__int64)&v144;
    v143.m128i_i64[1] = 0x1000000000LL;
    v44 = 144LL * a1->m128i_u32[2];
    if ( v43 != v43 + v44 )
    {
      v45 = (unsigned __int64 *)(v43 + v44);
      v46 = (unsigned __int64 *)v43;
      do
      {
        v45 -= 18;
        v31 = (__int64)(v45 + 2);
        if ( (unsigned __int64 *)*v45 != v45 + 2 )
          _libc_free(*v45);
      }
      while ( v46 != v45 );
      v5 = (__int64)a1;
    }
    v47 = *(_DWORD *)(v5 + 12);
    *(_DWORD *)(v5 + 8) = 0;
    if ( !v47 )
      sub_1BC1BD0(v5, 1u);
    v48 = *(_QWORD **)v5;
    *(_DWORD *)(v5 + 8) = 1;
    if ( v48 )
    {
      *v48 = v48 + 2;
      v48[1] = 0x1000000000LL;
      if ( v143.m128i_i32[2] )
        sub_1BB9B80((__int64)v48, (__int64)&v143, v31, v43, v27, v28);
    }
  }
  else
  {
    v32 = a1->m128i_i64[0];
    v149 = 1;
    v143.m128i_i64[0] = (__int64)&v144;
    v143.m128i_i64[1] = 0x1000000000LL;
    v33 = a1->m128i_u32[2];
    v147.m128i_i32[2] = 1;
    v34 = 144 * v33;
    if ( v32 != v32 + v34 )
    {
      v35 = (unsigned __int64 *)(v32 + v34);
      v36 = (unsigned __int64 *)v32;
      do
      {
        v35 -= 18;
        if ( (unsigned __int64 *)*v35 != v35 + 2 )
          _libc_free(*v35);
      }
      while ( v36 != v35 );
      v5 = (__int64)a1;
    }
    v37 = *(_DWORD *)(v5 + 12) <= 1u;
    *(_DWORD *)(v5 + 8) = 0;
    if ( v37 )
      sub_1BC1BD0(v5, 2u);
    v38 = *(_QWORD *)v5;
    *(_DWORD *)(v5 + 8) = 2;
    v137 = v5;
    v39 = v38 + 288;
    v40 = v38;
    v41 = v38 + 288;
    do
    {
      if ( v40 )
      {
        *(_DWORD *)(v40 + 8) = 0;
        *(_QWORD *)v40 = v40 + 16;
        *(_DWORD *)(v40 + 12) = 16;
        if ( v143.m128i_i32[2] )
          sub_1BB9B80(v40, (__int64)&v143, v39, v143.m128i_u32[2], v27, v28);
      }
      v40 += 144;
    }
    while ( v41 != v40 );
    v5 = v137;
  }
  if ( (__int64 *)v143.m128i_i64[0] != &v144 )
    _libc_free(v143.m128i_u64[0]);
  v49 = v147.m128i_u32[2];
  v50 = (__int64 *)v147.m128i_i64[0];
  if ( v147.m128i_i32[2] )
    goto LABEL_62;
LABEL_100:
  if ( v50 != &v148 )
    _libc_free((unsigned __int64)v50);
  return v14;
}
