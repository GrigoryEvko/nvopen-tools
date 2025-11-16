// Function: sub_9BF710
// Address: 0x9bf710
//
__int64 __fastcall sub_9BF710(__int64 a1, char a2)
{
  __int64 v2; // r15
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // rcx
  const __m128i *v6; // rax
  const __m128i *v7; // r15
  __m128i v9; // xmm2
  __m128i v10; // xmm3
  __int64 v11; // rax
  const __m128i *v12; // r13
  __int64 v13; // r9
  int v14; // ecx
  __int64 v15; // rbx
  int v16; // ecx
  unsigned int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // rdi
  char v20; // al
  unsigned __int64 v21; // rdx
  __int64 v22; // rcx
  unsigned int *v23; // r12
  __int64 v24; // r9
  _QWORD *v25; // r8
  __int64 *v26; // r10
  unsigned int *v27; // r9
  char v28; // r8
  unsigned int **v29; // rax
  unsigned int **v30; // rdx
  int v31; // eax
  __int64 v32; // r12
  __int64 v33; // rdi
  unsigned int v34; // eax
  __int64 v35; // rdx
  int v36; // r8d
  char v37; // al
  __int64 v38; // rax
  int v39; // ecx
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 *v43; // rax
  unsigned int v44; // edx
  __int64 v45; // rbx
  __int64 v46; // r12
  int v47; // eax
  unsigned int v48; // eax
  unsigned int **v49; // rdi
  unsigned int *v50; // r8
  const __m128i *v51; // rbx
  unsigned int *v52; // r13
  int v53; // r12d
  unsigned int v54; // esi
  __int64 *v55; // rax
  __int64 v56; // r10
  __int64 *v57; // rcx
  __int64 *v58; // rax
  __int64 v59; // rdi
  int *v60; // rax
  int v61; // r10d
  int v62; // eax
  int v63; // r8d
  int v64; // eax
  int v65; // r9d
  int v66; // ecx
  unsigned int v67; // edx
  __int64 *v68; // rax
  __int64 v69; // rdi
  __int64 *v70; // r14
  __int64 *v71; // rbx
  _DWORD *v72; // r12
  __int64 v73; // rdx
  __int64 v74; // rdi
  int v75; // r8d
  int *v76; // rax
  int v77; // r10d
  __int64 *v78; // rbx
  __int64 *v79; // r13
  __int64 v80; // r12
  __int64 v81; // rdx
  _QWORD *v82; // rsi
  _QWORD *v83; // rcx
  _QWORD *v84; // rax
  __int64 v85; // rdi
  __int64 v86; // r9
  __int64 v87; // rsi
  unsigned int v88; // eax
  unsigned int v89; // ecx
  unsigned int v90; // edi
  int *v91; // rdx
  int v92; // r11d
  __int64 v93; // rcx
  int v94; // r8d
  __int64 v95; // r10
  int v96; // r8d
  unsigned int v97; // edi
  __int64 *v98; // rdx
  __int64 v99; // r11
  __int64 *v100; // rdi
  __int64 v101; // rsi
  int v103; // edi
  __int64 v104; // r8
  _QWORD *v105; // rax
  int v106; // r8d
  char *v107; // r9
  _QWORD *v108; // rsi
  _QWORD *v109; // rdx
  _QWORD *v110; // rax
  __int64 v111; // rcx
  __int64 v112; // r12
  __int64 v113; // rsi
  unsigned int v114; // eax
  unsigned int v115; // ecx
  unsigned int v116; // r8d
  int *v117; // rdx
  int v118; // r11d
  __int64 v119; // rcx
  int v120; // r9d
  __int64 v121; // r10
  int v122; // r9d
  unsigned int v123; // r8d
  __int64 *v124; // rdx
  __int64 v125; // r11
  __int64 v126; // rdi
  unsigned int **v127; // rax
  _QWORD *v128; // rax
  int v129; // r9d
  bool v130; // zf
  _QWORD *v131; // rax
  int v132; // edx
  int v133; // r10d
  int v134; // edx
  int v135; // r9d
  __int64 v136; // r11
  int v137; // r10d
  unsigned int v138; // r8d
  int *v139; // rax
  int v140; // edi
  int v141; // eax
  int v142; // edx
  int v143; // r10d
  int v144; // edx
  int v145; // r8d
  _QWORD *v146; // rax
  int v147; // edx
  int v148; // edi
  int v149; // r10d
  int v150; // eax
  int v151; // eax
  int v152; // r9d
  __int64 v153; // r11
  int v154; // r9d
  unsigned int v155; // r10d
  __int64 v156; // rcx
  int v157; // eax
  unsigned int v158; // r10d
  __int64 v159; // rsi
  _DWORD *v160; // r8
  _DWORD *v161; // rsi
  int v162; // edx
  _DWORD *v163; // rax
  int v164; // r8d
  unsigned int *v165; // rbx
  int v166; // r8d
  __int64 v167; // r10
  _QWORD *v168; // rcx
  unsigned int v169; // eax
  _QWORD *v170; // rdx
  __int64 v171; // rdi
  unsigned int **v172; // rax
  __int64 v173; // rbx
  int v174; // r11d
  _QWORD *v175; // rax
  __int64 v176; // rdx
  int v177; // r8d
  __int64 v178; // [rsp+18h] [rbp-208h]
  __int64 v179; // [rsp+20h] [rbp-200h]
  __int64 v180; // [rsp+28h] [rbp-1F8h]
  __int64 v181; // [rsp+30h] [rbp-1F0h]
  int v183; // [rsp+44h] [rbp-1DCh]
  __int64 v184; // [rsp+50h] [rbp-1D0h]
  __int64 v185; // [rsp+58h] [rbp-1C8h]
  const __m128i *v186; // [rsp+60h] [rbp-1C0h]
  __int64 v187; // [rsp+68h] [rbp-1B8h]
  __int64 v188; // [rsp+68h] [rbp-1B8h]
  int v189; // [rsp+68h] [rbp-1B8h]
  __int64 v190; // [rsp+70h] [rbp-1B0h]
  char v191; // [rsp+70h] [rbp-1B0h]
  char v192; // [rsp+70h] [rbp-1B0h]
  __int64 v193; // [rsp+70h] [rbp-1B0h]
  int v194; // [rsp+70h] [rbp-1B0h]
  int v195; // [rsp+70h] [rbp-1B0h]
  __int64 v196; // [rsp+78h] [rbp-1A8h]
  unsigned int *v197; // [rsp+88h] [rbp-198h] BYREF
  __int64 v198; // [rsp+90h] [rbp-190h] BYREF
  __int64 v199; // [rsp+98h] [rbp-188h] BYREF
  __m128i v200; // [rsp+A0h] [rbp-180h]
  __m128i v201; // [rsp+B0h] [rbp-170h]
  __m128i v202; // [rsp+C0h] [rbp-160h] BYREF
  __m128i v203; // [rsp+D0h] [rbp-150h]
  __int64 v204; // [rsp+E0h] [rbp-140h] BYREF
  __int64 v205; // [rsp+E8h] [rbp-138h]
  __int64 v206; // [rsp+F0h] [rbp-130h]
  unsigned int v207; // [rsp+F8h] [rbp-128h]
  const __m128i *v208; // [rsp+100h] [rbp-120h]
  __int64 v209; // [rsp+108h] [rbp-118h]
  __int64 v210; // [rsp+110h] [rbp-110h] BYREF
  unsigned int **v211; // [rsp+118h] [rbp-108h]
  __int64 v212; // [rsp+120h] [rbp-100h]
  int v213; // [rsp+128h] [rbp-F8h]
  char v214; // [rsp+12Ch] [rbp-F4h]
  char v215; // [rsp+130h] [rbp-F0h] BYREF
  __int64 v216; // [rsp+150h] [rbp-D0h] BYREF
  __int64 v217; // [rsp+158h] [rbp-C8h]
  __int64 v218; // [rsp+160h] [rbp-C0h]
  __int64 v219; // [rsp+168h] [rbp-B8h]
  __int64 *v220; // [rsp+170h] [rbp-B0h]
  __int64 v221; // [rsp+178h] [rbp-A8h]
  _BYTE v222[32]; // [rsp+180h] [rbp-A0h] BYREF
  __int64 v223; // [rsp+1A0h] [rbp-80h] BYREF
  __int64 v224; // [rsp+1A8h] [rbp-78h]
  __int64 v225; // [rsp+1B0h] [rbp-70h]
  __int64 v226; // [rsp+1B8h] [rbp-68h]
  __int64 *v227; // [rsp+1C0h] [rbp-60h]
  __int64 v228; // [rsp+1C8h] [rbp-58h]
  _BYTE v229[80]; // [rsp+1D0h] [rbp-50h] BYREF

  v2 = a1;
  v3 = *(_QWORD *)(a1 + 32);
  v4 = (__int64)&v204;
  v204 = 0;
  v205 = 0;
  v178 = v3 + 120;
  v206 = 0;
  v207 = 0;
  v208 = (const __m128i *)&v210;
  v209 = 0;
  sub_9BACB0((_QWORD *)a1, (__int64)&v204, v3 + 120);
  if ( !(_DWORD)v209 )
    goto LABEL_130;
  sub_9BBB30(a1);
  v220 = (__int64 *)v222;
  v221 = 0x400000000LL;
  v228 = 0x400000000LL;
  v211 = (unsigned int **)&v215;
  v227 = (__int64 *)v229;
  v216 = 0;
  v6 = (const __m128i *)((char *)v208 + 40 * (unsigned int)v209);
  v217 = 0;
  v218 = 0;
  v219 = 0;
  v223 = 0;
  v224 = 0;
  v225 = 0;
  v226 = 0;
  v210 = 0;
  v212 = 4;
  v213 = 0;
  v214 = 1;
  v186 = v208;
  if ( v6 == v208 )
  {
    v202.m128i_i64[0] = a1;
    v202.m128i_i64[1] = v178;
    goto LABEL_102;
  }
  v7 = (const __m128i *)((char *)v6 - 40);
LABEL_4:
  v9 = _mm_loadu_si128((const __m128i *)&v7->m128i_u64[1]);
  v10 = _mm_loadu_si128((const __m128i *)((char *)v7 + 24));
  v196 = v7->m128i_i64[0];
  v11 = v7->m128i_i64[1];
  v179 = v7[1].m128i_i64[0];
  v180 = v11;
  v181 = v7[1].m128i_i64[1];
  v197 = 0;
  v183 = v11;
  v200 = v9;
  v201 = v10;
  if ( !sub_9BA2B0(v11) )
    goto LABEL_5;
  v4 = *(_QWORD *)(a1 + 8);
  if ( (unsigned __int8)sub_D364B0(*(_QWORD *)(v196 + 40), v4, *(_QWORD *)(a1 + 16)) == 1 && !a2 )
    goto LABEL_5;
  v66 = *(_DWORD *)(a1 + 72);
  v4 = *(_QWORD *)(a1 + 56);
  if ( !v66 )
    goto LABEL_213;
  v5 = (unsigned int)(v66 - 1);
  v67 = v5 & (((unsigned int)v196 >> 9) ^ ((unsigned int)v196 >> 4));
  v68 = (__int64 *)(v4 + 16LL * v67);
  v69 = *v68;
  if ( v196 != *v68 )
  {
    v150 = 1;
    while ( v69 != -4096 )
    {
      v177 = v150 + 1;
      v67 = v5 & (v150 + v67);
      v68 = (__int64 *)(v4 + 16LL * v67);
      v69 = *v68;
      if ( v196 == *v68 )
        goto LABEL_88;
      v150 = v177;
    }
LABEL_213:
    v197 = 0;
    goto LABEL_89;
  }
LABEL_88:
  v197 = (unsigned int *)v68[1];
  if ( v197 )
    goto LABEL_5;
LABEL_89:
  v197 = (unsigned int *)sub_9BEC40(a1, v196, v180, v201.m128i_i8[8]);
  if ( !(unsigned __int8)sub_B46490(v196) )
  {
    v4 = (__int64)&v197;
    sub_9BF1F0((__int64)&v223, (__int64 *)&v197);
LABEL_5:
    if ( v186 == v7 )
      goto LABEL_91;
    goto LABEL_6;
  }
  v4 = (__int64)&v197;
  sub_9BF1F0((__int64)&v216, (__int64 *)&v197);
  if ( v186 != v7 )
  {
LABEL_6:
    v12 = v7;
    while ( 1 )
    {
      v13 = v12[-3].m128i_i64[1];
      v14 = *(_DWORD *)(a1 + 72);
      v4 = *(_QWORD *)(a1 + 56);
      v198 = v13;
      v202 = _mm_loadu_si128(v12 - 2);
      v203 = _mm_loadu_si128(v12 - 1);
      v15 = v12[-2].m128i_i64[0];
      v184 = v12[-2].m128i_i64[1];
      v185 = v12[-1].m128i_i64[0];
      if ( !v14 )
        goto LABEL_59;
      v16 = v14 - 1;
      v17 = v16 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v18 = (__int64 *)(v4 + 16LL * v17);
      v19 = *v18;
      if ( v13 != *v18 )
        break;
LABEL_9:
      v190 = v18[1];
LABEL_10:
      v187 = v13;
      v199 = v190;
      v20 = sub_B46490(v13);
      v23 = v197;
      v24 = v187;
      if ( v20 && (unsigned int *)v190 != v197 )
      {
        if ( v197 )
        {
          if ( (_DWORD)v225 )
          {
            v22 = v224;
            v4 = v224 + 8LL * (unsigned int)v226;
            if ( (_DWORD)v226 )
            {
              v21 = (unsigned int)(v226 - 1);
              v48 = v21 & (((unsigned int)v197 >> 9) ^ ((unsigned int)v197 >> 4));
              v49 = (unsigned int **)(v224 + 8LL * v48);
              v50 = *v49;
              if ( *v49 == v197 )
              {
LABEL_62:
                if ( (unsigned int **)v4 != v49 )
                {
LABEL_63:
                  if ( !*v23 )
                    goto LABEL_32;
                  v188 = v15;
                  v51 = v12;
                  v52 = v23;
                  v53 = 0;
                  while ( 2 )
                  {
                    v21 = v52[8];
                    v59 = *((_QWORD *)v52 + 2);
                    v22 = v53 + v52[10];
                    if ( !(_DWORD)v21 )
                      goto LABEL_69;
                    v21 = (unsigned int)(v21 - 1);
                    v4 = (unsigned int)v21 & (37 * (_DWORD)v22);
                    v60 = (int *)(v59 + 16 * v4);
                    v61 = *v60;
                    if ( (_DWORD)v22 != *v60 )
                    {
                      v62 = 1;
                      while ( v61 != 0x7FFFFFFF )
                      {
                        v63 = v62 + 1;
                        v4 = (unsigned int)v21 & (v62 + (_DWORD)v4);
                        v60 = (int *)(v59 + 16LL * (unsigned int)v4);
                        v61 = *v60;
                        if ( (_DWORD)v22 == *v60 )
                          goto LABEL_72;
                        v62 = v63;
                      }
                      goto LABEL_69;
                    }
LABEL_72:
                    v22 = *((_QWORD *)v60 + 1);
                    if ( !v22 )
                      goto LABEL_69;
                    if ( !v207 )
                      goto LABEL_74;
                    v54 = (v207 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
                    v55 = (__int64 *)(v205 + 16LL * v54);
                    v56 = *v55;
                    if ( v22 == *v55 )
                    {
LABEL_66:
                      v57 = (__int64 *)v208;
                      if ( v55 != (__int64 *)(v205 + 16LL * v207) )
                      {
                        v58 = &v208->m128i_i64[5 * *((unsigned int *)v55 + 2)];
LABEL_68:
                        v4 = v51[-3].m128i_i64[1];
                        if ( !(unsigned __int8)sub_9BA2D0(a1, v4, v51[-2].m128i_i64[0], *v58, v58[1]) )
                        {
                          v12 = v51;
                          v15 = v188;
                          goto LABEL_17;
                        }
LABEL_69:
                        if ( *v52 <= ++v53 )
                        {
                          v12 = v51;
                          v15 = v188;
                          goto LABEL_31;
                        }
                        continue;
                      }
                    }
                    else
                    {
                      v64 = 1;
                      while ( v56 != -4096 )
                      {
                        v65 = v64 + 1;
                        v54 = (v207 - 1) & (v64 + v54);
                        v55 = (__int64 *)(v205 + 16LL * v54);
                        v56 = *v55;
                        if ( v22 == *v55 )
                          goto LABEL_66;
                        v64 = v65;
                      }
LABEL_74:
                      v57 = (__int64 *)v208;
                    }
                    break;
                  }
                  v58 = &v57[5 * (unsigned int)v209];
                  goto LABEL_68;
                }
              }
              else
              {
                v148 = 1;
                while ( v50 != (unsigned int *)-4096LL )
                {
                  v149 = v148 + 1;
                  v48 = v21 & (v48 + v148);
                  v49 = (unsigned int **)(v224 + 8LL * v48);
                  v50 = *v49;
                  if ( *v49 == v197 )
                    goto LABEL_62;
                  v148 = v149;
                }
              }
            }
          }
          else
          {
            v4 = (__int64)&v227[(unsigned int)v228];
            if ( (_QWORD *)v4 != sub_9B6740(v227, v4, (__int64 *)&v197) )
              goto LABEL_63;
          }
        }
        v4 = v24;
        if ( (unsigned __int8)sub_9BA2D0(a1, v24, v15, v7->m128i_i64[0], v7->m128i_i64[1]) != 1 && v196 )
        {
LABEL_17:
          if ( !v190 )
            goto LABEL_20;
          if ( (_DWORD)v218 )
          {
            v4 = v217;
            if ( !(_DWORD)v219 )
              goto LABEL_20;
            v103 = v219 - 1;
            v21 = ((_DWORD)v219 - 1) & (((unsigned int)v190 >> 9) ^ ((unsigned int)v190 >> 4));
            v22 = v217 + 8 * v21;
            v104 = *(_QWORD *)v22;
            if ( v190 != *(_QWORD *)v22 )
            {
              v153 = *(_QWORD *)v22;
              v154 = v21;
              v22 = 1;
              while ( v153 != -4096 )
              {
                v155 = v22 + 1;
                v156 = v103 & (unsigned int)(v154 + v22);
                v154 = v156;
                v22 = v217 + 8 * v156;
                v153 = *(_QWORD *)v22;
                if ( v190 == *(_QWORD *)v22 )
                {
                  if ( v22 == v217 + 8LL * (unsigned int)v219 )
                    goto LABEL_20;
                  v157 = 1;
                  while ( v104 != -4096 )
                  {
                    v21 = v103 & (unsigned int)(v157 + v21);
                    v22 = v217 + 8LL * (unsigned int)v21;
                    v104 = *(_QWORD *)v22;
                    if ( v190 == *(_QWORD *)v22 )
                      goto LABEL_137;
                    ++v157;
                  }
                  goto LABEL_140;
                }
                v22 = v155;
              }
              goto LABEL_20;
            }
            if ( v22 == v217 + 8LL * (unsigned int)v219 )
              goto LABEL_20;
LABEL_137:
            *(_QWORD *)v22 = -8192;
            LODWORD(v218) = v218 - 1;
            ++HIDWORD(v218);
            v105 = sub_9B6800(v220, (__int64)&v220[(unsigned int)v221], &v199);
            if ( v105 + 1 != (_QWORD *)v107 )
            {
              memmove(v105, v105 + 1, v107 - (char *)(v105 + 1));
              v106 = v221;
              v190 = v199;
            }
            LODWORD(v221) = v106 - 1;
          }
          else
          {
            v4 = (__int64)&v220[(unsigned int)v221];
            if ( (_QWORD *)v4 == sub_9B6740(v220, v4, &v199) )
              goto LABEL_20;
            v128 = sub_9B6800(v25, v4, v26);
            if ( (_QWORD *)v4 != v128 )
            {
              if ( (_QWORD *)v4 != v128 + 1 )
              {
                memmove(v128, v128 + 1, v4 - (_QWORD)(v128 + 1));
                v129 = v221;
                v190 = v199;
              }
              v130 = *(_BYTE *)(a1 + 108) == 0;
              LODWORD(v221) = v129 - 1;
              if ( v130 )
              {
LABEL_169:
                v131 = (_QWORD *)sub_C8CA60(a1 + 80, v190, v21, v22);
                if ( v131 )
                {
                  *v131 = -2;
                  ++*(_DWORD *)(a1 + 104);
                  ++*(_QWORD *)(a1 + 80);
                }
                goto LABEL_146;
              }
LABEL_141:
              v108 = *(_QWORD **)(a1 + 88);
              v109 = &v108[*(unsigned int *)(a1 + 100)];
              v110 = v108;
              if ( v108 != v109 )
              {
                while ( *v110 != v190 )
                {
                  if ( v109 == ++v110 )
                    goto LABEL_146;
                }
                v111 = (unsigned int)(*(_DWORD *)(a1 + 100) - 1);
                *(_DWORD *)(a1 + 100) = v111;
                *v110 = v108[v111];
                ++*(_QWORD *)(a1 + 80);
              }
LABEL_146:
              v112 = *(_QWORD *)(v190 + 16);
              v113 = *(unsigned int *)(v190 + 32);
              if ( *(_DWORD *)v190 )
              {
                v114 = 0;
                do
                {
                  v115 = v114 + *(_DWORD *)(v190 + 40);
                  if ( (_DWORD)v113 )
                  {
                    v116 = (v113 - 1) & (37 * v115);
                    v117 = (int *)(v112 + 16LL * v116);
                    v118 = *v117;
                    if ( v115 == *v117 )
                    {
LABEL_150:
                      v119 = *((_QWORD *)v117 + 1);
                      if ( v119 )
                      {
                        v120 = *(_DWORD *)(a1 + 72);
                        v121 = *(_QWORD *)(a1 + 56);
                        if ( v120 )
                        {
                          v122 = v120 - 1;
                          v123 = v122 & (((unsigned int)v119 >> 9) ^ ((unsigned int)v119 >> 4));
                          v124 = (__int64 *)(v121 + 16LL * v123);
                          v125 = *v124;
                          if ( v119 == *v124 )
                          {
LABEL_153:
                            *v124 = -8192;
                            --*(_DWORD *)(a1 + 64);
                            ++*(_DWORD *)(a1 + 68);
                            v113 = *(unsigned int *)(v190 + 32);
                            v112 = *(_QWORD *)(v190 + 16);
                          }
                          else
                          {
                            v147 = 1;
                            while ( v125 != -4096 )
                            {
                              v123 = v122 & (v147 + v123);
                              v189 = v147 + 1;
                              v124 = (__int64 *)(v121 + 16LL * v123);
                              v125 = *v124;
                              if ( v119 == *v124 )
                                goto LABEL_153;
                              v147 = v189;
                            }
                          }
                        }
                      }
                    }
                    else
                    {
                      v132 = 1;
                      while ( v118 != 0x7FFFFFFF )
                      {
                        v133 = v132 + 1;
                        v116 = (v113 - 1) & (v132 + v116);
                        v117 = (int *)(v112 + 16LL * v116);
                        v118 = *v117;
                        if ( v115 == *v117 )
                          goto LABEL_150;
                        v132 = v133;
                      }
                    }
                  }
                  ++v114;
                }
                while ( *(_DWORD *)v190 > v114 );
              }
              sub_C7D6A0(v112, 16 * v113, 8);
              v4 = 56;
              j_j___libc_free_0(v190, 56);
LABEL_20:
              v27 = v197;
              v28 = v214;
              v23 = v197;
              if ( v197 )
              {
                if ( (_DWORD)v225 )
                {
                  v21 = (unsigned int)v226;
                  v22 = v224;
                  if ( (_DWORD)v226 )
                  {
                    v21 = ((_DWORD)v226 - 1) & (((unsigned int)v197 >> 9) ^ ((unsigned int)v197 >> 4));
                    v4 = v224 + 8 * v21;
                    v126 = *(_QWORD *)v4;
                    if ( v197 == *(unsigned int **)v4 )
                    {
LABEL_158:
                      if ( v224 + 8LL * (unsigned int)v226 != v4 )
                        goto LABEL_159;
                    }
                    else
                    {
                      v4 = 1;
                      while ( v126 != -4096 )
                      {
                        v158 = v4 + 1;
                        v159 = ((_DWORD)v226 - 1) & (unsigned int)(v21 + v4);
                        v21 = (unsigned int)v159;
                        v4 = v224 + 8 * v159;
                        v126 = *(_QWORD *)v4;
                        if ( v197 == *(unsigned int **)v4 )
                          goto LABEL_158;
                        v4 = v158;
                      }
                    }
                  }
                }
                else
                {
                  v4 = (__int64)&v227[(unsigned int)v228];
                  if ( (_QWORD *)v4 == sub_9B6740(v227, v4, (__int64 *)&v197) )
                    goto LABEL_23;
LABEL_159:
                  if ( v28 )
                  {
                    v127 = v211;
                    v22 = HIDWORD(v212);
                    v21 = (unsigned __int64)&v211[HIDWORD(v212)];
                    if ( v211 == (unsigned int **)v21 )
                    {
LABEL_190:
                      if ( HIDWORD(v212) >= (unsigned int)v212 )
                        goto LABEL_191;
                      v22 = (unsigned int)++HIDWORD(v212);
                      *(_QWORD *)v21 = v27;
                      ++v210;
                      v28 = v214;
                      v23 = v197;
                    }
                    else
                    {
                      while ( v27 != *v127 )
                      {
                        if ( (unsigned int **)v21 == ++v127 )
                          goto LABEL_190;
                      }
                      v23 = v27;
                    }
                  }
                  else
                  {
LABEL_191:
                    v4 = (__int64)v27;
                    sub_C8CC70(&v210, v27);
                    v28 = v214;
                    v23 = v197;
                  }
                }
              }
LABEL_23:
              if ( !v28 )
                goto LABEL_33;
              goto LABEL_24;
            }
          }
LABEL_140:
          if ( !*(_BYTE *)(a1 + 108) )
            goto LABEL_169;
          goto LABEL_141;
        }
LABEL_31:
        v23 = v197;
      }
LABEL_32:
      if ( !v214 )
      {
LABEL_33:
        v4 = (__int64)v23;
        if ( sub_C8CA60(&v210, v23, v21, v22) )
          goto LABEL_28;
        goto LABEL_34;
      }
LABEL_24:
      v29 = v211;
      v30 = &v211[HIDWORD(v212)];
      if ( v211 != v30 )
      {
        while ( *v29 != v23 )
        {
          if ( v30 == ++v29 )
            goto LABEL_34;
        }
        goto LABEL_28;
      }
LABEL_34:
      if ( !sub_9BA2B0(v15) || !sub_9BA2B0(v183) )
        goto LABEL_28;
      v31 = *(_DWORD *)(a1 + 72);
      v32 = v198;
      v33 = *(_QWORD *)(a1 + 56);
      if ( v31 )
      {
        v4 = (unsigned int)(v31 - 1);
        v34 = v4 & (((unsigned int)v198 >> 9) ^ ((unsigned int)v198 >> 4));
        v35 = *(_QWORD *)(v33 + 16LL * v34);
        if ( v198 == v35 )
          goto LABEL_28;
        v36 = 1;
        while ( v35 != -4096 )
        {
          v34 = v4 & (v36 + v34);
          v35 = *(_QWORD *)(v33 + 16LL * v34);
          if ( v198 == v35 )
            goto LABEL_28;
          ++v36;
        }
      }
      v191 = sub_B46420(v198);
      if ( v191 != (unsigned __int8)sub_B46420(v196) )
        goto LABEL_28;
      v192 = sub_B46490(v32);
      v37 = sub_B46490(v196);
      v4 = v181;
      LOBYTE(v4) = v15 != v180 || v185 != v181;
      if ( (_BYTE)v4 || v192 != v37 )
        goto LABEL_28;
      v38 = *(_QWORD *)(*(_QWORD *)(v32 - 32) + 8LL);
      if ( (unsigned int)*(unsigned __int8 *)(v38 + 8) - 17 <= 1 )
        v38 = **(_QWORD **)(v38 + 16);
      v39 = *(_DWORD *)(v38 + 8) >> 8;
      v40 = *(_QWORD *)(*(_QWORD *)(v196 - 32) + 8LL);
      if ( (unsigned int)*(unsigned __int8 *)(v40 + 8) - 17 <= 1 )
        v40 = **(_QWORD **)(v40 + 16);
      if ( *(_DWORD *)(v40 + 8) >> 8 != v39 )
        goto LABEL_28;
      v4 = v184;
      v41 = sub_DCC810(*(_QWORD *)(*(_QWORD *)a1 + 112LL), v184, v179, 0, 0);
      if ( *(_WORD *)(v41 + 24) )
        goto LABEL_28;
      v42 = *(_QWORD *)(v41 + 32);
      v43 = *(__int64 **)(v42 + 24);
      v44 = *(_DWORD *)(v42 + 32);
      if ( v44 > 0x40 )
      {
        v45 = *v43;
        goto LABEL_51;
      }
      if ( v44 )
      {
        v45 = (__int64)((_QWORD)v43 << (64 - (unsigned __int8)v44)) >> (64 - (unsigned __int8)v44);
LABEL_51:
        if ( v45 % v181 )
          goto LABEL_28;
        goto LABEL_52;
      }
      v45 = 0;
LABEL_52:
      v4 = *(_QWORD *)(a1 + 8);
      v193 = *(_QWORD *)(v198 + 40);
      v46 = *(_QWORD *)(v196 + 40);
      if ( !(unsigned __int8)sub_D364B0(v193, v4, *(_QWORD *)(a1 + 16))
        && (v4 = *(_QWORD *)(a1 + 8), !(unsigned __int8)sub_D364B0(v46, v4, *(_QWORD *)(a1 + 16)))
        || a2 == 1 && v46 == v193 )
      {
        if ( !v197[6] )
          goto LABEL_56;
        v160 = (_DWORD *)*((_QWORD *)v197 + 2);
        v161 = &v160[4 * v197[8]];
        if ( v160 == v161 )
          goto LABEL_56;
        while ( 1 )
        {
          v162 = *v160;
          v163 = v160;
          if ( (unsigned int)(*v160 + 0x7FFFFFFF) <= 0xFFFFFFFD )
            break;
          v160 += 4;
          if ( v161 == v160 )
            BUG();
        }
        if ( v160 == v161 )
LABEL_56:
          BUG();
        while ( v196 != *((_QWORD *)v163 + 1) )
        {
          v163 += 4;
          if ( v163 == v161 )
            goto LABEL_56;
          while ( 1 )
          {
            v162 = *v163;
            if ( (unsigned int)(*v163 + 0x7FFFFFFF) <= 0xFFFFFFFD )
              break;
            v163 += 4;
            if ( v161 == v163 )
              BUG();
          }
          if ( v161 == v163 )
            BUG();
        }
        v4 = v198;
        if ( (unsigned __int8)sub_9BE8F0(v197, v198, v162 - v197[10] + (unsigned int)(v45 / v181), v203.m128i_u8[8]) )
        {
          v164 = *(_DWORD *)(a1 + 72);
          v165 = v197;
          if ( v164 )
          {
            v166 = v164 - 1;
            v167 = *(_QWORD *)(a1 + 56);
            v4 = 1;
            v168 = 0;
            v169 = v166 & (((unsigned int)v198 >> 9) ^ ((unsigned int)v198 >> 4));
            v170 = (_QWORD *)(v167 + 16LL * v169);
            v171 = *v170;
            if ( *v170 == v198 )
            {
LABEL_253:
              v172 = (unsigned int **)(v170 + 1);
              goto LABEL_254;
            }
            while ( v171 != -4096 )
            {
              if ( !v168 && v171 == -8192 )
                v168 = v170;
              v174 = v4 + 1;
              v4 = v169 + (unsigned int)v4;
              v169 = v166 & v4;
              v170 = (_QWORD *)(v167 + 16LL * (v166 & (unsigned int)v4));
              v171 = *v170;
              if ( v198 == *v170 )
                goto LABEL_253;
              LODWORD(v4) = v174;
            }
            if ( v168 )
              v170 = v168;
          }
          else
          {
            v170 = 0;
          }
          v4 = (__int64)&v198;
          v175 = sub_9BB740(a1 + 48, &v198, v170);
          v176 = v198;
          v175[1] = 0;
          v172 = (unsigned int **)(v175 + 1);
          *(v172 - 1) = (unsigned int *)v176;
LABEL_254:
          *v172 = v165;
          v173 = v198;
          if ( (unsigned __int8)sub_B46420(v198) )
            *((_QWORD *)v197 + 6) = v173;
        }
      }
LABEL_28:
      v12 = (const __m128i *)((char *)v12 - 40);
      if ( v186 == v12 )
      {
        v7 = (const __m128i *)((char *)v7 - 40);
        goto LABEL_4;
      }
    }
    v47 = 1;
    while ( v19 != -4096 )
    {
      v145 = v47 + 1;
      v17 = v16 & (v47 + v17);
      v18 = (__int64 *)(v4 + 16LL * v17);
      v19 = *v18;
      if ( v13 == *v18 )
        goto LABEL_9;
      v47 = v145;
    }
LABEL_59:
    v190 = 0;
    goto LABEL_10;
  }
LABEL_91:
  v2 = a1;
  v70 = v227;
  v202.m128i_i64[0] = v2;
  v71 = &v227[(unsigned int)v228];
  v202.m128i_i64[1] = v178;
  if ( v227 != v71 )
  {
    do
    {
      while ( 1 )
      {
        v72 = (_DWORD *)*v70;
        if ( *(_DWORD *)*v70 == *(_DWORD *)(*v70 + 24) )
          goto LABEL_94;
        v4 = *v70;
        if ( (unsigned __int8)sub_9B6FD0(&v202, *v70, 0) )
          goto LABEL_94;
        v73 = (unsigned int)v72[8];
        v74 = *((_QWORD *)v72 + 2);
        v75 = *v72 - 1;
        v5 = (unsigned int)(v75 + v72[10]);
        if ( (_DWORD)v73 )
          break;
LABEL_100:
        if ( *((_BYTE *)v72 + 4) )
        {
          v4 = (__int64)v72;
          sub_9B72A0(v2, (__int64)v72, v73, v5);
          goto LABEL_94;
        }
        ++v70;
        *(_BYTE *)(v2 + 40) = 1;
        if ( v71 == v70 )
          goto LABEL_102;
      }
      v73 = (unsigned int)(v73 - 1);
      v4 = (unsigned int)v73 & (37 * (_DWORD)v5);
      v76 = (int *)(v74 + 16 * v4);
      v77 = *v76;
      if ( *v76 != (_DWORD)v5 )
      {
        v151 = 1;
        while ( v77 != 0x7FFFFFFF )
        {
          v152 = v151 + 1;
          v4 = (unsigned int)v73 & (v151 + (_DWORD)v4);
          v76 = (int *)(v74 + 16LL * (unsigned int)v4);
          v77 = *v76;
          if ( (_DWORD)v5 == *v76 )
            goto LABEL_99;
          v151 = v152;
        }
        goto LABEL_100;
      }
LABEL_99:
      if ( !*((_QWORD *)v76 + 1) )
        goto LABEL_100;
      v4 = (__int64)v72;
      sub_9B6FD0(&v202, (__int64)v72, v75);
LABEL_94:
      ++v70;
    }
    while ( v71 != v70 );
  }
LABEL_102:
  v78 = v220;
  v79 = &v220[(unsigned int)v221];
  if ( v79 != v220 )
  {
    do
    {
      while ( 1 )
      {
        v80 = *v78;
        v81 = *(unsigned int *)*v78;
        if ( (_DWORD)v81 == *(_DWORD *)(*v78 + 24) )
          goto LABEL_105;
        if ( a2 )
          break;
        if ( *(_BYTE *)(v2 + 108) )
        {
          v82 = *(_QWORD **)(v2 + 88);
          v83 = &v82[*(unsigned int *)(v2 + 100)];
          if ( v82 != v83 )
          {
            v84 = *(_QWORD **)(v2 + 88);
            while ( v80 != *v84 )
            {
              if ( v83 == ++v84 )
                goto LABEL_114;
            }
            v85 = (unsigned int)(*(_DWORD *)(v2 + 100) - 1);
            *(_DWORD *)(v2 + 100) = v85;
            *v84 = v82[v85];
            ++*(_QWORD *)(v2 + 80);
            LODWORD(v81) = *(_DWORD *)v80;
          }
        }
        else
        {
          v146 = (_QWORD *)sub_C8CA60(v2 + 80, *v78, v81, v5);
          if ( v146 )
          {
            *v146 = -2;
            ++*(_DWORD *)(v2 + 104);
            ++*(_QWORD *)(v2 + 80);
          }
          LODWORD(v81) = *(_DWORD *)v80;
        }
LABEL_114:
        v86 = *(_QWORD *)(v80 + 16);
        v87 = *(unsigned int *)(v80 + 32);
        if ( (_DWORD)v81 )
        {
          v88 = 0;
          do
          {
            v89 = v88 + *(_DWORD *)(v80 + 40);
            if ( (_DWORD)v87 )
            {
              v90 = (v87 - 1) & (37 * v89);
              v91 = (int *)(v86 + 16LL * v90);
              v92 = *v91;
              if ( *v91 == v89 )
              {
LABEL_118:
                v93 = *((_QWORD *)v91 + 1);
                if ( v93 )
                {
                  v94 = *(_DWORD *)(v2 + 72);
                  v95 = *(_QWORD *)(v2 + 56);
                  if ( v94 )
                  {
                    v96 = v94 - 1;
                    v97 = v96 & (((unsigned int)v93 >> 9) ^ ((unsigned int)v93 >> 4));
                    v98 = (__int64 *)(v95 + 16LL * v97);
                    v99 = *v98;
                    if ( *v98 == v93 )
                    {
LABEL_121:
                      *v98 = -8192;
                      --*(_DWORD *)(v2 + 64);
                      ++*(_DWORD *)(v2 + 68);
                      v87 = *(unsigned int *)(v80 + 32);
                      v86 = *(_QWORD *)(v80 + 16);
                    }
                    else
                    {
                      v144 = 1;
                      while ( v99 != -4096 )
                      {
                        v97 = v96 & (v144 + v97);
                        v195 = v144 + 1;
                        v98 = (__int64 *)(v95 + 16LL * v97);
                        v99 = *v98;
                        if ( v93 == *v98 )
                          goto LABEL_121;
                        v144 = v195;
                      }
                    }
                  }
                }
              }
              else
              {
                v142 = 1;
                while ( v92 != 0x7FFFFFFF )
                {
                  v143 = v142 + 1;
                  v90 = (v87 - 1) & (v142 + v90);
                  v91 = (int *)(v86 + 16LL * v90);
                  v92 = *v91;
                  if ( v89 == *v91 )
                    goto LABEL_118;
                  v142 = v143;
                }
              }
            }
            ++v88;
          }
          while ( *(_DWORD *)v80 > v88 );
        }
        ++v78;
        sub_C7D6A0(v86, 16 * v87, 8);
        v4 = 56;
        j_j___libc_free_0(v80, 56);
        if ( v79 == v78 )
          goto LABEL_124;
      }
      v4 = *v78;
      if ( !(unsigned __int8)sub_9B6FD0(&v202, *v78, 0) )
      {
        v134 = *(_DWORD *)v80 - 1;
        if ( v134 > 0 )
        {
          v135 = *(_DWORD *)(v80 + 32);
          v136 = *(_QWORD *)(v80 + 16);
          LODWORD(v4) = 37 * (*(_DWORD *)(v80 + 40) + *(_DWORD *)v80) - 37;
          v137 = v135 - 1;
          LODWORD(v5) = v134 + *(_DWORD *)(v80 + 40);
          do
          {
            if ( v135 )
            {
              v138 = v137 & v4;
              v139 = (int *)(v136 + 16LL * (v137 & (unsigned int)v4));
              v140 = *v139;
              if ( *v139 == (_DWORD)v5 )
              {
LABEL_177:
                if ( *((_QWORD *)v139 + 1) )
                {
                  v4 = v80;
                  sub_9B6FD0(&v202, v80, v134);
                  break;
                }
              }
              else
              {
                v141 = 1;
                while ( v140 != 0x7FFFFFFF )
                {
                  v138 = v137 & (v141 + v138);
                  v194 = v141 + 1;
                  v139 = (int *)(v136 + 16LL * v138);
                  v140 = *v139;
                  if ( (_DWORD)v5 == *v139 )
                    goto LABEL_177;
                  v141 = v194;
                }
              }
            }
            v4 = (unsigned int)(v4 - 37);
            v5 = (unsigned int)(v5 - 1);
            --v134;
          }
          while ( v134 );
        }
      }
LABEL_105:
      ++v78;
    }
    while ( v79 != v78 );
  }
LABEL_124:
  if ( v214 )
  {
    v100 = v227;
    if ( v227 != (__int64 *)v229 )
      goto LABEL_126;
  }
  else
  {
    _libc_free(v211, v4);
    v100 = v227;
    if ( v227 != (__int64 *)v229 )
LABEL_126:
      _libc_free(v100, v4);
  }
  v101 = 8LL * (unsigned int)v226;
  sub_C7D6A0(v224, v101, 8);
  if ( v220 != (__int64 *)v222 )
    _libc_free(v220, v101);
  v4 = 8LL * (unsigned int)v219;
  sub_C7D6A0(v217, v4, 8);
LABEL_130:
  if ( v208 != (const __m128i *)&v210 )
    _libc_free(v208, v4);
  return sub_C7D6A0(v205, 16LL * v207, 8);
}
