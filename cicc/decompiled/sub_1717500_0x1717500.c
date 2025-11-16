// Function: sub_1717500
// Address: 0x1717500
//
__int64 __fastcall sub_1717500(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128i a13,
        __m128 a14,
        __int64 a15,
        char a16,
        char a17,
        __int64 a18)
{
  __int64 v20; // rax
  __m128i v21; // xmm1
  __m128 v22; // xmm0
  __int64 v23; // rax
  __int64 v24; // r9
  __m128i v25; // xmm2
  unsigned __int32 v26; // ebx
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 *v29; // rsi
  __int64 *v30; // rax
  _QWORD *v31; // rcx
  __int64 v32; // r8
  char v33; // dl
  __int64 *v34; // rsi
  __int64 *v35; // rcx
  __int64 *v36; // rbx
  __int64 v37; // rax
  int v38; // r8d
  int v39; // r9d
  double v40; // xmm4_8
  double v41; // xmm5_8
  __int64 *v42; // rcx
  __int64 v43; // r11
  __int64 *v44; // r13
  __int64 v45; // rdi
  char v46; // al
  char v47; // bl
  int v48; // edx
  __int64 v49; // rax
  __int64 *v50; // r13
  __int64 v51; // r12
  char v52; // al
  unsigned int v53; // edi
  __int64 *v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rdx
  __int64 v57; // rsi
  unsigned __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rax
  unsigned __int64 v61; // r14
  char v62; // al
  __int64 v63; // rax
  unsigned int v64; // r15d
  int v65; // r12d
  __int64 v66; // rbx
  __int64 v67; // rax
  int v68; // r10d
  __int64 *v69; // r15
  int v70; // eax
  __int64 v71; // rax
  unsigned int v72; // edx
  __int64 v73; // rdi
  int v74; // r10d
  __int64 *v75; // r9
  __int64 *v76; // r8
  unsigned int v77; // r14d
  int v78; // r9d
  __int64 v79; // rsi
  __int64 v80; // rax
  __int64 *v81; // rcx
  __int64 v82; // rsi
  __int64 v83; // r11
  __int64 v84; // rax
  unsigned int v85; // edx
  __int64 v86; // rdi
  __int64 v87; // rdi
  __int64 v88; // rdi
  __int64 v89; // rdi
  __int64 v90; // rdx
  unsigned __int64 v91; // r14
  __int64 v92; // rbx
  __int64 v93; // rax
  __int64 v94; // r12
  __int64 v95; // r11
  unsigned __int64 v96; // rdx
  __int32 v97; // r13d
  __int64 v98; // rdx
  unsigned __int64 v99; // rcx
  unsigned int v100; // esi
  __int64 v101; // r13
  __int64 v102; // r8
  unsigned int v103; // edi
  _QWORD *v104; // rax
  __int64 v105; // rcx
  __int64 v106; // rax
  unsigned int v107; // esi
  int v108; // r9d
  __int64 v109; // r14
  int v110; // r15d
  int v111; // r15d
  __int64 v112; // r10
  unsigned int v113; // ecx
  int v114; // eax
  _QWORD *v115; // rdx
  __int64 v116; // r14
  __int16 *v117; // r15
  __int64 v118; // rdx
  __int64 v119; // rcx
  char v120; // r14
  void *v121; // rax
  __int64 v122; // rax
  __int64 v123; // rdx
  __int64 *v124; // rdi
  __int64 v125; // rax
  __int64 v126; // r14
  _QWORD *v127; // rax
  __int64 v128; // r15
  double v129; // xmm4_8
  double v130; // xmm5_8
  unsigned __int64 v131; // rdi
  __int64 v132; // r13
  __int64 *v133; // r15
  __int64 *v134; // rax
  int v135; // eax
  __int64 v136; // r14
  char v137; // al
  double v138; // xmm4_8
  double v139; // xmm5_8
  _BYTE *v140; // r13
  unsigned __int64 v141; // r12
  __int64 v142; // rax
  int v143; // eax
  int v144; // eax
  int v145; // r10d
  _QWORD *v146; // rcx
  unsigned int v147; // r15d
  int v148; // esi
  __int64 v149; // rdi
  __int64 v150; // rax
  __int64 v151; // rdx
  __int64 v152; // rax
  __int64 v153; // rdx
  __int64 v154; // r12
  __int64 v155; // rbx
  __int64 *v156; // rdx
  __int64 v157; // rdx
  _BYTE *v158; // r14
  unsigned __int64 v159; // r13
  __int64 v160; // rax
  __int64 v162; // rdx
  __int64 v163; // rdx
  __int64 v164; // rdx
  int v165; // edi
  _QWORD *v166; // rsi
  __int64 v172; // [rsp+60h] [rbp-1140h]
  __int64 v173; // [rsp+68h] [rbp-1138h]
  __int64 *v174; // [rsp+68h] [rbp-1138h]
  __int64 v175; // [rsp+68h] [rbp-1138h]
  __int64 v176; // [rsp+70h] [rbp-1130h]
  __int64 v177; // [rsp+70h] [rbp-1130h]
  __int64 v178; // [rsp+70h] [rbp-1130h]
  _BYTE *v179; // [rsp+78h] [rbp-1128h]
  __int64 v180; // [rsp+80h] [rbp-1120h]
  __int64 v181; // [rsp+80h] [rbp-1120h]
  __int64 v182; // [rsp+80h] [rbp-1120h]
  __int64 v183; // [rsp+80h] [rbp-1120h]
  __int64 v184; // [rsp+80h] [rbp-1120h]
  int v185; // [rsp+88h] [rbp-1118h]
  char v186; // [rsp+8Dh] [rbp-1113h]
  char v187; // [rsp+8Eh] [rbp-1112h]
  char v188; // [rsp+8Fh] [rbp-1111h]
  __int64 v189; // [rsp+90h] [rbp-1110h]
  __int64 v190; // [rsp+98h] [rbp-1108h]
  __int64 v191; // [rsp+98h] [rbp-1108h]
  int v192; // [rsp+98h] [rbp-1108h]
  __int64 v193; // [rsp+98h] [rbp-1108h]
  int v194; // [rsp+98h] [rbp-1108h]
  int v195; // [rsp+98h] [rbp-1108h]
  __int64 v196; // [rsp+98h] [rbp-1108h]
  __int64 v197[4]; // [rsp+A0h] [rbp-1100h] BYREF
  __int64 v198; // [rsp+C0h] [rbp-10E0h] BYREF
  __int64 v199; // [rsp+C8h] [rbp-10D8h]
  __int64 v200; // [rsp+D0h] [rbp-10D0h]
  unsigned int v201; // [rsp+D8h] [rbp-10C8h]
  __int64 v202; // [rsp+E0h] [rbp-10C0h] BYREF
  void *v203; // [rsp+E8h] [rbp-10B8h] BYREF
  __int64 v204; // [rsp+F0h] [rbp-10B0h]
  __int64 v205[5]; // [rsp+100h] [rbp-10A0h] BYREF
  int v206; // [rsp+128h] [rbp-1078h]
  __int64 v207; // [rsp+130h] [rbp-1070h]
  __int64 v208; // [rsp+138h] [rbp-1068h]
  __m128i v209; // [rsp+140h] [rbp-1060h] BYREF
  __int64 (__fastcall *v210)(const __m128i **, const __m128i *, int); // [rsp+150h] [rbp-1050h]
  void (__fastcall *v211)(__int64 *, __int64 *); // [rsp+158h] [rbp-1048h]
  _BYTE *v212; // [rsp+160h] [rbp-1040h]
  __int64 v213; // [rsp+170h] [rbp-1030h] BYREF
  __int64 *v214; // [rsp+178h] [rbp-1028h]
  __int64 *v215; // [rsp+180h] [rbp-1020h]
  __int64 v216; // [rsp+188h] [rbp-1018h]
  int v217; // [rsp+190h] [rbp-1010h]
  _BYTE v218[264]; // [rsp+198h] [rbp-1008h] BYREF
  __m128i v219; // [rsp+2A0h] [rbp-F00h] BYREF
  _QWORD v220[128]; // [rsp+2B0h] [rbp-EF0h] BYREF
  __m128i v221; // [rsp+6B0h] [rbp-AF0h] BYREF
  _QWORD v222[2]; // [rsp+6C0h] [rbp-AE0h] BYREF
  __int64 v223; // [rsp+6D0h] [rbp-AD0h]
  __int64 v224; // [rsp+6D8h] [rbp-AC8h]
  __int64 v225; // [rsp+6E0h] [rbp-AC0h]
  _BYTE *v226; // [rsp+6E8h] [rbp-AB8h]
  __int64 v227; // [rsp+6F0h] [rbp-AB0h]
  _BYTE v228[2560]; // [rsp+6F8h] [rbp-AA8h] BYREF
  __int64 v229; // [rsp+10F8h] [rbp-A8h]
  __int64 v230; // [rsp+1100h] [rbp-A0h]
  __int64 v231; // [rsp+1108h] [rbp-98h]
  __int64 v232; // [rsp+1110h] [rbp-90h]
  _BYTE *v233; // [rsp+1118h] [rbp-88h]
  _BYTE *v234; // [rsp+1120h] [rbp-80h]
  __int64 v235; // [rsp+1128h] [rbp-78h]
  __int64 v236; // [rsp+1130h] [rbp-70h]
  __int64 v237; // [rsp+1138h] [rbp-68h]
  __int64 v238; // [rsp+1140h] [rbp-60h]
  __int64 v239; // [rsp+1148h] [rbp-58h]
  __int64 v240; // [rsp+1150h] [rbp-50h]
  char v241; // [rsp+1158h] [rbp-48h]
  char v242; // [rsp+1159h] [rbp-47h]
  __int64 v243; // [rsp+1160h] [rbp-40h]

  v20 = sub_1632FA0(*(_QWORD *)(a1 + 40));
  v219.m128i_i64[0] = a2;
  v21 = _mm_loadu_si128(&v221);
  v219.m128i_i64[1] = a4;
  v22 = (__m128)_mm_loadu_si128(&v219);
  v179 = (_BYTE *)v20;
  v186 = byte_4FA2520 | a16;
  v220[0] = 0;
  v220[1] = 0;
  v219 = v21;
  v221 = (__m128i)v22;
  v23 = sub_15E0530(a1);
  v25 = _mm_loadu_si128(&v221);
  memset(v205, 0, 24);
  v205[3] = v23;
  v210 = sub_1704300;
  v212 = v179;
  v205[4] = 0;
  v206 = 0;
  v207 = 0;
  v208 = 0;
  v211 = sub_170C140;
  v209 = v25;
  if ( v220[0] )
    ((void (__fastcall *)(__m128i *, __m128i *, __int64))v220[0])(&v219, &v219, 3);
  v187 = 0;
  if ( dword_4FA2360 )
    v187 = sub_1AE9CD0(a1);
  v189 = a5;
  v185 = 0;
  while ( 2 )
  {
    ++v185;
    v26 = 1;
    v213 = 0;
    v27 = *(_QWORD *)(a1 + 80);
    v216 = 32;
    v214 = (__int64 *)v218;
    v28 = v27 - 24;
    v215 = (__int64 *)v218;
    if ( !v27 )
      v28 = 0;
    v221.m128i_i64[0] = (__int64)v222;
    v29 = (__int64 *)v218;
    v217 = 0;
    v222[0] = v28;
    v221.m128i_i64[1] = 0x10000000001LL;
    v198 = 0;
    v219.m128i_i64[0] = (__int64)v220;
    v219.m128i_i64[1] = 0x8000000000LL;
    v30 = (__int64 *)v218;
    v31 = v222;
    v199 = 0;
    v200 = 0;
    v201 = 0;
    v188 = 0;
    while ( 1 )
    {
      v32 = v31[v26 - 1];
      v221.m128i_i32[2] = v26 - 1;
      v172 = v32;
      if ( v30 != v29 )
        goto LABEL_9;
      v34 = &v30[HIDWORD(v216)];
      if ( v34 != v30 )
      {
        v35 = 0;
        do
        {
          if ( v32 == *v30 )
            goto LABEL_10;
          if ( *v30 == -2 )
            v35 = v30;
          ++v30;
        }
        while ( v34 != v30 );
        if ( v35 )
        {
          *v35 = v32;
          --v217;
          ++v213;
          goto LABEL_22;
        }
      }
      if ( HIDWORD(v216) < (unsigned int)v216 )
      {
        ++HIDWORD(v216);
        *v34 = v32;
        ++v213;
      }
      else
      {
LABEL_9:
        sub_16CCBA0((__int64)&v213, v32);
        if ( !v33 )
          goto LABEL_10;
      }
LABEL_22:
      v176 = v172 + 40;
      v190 = *(_QWORD *)(v172 + 48);
      if ( v190 != v172 + 40 )
      {
        while ( 1 )
        {
          v44 = (__int64 *)(v190 - 24);
          v180 = v190;
          v45 = v190 - 24;
          v190 = *(_QWORD *)(v190 + 8);
          v46 = sub_1AE9990(v45, v189);
          v43 = v180;
          v47 = v46;
          if ( v46 )
          {
            sub_1AEAA40(v44);
            sub_15F20C0(v44);
            v188 = v47;
            goto LABEL_28;
          }
          if ( byte_4FA1AA0 )
          {
            if ( !byte_4FA19C0 && *(_BYTE *)(v180 - 8) == 43 )
            {
              v116 = *(_QWORD *)(v180 - 72);
              if ( *(_BYTE *)(v116 + 16) == 14 )
              {
                v117 = (__int16 *)sub_1698280();
                v22 = (__m128)0x3FF0000000000000uLL;
                sub_169D3F0((__int64)v197, 1.0);
                sub_169E320(&v203, v197, v117);
                sub_1698460((__int64)v197);
                sub_16A3360((__int64)&v202, *(__int16 **)(v116 + 32), 0, (bool *)v197);
                v120 = sub_1594120(v116, (__int64)&v202, v118, v119);
                v121 = sub_16982C0();
                v43 = v180;
                if ( v203 == v121 )
                {
                  v152 = v204;
                  if ( v204 )
                  {
                    v153 = 32LL * *(_QWORD *)(v204 - 8);
                    if ( v204 != v204 + v153 )
                    {
                      v154 = v204 + v153;
                      v155 = v204;
                      do
                      {
                        v154 -= 32;
                        sub_127D120((_QWORD *)(v154 + 8));
                      }
                      while ( v155 != v154 );
                      v152 = v155;
                      v43 = v180;
                    }
                    v184 = v43;
                    j_j_j___libc_free_0_0(v152 - 8);
                    v43 = v184;
                  }
                }
                else
                {
                  sub_1698460((__int64)&v203);
                  v43 = v180;
                }
                if ( v120 )
                {
                  v122 = *(_QWORD *)(v43 - 48);
                  if ( *(_BYTE *)(v122 + 16) == 78 )
                  {
                    v123 = *(_QWORD *)(v122 - 24);
                    if ( !*(_BYTE *)(v123 + 16)
                      && (*(_BYTE *)(v123 + 33) & 0x20) != 0
                      && (unsigned int)(*(_DWORD *)(v123 + 36) - 4474) <= 1 )
                    {
                      v175 = v43;
                      v124 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(v43 + 16) + 56LL) + 40LL);
                      v197[0] = *(_QWORD *)(v122 - 24LL * (*(_DWORD *)(v122 + 20) & 0xFFFFFFF));
                      v202 = *(_QWORD *)v197[0];
                      v125 = sub_15E26F0(v124, 4366, &v202, 1);
                      LOWORD(v204) = 257;
                      v183 = v125;
                      v126 = *(_QWORD *)(*(_QWORD *)v125 + 24LL);
                      v127 = sub_1648AB0(72, 2u, 0);
                      v43 = v175;
                      v128 = (__int64)v127;
                      if ( v127 )
                      {
                        sub_15F1EA0((__int64)v127, **(_QWORD **)(v126 + 16), 54, (__int64)(v127 - 6), 2, 0);
                        *(_QWORD *)(v128 + 56) = 0;
                        sub_15F5B40(v128, v126, v183, v197, 1, (__int64)&v202, 0, 0);
                        sub_164D160(
                          (__int64)v44,
                          v128,
                          (__m128)0x3FF0000000000000uLL,
                          *(double *)v21.m128i_i64,
                          *(double *)v25.m128i_i64,
                          a10,
                          v129,
                          v130,
                          *(double *)a13.m128i_i64,
                          a14);
                        sub_164B7C0(v128, (__int64)v44);
                        sub_15F2180(v128, (__int64)v44);
                        sub_15F20C0(v44);
                        goto LABEL_28;
                      }
                    }
                  }
                }
              }
            }
          }
          v48 = *(_DWORD *)(v43 - 4);
          v42 = v44;
          v49 = v48 & 0xFFFFFFF;
          if ( !*(_QWORD *)(v43 - 16) )
            goto LABEL_65;
          if ( !(_DWORD)v49 )
            goto LABEL_25;
          if ( (*(_BYTE *)(v43 - 1) & 0x40) != 0 )
            break;
          v36 = &v44[-3 * v49];
          if ( *(_BYTE *)(*v36 + 16) <= 0x10u )
            goto LABEL_25;
LABEL_37:
          if ( v36 != v42 )
          {
            v174 = v44;
            v50 = v42;
            v181 = v43;
            while ( 1 )
            {
              v51 = *v36;
              v52 = *(_BYTE *)(*v36 + 16);
              if ( v52 != 5 && v52 != 8 )
                goto LABEL_40;
              if ( !v201 )
                break;
              v39 = v201 - 1;
              v38 = v199;
              v53 = (v201 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
              v54 = (__int64 *)(v199 + 16LL * v53);
              v55 = *v54;
              if ( v51 == *v54 )
              {
LABEL_44:
                v56 = v54[1];
                if ( !v56 )
                {
                  v69 = v54;
                  goto LABEL_79;
                }
                if ( v51 != v56 )
                  goto LABEL_46;
LABEL_40:
                v36 += 3;
                if ( v50 == v36 )
                  goto LABEL_52;
              }
              else
              {
                v68 = 1;
                v69 = 0;
                while ( v55 != -8 )
                {
                  if ( !v69 && v55 == -16 )
                    v69 = v54;
                  v53 = v39 & (v68 + v53);
                  v54 = (__int64 *)(v199 + 16LL * v53);
                  v55 = *v54;
                  if ( v51 == *v54 )
                    goto LABEL_44;
                  ++v68;
                }
                if ( !v69 )
                  v69 = v54;
                ++v198;
                v70 = v200 + 1;
                if ( 4 * ((int)v200 + 1) < 3 * v201 )
                {
                  if ( v201 - HIDWORD(v200) - v70 <= v201 >> 3 )
                  {
                    sub_1717340((__int64)&v198, v201);
                    if ( !v201 )
                    {
LABEL_288:
                      LODWORD(v200) = v200 + 1;
                      BUG();
                    }
                    v76 = 0;
                    v77 = (v201 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
                    v78 = 1;
                    v70 = v200 + 1;
                    v69 = (__int64 *)(v199 + 16LL * v77);
                    v79 = *v69;
                    if ( v51 != *v69 )
                    {
                      while ( v79 != -8 )
                      {
                        if ( !v76 && v79 == -16 )
                          v76 = v69;
                        v77 = (v201 - 1) & (v78 + v77);
                        v69 = (__int64 *)(v199 + 16LL * v77);
                        v79 = *v69;
                        if ( v51 == *v69 )
                          goto LABEL_76;
                        ++v78;
                      }
                      if ( v76 )
                        v69 = v76;
                    }
                  }
                  goto LABEL_76;
                }
LABEL_83:
                sub_1717340((__int64)&v198, 2 * v201);
                if ( !v201 )
                  goto LABEL_288;
                v72 = (v201 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
                v70 = v200 + 1;
                v69 = (__int64 *)(v199 + 16LL * v72);
                v73 = *v69;
                if ( v51 != *v69 )
                {
                  v74 = 1;
                  v75 = 0;
                  while ( v73 != -8 )
                  {
                    if ( !v75 && v73 == -16 )
                      v75 = v69;
                    v72 = (v201 - 1) & (v74 + v72);
                    v69 = (__int64 *)(v199 + 16LL * v72);
                    v73 = *v69;
                    if ( v51 == *v69 )
                      goto LABEL_76;
                    ++v74;
                  }
                  if ( v75 )
                    v69 = v75;
                }
LABEL_76:
                LODWORD(v200) = v70;
                if ( *v69 != -8 )
                  --HIDWORD(v200);
                *v69 = v51;
                v69[1] = 0;
LABEL_79:
                v71 = sub_14DBA30(v51, (__int64)v179, v189);
                v56 = v71;
                if ( !v71 )
                {
                  v69[1] = v51;
                  goto LABEL_40;
                }
                v69[1] = v71;
                if ( v51 == v71 )
                  goto LABEL_40;
                if ( !*v36 )
                {
                  *v36 = v71;
                  goto LABEL_49;
                }
LABEL_46:
                v57 = v36[1];
                v58 = v36[2] & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v58 = v57;
                if ( v57 )
                  *(_QWORD *)(v57 + 16) = *(_QWORD *)(v57 + 16) & 3LL | v58;
                *v36 = v56;
                v188 = 1;
                if ( !v56 )
                  goto LABEL_40;
LABEL_49:
                v59 = *(_QWORD *)(v56 + 8);
                v36[1] = v59;
                if ( v59 )
                {
                  v38 = (_DWORD)v36 + 8;
                  *(_QWORD *)(v59 + 16) = (unsigned __int64)(v36 + 1) | *(_QWORD *)(v59 + 16) & 3LL;
                }
                v188 = 1;
                v36[2] = (v56 + 8) | v36[2] & 3;
                *(_QWORD *)(v56 + 8) = v36;
                v36 += 3;
                if ( v50 == v36 )
                {
LABEL_52:
                  v43 = v181;
                  v44 = v174;
                  goto LABEL_53;
                }
              }
            }
            ++v198;
            goto LABEL_83;
          }
LABEL_53:
          if ( *(_BYTE *)(v43 - 8) != 78
            || (v80 = *(_QWORD *)(v43 - 48), *(_BYTE *)(v80 + 16))
            || (*(_BYTE *)(v80 + 33) & 0x20) == 0 )
          {
            v60 = v219.m128i_u32[2];
            if ( v219.m128i_i32[2] >= (unsigned __int32)v219.m128i_i32[3] )
              goto LABEL_100;
            goto LABEL_55;
          }
          if ( (unsigned int)(*(_DWORD *)(v80 + 36) - 35) <= 3 )
          {
LABEL_28:
            if ( v176 == v190 )
              goto LABEL_56;
          }
          else
          {
            v60 = v219.m128i_u32[2];
            if ( v219.m128i_i32[2] >= (unsigned __int32)v219.m128i_i32[3] )
            {
LABEL_100:
              sub_16CD150((__int64)&v219, v220, 0, 8, v38, v39);
              v60 = v219.m128i_u32[2];
            }
LABEL_55:
            *(_QWORD *)(v219.m128i_i64[0] + 8 * v60) = v44;
            ++v219.m128i_i32[2];
            if ( v176 == v190 )
              goto LABEL_56;
          }
        }
        v36 = *(__int64 **)(v43 - 32);
        if ( *(_BYTE *)(*v36 + 16) <= 0x10u )
        {
LABEL_25:
          v173 = v43;
          v37 = sub_14DD210(v44, v179, v189);
          v42 = v44;
          v43 = v173;
          if ( v37 )
          {
            sub_164D160(
              (__int64)v44,
              v37,
              v22,
              *(double *)v21.m128i_i64,
              *(double *)v25.m128i_i64,
              a10,
              v40,
              v41,
              *(double *)a13.m128i_i64,
              a14);
            v188 = sub_1AE9990(v44, v189);
            if ( v188 )
              sub_15F20C0(v44);
            else
              v188 = 1;
            goto LABEL_28;
          }
          v48 = *(_DWORD *)(v173 - 4);
LABEL_65:
          v49 = v48 & 0xFFFFFFF;
          if ( (*(_BYTE *)(v43 - 1) & 0x40) == 0 )
          {
            v36 = &v44[-3 * v49];
            goto LABEL_37;
          }
          v36 = *(__int64 **)(v43 - 32);
        }
        v42 = &v36[3 * v49];
        goto LABEL_37;
      }
LABEL_56:
      v61 = sub_157EBA0(v172);
      v62 = *(_BYTE *)(v61 + 16);
      if ( v62 == 26 )
      {
        if ( (*(_DWORD *)(v61 + 20) & 0xFFFFFFF) != 3 )
          goto LABEL_59;
        v63 = *(_QWORD *)(v61 - 72);
        if ( *(_BYTE *)(v63 + 16) != 13 )
          goto LABEL_59;
        if ( *(_DWORD *)(v63 + 32) <= 0x40u )
          v150 = *(_QWORD *)(v63 + 24);
        else
          v150 = **(_QWORD **)(v63 + 24);
        v151 = 3LL * (v150 == 0);
        v93 = v221.m128i_u32[2];
        v92 = *(_QWORD *)(v61 - 8 * v151 - 24);
        if ( v221.m128i_i32[2] >= (unsigned __int32)v221.m128i_i32[3] )
          goto LABEL_208;
        goto LABEL_123;
      }
      if ( v62 == 27 )
      {
        v81 = (*(_BYTE *)(v61 + 23) & 0x40) != 0
            ? *(__int64 **)(v61 - 8)
            : (__int64 *)(v61 - 24LL * (*(_DWORD *)(v61 + 20) & 0xFFFFFFF));
        v82 = *v81;
        if ( *(_BYTE *)(*v81 + 16) == 13 )
          break;
      }
LABEL_59:
      v64 = 0;
      v65 = sub_15F4D60(v61);
      if ( !v65 )
      {
LABEL_10:
        v26 = v221.m128i_u32[2];
        goto LABEL_11;
      }
      do
      {
        v66 = sub_15F4DF0(v61, v64);
        v67 = v221.m128i_u32[2];
        if ( v221.m128i_i32[2] >= (unsigned __int32)v221.m128i_i32[3] )
        {
          sub_16CD150((__int64)&v221, v222, 0, 8, v32, v24);
          v67 = v221.m128i_u32[2];
        }
        ++v64;
        *(_QWORD *)(v221.m128i_i64[0] + 8 * v67) = v66;
        v26 = ++v221.m128i_i32[2];
      }
      while ( v65 != v64 );
LABEL_11:
      if ( !v26 )
        goto LABEL_124;
LABEL_12:
      v31 = (_QWORD *)v221.m128i_i64[0];
      v29 = v215;
      v30 = v214;
    }
    v83 = ((*(_DWORD *)(v61 + 20) & 0xFFFFFFFu) >> 1) - 1;
    v84 = v83 >> 2;
    if ( v83 >> 2 )
    {
      v24 = 4 * v84;
      v85 = 2;
      v84 = 0;
      while ( 1 )
      {
        v32 = v84 + 1;
        v89 = v81[3 * v85];
        if ( v89 )
        {
          if ( v82 == v89 )
            goto LABEL_116;
        }
        v86 = v81[3 * v85 + 6];
        if ( v86 && v82 == v86 )
          goto LABEL_117;
        v32 = v84 + 3;
        v87 = v81[3 * v85 + 12];
        if ( v87 && v82 == v87 )
        {
          v32 = v84 + 2;
          goto LABEL_117;
        }
        v84 += 4;
        v88 = v81[3 * (unsigned int)(2 * v84)];
        if ( v88 && v82 == v88 )
          goto LABEL_117;
        v85 += 8;
        if ( v24 == v84 )
        {
          v157 = v83 - v84;
          goto LABEL_223;
        }
      }
    }
    v157 = ((*(_DWORD *)(v61 + 20) & 0xFFFFFFFu) >> 1) - 1;
LABEL_223:
    if ( v157 == 2 )
    {
      v32 = v84;
LABEL_242:
      v84 = v32 + 1;
      v162 = v81[3 * (unsigned int)(2 * (v32 + 1))];
      if ( v162 && v82 == v162 )
      {
LABEL_117:
        if ( v83 != v32 && (_DWORD)v32 != -2 )
        {
          v90 = 24LL * (unsigned int)(2 * v32 + 3);
          goto LABEL_120;
        }
        goto LABEL_226;
      }
      goto LABEL_244;
    }
    if ( v157 == 3 )
    {
      v32 = v84 + 1;
      v164 = v81[3 * (unsigned int)(2 * (v84 + 1))];
      if ( v164 && v82 == v164 )
        goto LABEL_116;
      goto LABEL_242;
    }
    if ( v157 != 1 )
    {
LABEL_226:
      v90 = 24;
      goto LABEL_120;
    }
LABEL_244:
    v163 = v81[3 * (unsigned int)(2 * v84 + 2)];
    if ( !v163 )
      goto LABEL_226;
    if ( v82 == v163 )
    {
LABEL_116:
      v32 = v84;
      goto LABEL_117;
    }
    v90 = 24;
LABEL_120:
    if ( (*(_BYTE *)(v61 + 23) & 0x40) != 0 )
      v91 = *(_QWORD *)(v61 - 8);
    else
      v91 = v61 - 24LL * (*(_DWORD *)(v61 + 20) & 0xFFFFFFF);
    v92 = *(_QWORD *)(v91 + v90);
    v93 = v221.m128i_u32[2];
    if ( v221.m128i_i32[2] >= (unsigned __int32)v221.m128i_i32[3] )
    {
LABEL_208:
      sub_16CD150((__int64)&v221, v222, 0, 8, v32, v24);
      v93 = v221.m128i_u32[2];
    }
LABEL_123:
    *(_QWORD *)(v221.m128i_i64[0] + 8 * v93) = v92;
    v26 = v221.m128i_i32[2] + 1;
    v221.m128i_i32[2] = v26;
    if ( v26 )
      goto LABEL_12;
LABEL_124:
    v94 = v219.m128i_u32[2];
    v95 = v219.m128i_i64[0];
    v96 = v219.m128i_u32[2] + 16LL;
    v97 = v219.m128i_i32[2];
    if ( v96 > *(unsigned int *)(a2 + 12) )
    {
      v196 = v219.m128i_i64[0];
      sub_16CD150(a2, (const void *)(a2 + 16), v96, 8, v32, v24);
      v95 = v196;
    }
    v98 = *(_QWORD *)(a2 + 2064) + 1LL;
    if ( v97 )
    {
      *(_QWORD *)(a2 + 2064) = v98;
      v182 = a2 + 2064;
      v99 = ((((((4 * v97 / 3u + 1) | ((unsigned __int64)(4 * v97 / 3u + 1) >> 1)) >> 2)
             | (4 * v97 / 3u + 1)
             | ((unsigned __int64)(4 * v97 / 3u + 1) >> 1)) >> 4)
           | (((4 * v97 / 3u + 1) | ((unsigned __int64)(4 * v97 / 3u + 1) >> 1)) >> 2)
           | (4 * v97 / 3u + 1)
           | ((unsigned __int64)(4 * v97 / 3u + 1) >> 1)) >> 8;
      v100 = (((v99
              | (((((4 * v97 / 3u + 1) | ((unsigned __int64)(4 * v97 / 3u + 1) >> 1)) >> 2)
                | (4 * v97 / 3u + 1)
                | ((unsigned __int64)(4 * v97 / 3u + 1) >> 1)) >> 4)
              | (((4 * v97 / 3u + 1) | ((unsigned __int64)(4 * v97 / 3u + 1) >> 1)) >> 2)
              | (4 * v97 / 3u + 1)
              | ((unsigned __int64)(4 * v97 / 3u + 1) >> 1)) >> 16)
            | v99
            | (((((4 * v97 / 3u + 1) | ((unsigned __int64)(4 * v97 / 3u + 1) >> 1)) >> 2)
              | (4 * v97 / 3u + 1)
              | ((unsigned __int64)(4 * v97 / 3u + 1) >> 1)) >> 4)
            | (((4 * v97 / 3u + 1) | ((unsigned __int64)(4 * v97 / 3u + 1) >> 1)) >> 2)
            | (4 * v97 / 3u + 1)
            | ((4 * v97 / 3u + 1) >> 1))
           + 1;
      if ( *(_DWORD *)(a2 + 2088) < v100 )
      {
        v191 = v95;
        sub_14672C0(a2 + 2064, v100);
        v95 = v191;
      }
      v101 = v95 + 8 * v94;
      while ( 1 )
      {
        v107 = *(_DWORD *)(a2 + 2088);
        v108 = v26;
        v109 = *(_QWORD *)(v101 - 8);
        ++v26;
        if ( !v107 )
          break;
        v102 = *(_QWORD *)(a2 + 2072);
        v103 = (v107 - 1) & (((unsigned int)v109 >> 9) ^ ((unsigned int)v109 >> 4));
        v104 = (_QWORD *)(v102 + 16LL * v103);
        v105 = *v104;
        if ( v109 != *v104 )
        {
          v194 = 1;
          v115 = 0;
          while ( v105 != -8 )
          {
            if ( v115 || v105 != -16 )
              v104 = v115;
            v103 = (v107 - 1) & (v194 + v103);
            v105 = *(_QWORD *)(v102 + 16LL * v103);
            if ( v109 == v105 )
              goto LABEL_131;
            ++v194;
            v115 = v104;
            v104 = (_QWORD *)(v102 + 16LL * v103);
          }
          if ( !v115 )
            v115 = v104;
          v143 = *(_DWORD *)(a2 + 2080);
          ++*(_QWORD *)(a2 + 2064);
          v114 = v143 + 1;
          if ( 4 * v114 < 3 * v107 )
          {
            if ( v107 - *(_DWORD *)(a2 + 2084) - v114 <= v107 >> 3 )
            {
              v178 = v95;
              v195 = v108;
              sub_14672C0(v182, v107);
              v144 = *(_DWORD *)(a2 + 2088);
              if ( !v144 )
              {
LABEL_287:
                ++*(_DWORD *)(a2 + 2080);
                BUG();
              }
              v145 = v144 - 1;
              v146 = 0;
              v102 = *(_QWORD *)(a2 + 2072);
              v147 = (v144 - 1) & (((unsigned int)v109 >> 9) ^ ((unsigned int)v109 >> 4));
              v108 = v195;
              v148 = 1;
              v95 = v178;
              v114 = *(_DWORD *)(a2 + 2080) + 1;
              v115 = (_QWORD *)(v102 + 16LL * v147);
              v149 = *v115;
              if ( v109 != *v115 )
              {
                while ( v149 != -8 )
                {
                  if ( v149 == -16 && !v146 )
                    v146 = v115;
                  v147 = v145 & (v148 + v147);
                  v115 = (_QWORD *)(v102 + 16LL * v147);
                  v149 = *v115;
                  if ( v109 == *v115 )
                    goto LABEL_137;
                  ++v148;
                }
                if ( v146 )
                  v115 = v146;
              }
            }
LABEL_137:
            *(_DWORD *)(a2 + 2080) = v114;
            if ( *v115 != -8 )
              --*(_DWORD *)(a2 + 2084);
            *v115 = v109;
            *((_DWORD *)v115 + 2) = v108;
            v106 = *(unsigned int *)(a2 + 8);
            if ( (unsigned int)v106 < *(_DWORD *)(a2 + 12) )
              goto LABEL_132;
LABEL_140:
            v193 = v95;
            sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v102, v108);
            v106 = *(unsigned int *)(a2 + 8);
            v95 = v193;
            goto LABEL_132;
          }
LABEL_135:
          v177 = v95;
          v192 = v108;
          sub_14672C0(v182, 2 * v107);
          v110 = *(_DWORD *)(a2 + 2088);
          if ( !v110 )
            goto LABEL_287;
          v111 = v110 - 1;
          v112 = *(_QWORD *)(a2 + 2072);
          v108 = v192;
          v95 = v177;
          v113 = v111 & (((unsigned int)v109 >> 9) ^ ((unsigned int)v109 >> 4));
          v114 = *(_DWORD *)(a2 + 2080) + 1;
          v115 = (_QWORD *)(v112 + 16LL * v113);
          v102 = *v115;
          if ( v109 != *v115 )
          {
            v165 = 1;
            v166 = 0;
            while ( v102 != -8 )
            {
              if ( v102 == -16 && !v166 )
                v166 = v115;
              v113 = v111 & (v165 + v113);
              v115 = (_QWORD *)(v112 + 16LL * v113);
              v102 = *v115;
              if ( v109 == *v115 )
                goto LABEL_137;
              ++v165;
            }
            if ( v166 )
              v115 = v166;
          }
          goto LABEL_137;
        }
LABEL_131:
        v106 = *(unsigned int *)(a2 + 8);
        if ( (unsigned int)v106 >= *(_DWORD *)(a2 + 12) )
          goto LABEL_140;
LABEL_132:
        v101 -= 8;
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v106) = v109;
        ++*(_DWORD *)(a2 + 8);
        if ( v95 == v101 )
          goto LABEL_152;
      }
      ++*(_QWORD *)(a2 + 2064);
      goto LABEL_135;
    }
    *(_QWORD *)(a2 + 2064) = v98;
LABEL_152:
    j___libc_free_0(v199);
    if ( (_QWORD *)v219.m128i_i64[0] != v220 )
      _libc_free(v219.m128i_u64[0]);
    if ( (_QWORD *)v221.m128i_i64[0] != v222 )
      _libc_free(v221.m128i_u64[0]);
    v131 = (unsigned __int64)v215;
    v132 = *(_QWORD *)(a1 + 80);
    if ( v132 != a1 + 72 )
    {
      while ( 1 )
      {
        v134 = v214;
        v136 = v132 - 24;
        if ( !v132 )
          v136 = 0;
        if ( (__int64 *)v131 == v214 )
          break;
        v133 = (__int64 *)(v131 + 8LL * (unsigned int)v216);
        v134 = sub_16CC9F0((__int64)&v213, v136);
        if ( v136 == *v134 )
        {
          v131 = (unsigned __int64)v215;
          if ( v215 == v214 )
            v156 = &v215[HIDWORD(v216)];
          else
            v156 = &v215[(unsigned int)v216];
          goto LABEL_172;
        }
        v131 = (unsigned __int64)v215;
        if ( v215 == v214 )
        {
          v134 = &v215[HIDWORD(v216)];
          v156 = v134;
          goto LABEL_172;
        }
        v134 = &v215[(unsigned int)v216];
LABEL_161:
        if ( v134 == v133 )
        {
          v135 = sub_1AEBFA0(v136);
          v131 = (unsigned __int64)v215;
          v188 |= v135 != 0;
        }
        v132 = *(_QWORD *)(v132 + 8);
        if ( a1 + 72 == v132 )
          goto LABEL_179;
      }
      v133 = (__int64 *)(v131 + 8LL * HIDWORD(v216));
      if ( (__int64 *)v131 == v133 )
      {
        v156 = (__int64 *)v131;
      }
      else
      {
        do
        {
          if ( v136 == *v134 )
            break;
          ++v134;
        }
        while ( v133 != v134 );
        v156 = (__int64 *)(v131 + 8LL * HIDWORD(v216));
      }
LABEL_172:
      while ( v156 != v134 )
      {
        if ( (unsigned __int64)*v134 < 0xFFFFFFFFFFFFFFFELL )
          break;
        ++v134;
      }
      goto LABEL_161;
    }
LABEL_179:
    if ( (__int64 *)v131 != v214 )
      _libc_free(v131);
    v187 |= v188;
    v137 = sub_1560180(a1 + 112, 17);
    v241 = 0;
    LOBYTE(v222[0]) = v137;
    v221.m128i_i64[0] = a2;
    BYTE1(v222[0]) = v186;
    v227 = 0x4000000000LL;
    v232 = a6;
    v229 = a3;
    v236 = a6;
    v230 = a4;
    v237 = a4;
    v221.m128i_i64[1] = (__int64)v205;
    v239 = a15;
    v222[1] = 0;
    v223 = 0;
    v224 = 0;
    v225 = 0;
    v226 = v228;
    v231 = v189;
    v233 = v179;
    v234 = v179;
    v235 = v189;
    v238 = 0;
    v240 = a18;
    v242 = a17;
    v243 = (unsigned int)dword_4FA2440;
    if ( (unsigned __int8)sub_1716560(
                            (__int64)&v221,
                            a1,
                            v22,
                            *(double *)v21.m128i_i64,
                            *(double *)v25.m128i_i64,
                            a10,
                            v138,
                            v139,
                            a13,
                            a14) )
    {
      v140 = v226;
      v141 = (unsigned __int64)&v226[40 * (unsigned int)v227];
      if ( v226 != (_BYTE *)v141 )
      {
        do
        {
          v142 = *(_QWORD *)(v141 - 16);
          v141 -= 40LL;
          *(_QWORD *)v141 = &unk_49EE2B0;
          if ( v142 != -8 && v142 != 0 && v142 != -16 )
            sub_1649B30((_QWORD *)(v141 + 8));
        }
        while ( v140 != (_BYTE *)v141 );
        v141 = (unsigned __int64)v226;
      }
      if ( (_BYTE *)v141 != v228 )
        _libc_free(v141);
      j___libc_free_0(v223);
      continue;
    }
    break;
  }
  v158 = v226;
  v159 = (unsigned __int64)&v226[40 * (unsigned int)v227];
  if ( v226 != (_BYTE *)v159 )
  {
    do
    {
      v160 = *(_QWORD *)(v159 - 16);
      v159 -= 40LL;
      *(_QWORD *)v159 = &unk_49EE2B0;
      if ( v160 != -8 && v160 != 0 && v160 != -16 )
        sub_1649B30((_QWORD *)(v159 + 8));
    }
    while ( v158 != (_BYTE *)v159 );
    v159 = (unsigned __int64)v226;
  }
  if ( (_BYTE *)v159 != v228 )
    _libc_free(v159);
  j___libc_free_0(v223);
  if ( v210 )
    v210((const __m128i **)&v209, &v209, 3);
  if ( v205[0] )
    sub_161E7C0((__int64)v205, v205[0]);
  LOBYTE(v159) = v187 | (v185 > 1);
  return (unsigned int)v159;
}
