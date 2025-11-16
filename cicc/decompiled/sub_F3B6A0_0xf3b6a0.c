// Function: sub_F3B6A0
// Address: 0xf3b6a0
//
__int64 __fastcall sub_F3B6A0(__int64 a1, __int64 a2)
{
  _QWORD **v2; // r12
  char v3; // al
  bool v4; // zf
  __int64 *v5; // rax
  __int64 v6; // rdi
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  _QWORD *i; // rcx
  void **v10; // r13
  char *v11; // r12
  char *v12; // r15
  void *v13; // rsi
  __int64 v14; // rax
  unsigned __int8 v15; // dl
  void *v16; // rbx
  __int64 v17; // rax
  __int64 *v18; // r14
  int v19; // ebx
  unsigned int v20; // eax
  char *v21; // rbx
  __int64 v22; // r12
  __int64 v23; // rdi
  __int64 v24; // r9
  __int64 *v25; // r9
  __int64 v26; // r8
  __int64 v27; // rax
  __int64 v28; // r9
  signed __int64 v29; // r14
  unsigned __int64 v30; // rdx
  char *v31; // rcx
  int v32; // eax
  __int64 v33; // rdx
  _QWORD *v34; // rax
  int v35; // eax
  __int64 *v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // rdx
  size_t v39; // rdx
  int v40; // eax
  __int64 v41; // r14
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // r14
  __int64 *v46; // r14
  int v47; // eax
  __int64 v48; // rcx
  __int64 v49; // r8
  __m128i *v50; // rdx
  __int64 v51; // r9
  __int64 v52; // r12
  __int64 v53; // r13
  __int64 *v54; // rbx
  __int64 v55; // r14
  __int64 v56; // rdx
  unsigned int v57; // ecx
  unsigned int v58; // r14d
  unsigned int v59; // eax
  unsigned int v60; // esi
  __int64 v61; // r14
  __int64 v62; // rax
  __int64 v63; // rax
  unsigned __int8 v64; // dl
  void *v65; // rcx
  __int64 v66; // r15
  void *v67; // rax
  __int64 *v68; // r13
  int v69; // ebx
  unsigned int j; // ebx
  __int64 v71; // r12
  __int64 *v72; // r9
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rcx
  __int64 v77; // r8
  __int64 v78; // r9
  char *v79; // r8
  __int64 v80; // r9
  __int64 *v81; // rax
  __int64 v82; // rdx
  __int64 v83; // rcx
  size_t v84; // rdx
  int v85; // eax
  __int64 v86; // rdx
  __int64 v87; // r15
  __m128i *v88; // rdx
  __int64 v89; // rcx
  __int64 v90; // r8
  __int64 v91; // r9
  __int64 v92; // rdx
  unsigned int v93; // eax
  unsigned int v94; // esi
  unsigned int v95; // ecx
  bool v96; // al
  __int64 v97; // rax
  __int64 v98; // r13
  void *v99; // rdi
  _QWORD **v100; // rbx
  int v101; // eax
  _QWORD *v102; // rdi
  _QWORD **v104; // rbx
  _QWORD *v105; // rdi
  __int64 v106; // rax
  __int64 v107; // rbx
  __int64 *v108; // r14
  int v109; // eax
  __int64 *v110; // r12
  __int64 v111; // r13
  __int64 v112; // rbx
  __int64 v113; // rdx
  unsigned int v114; // ecx
  unsigned int v115; // r14d
  unsigned int v116; // eax
  unsigned int v117; // esi
  __int64 v118; // rax
  __int64 v119; // rax
  __int64 v120; // rdx
  __int64 v121; // rax
  unsigned __int64 v122; // rdx
  void *v123; // rdi
  bool v124; // al
  __int64 v125; // rax
  __int64 v126; // r14
  bool v127; // al
  bool v128; // al
  __int64 v129; // rax
  __int64 v130; // rbx
  __int64 v131; // rax
  unsigned __int64 v132; // rdx
  __int64 v133; // rdx
  unsigned int v134; // eax
  unsigned int v135; // esi
  unsigned int v136; // ecx
  bool v137; // al
  __int64 v138; // r9
  bool v139; // al
  unsigned int v140; // edx
  __int64 v141; // [rsp+10h] [rbp-450h]
  int v142; // [rsp+24h] [rbp-43Ch]
  int v143; // [rsp+24h] [rbp-43Ch]
  void **v144; // [rsp+30h] [rbp-430h]
  char *v145; // [rsp+38h] [rbp-428h]
  char *v146; // [rsp+40h] [rbp-420h]
  __int64 v147; // [rsp+48h] [rbp-418h]
  int v148; // [rsp+48h] [rbp-418h]
  unsigned int v149; // [rsp+48h] [rbp-418h]
  int v150; // [rsp+50h] [rbp-410h]
  unsigned int v151; // [rsp+50h] [rbp-410h]
  __int64 v152; // [rsp+50h] [rbp-410h]
  unsigned int v153; // [rsp+50h] [rbp-410h]
  bool v154; // [rsp+58h] [rbp-408h]
  int v155; // [rsp+58h] [rbp-408h]
  char *v156; // [rsp+58h] [rbp-408h]
  __m128i *v157; // [rsp+58h] [rbp-408h]
  char *v158; // [rsp+68h] [rbp-3F8h]
  __int64 *v159; // [rsp+68h] [rbp-3F8h]
  int v160; // [rsp+70h] [rbp-3F0h]
  __int64 v161; // [rsp+70h] [rbp-3F0h]
  __int64 v162; // [rsp+70h] [rbp-3F0h]
  int v163; // [rsp+70h] [rbp-3F0h]
  bool v164; // [rsp+70h] [rbp-3F0h]
  int v165; // [rsp+70h] [rbp-3F0h]
  __int64 *v166; // [rsp+70h] [rbp-3F0h]
  __m128i *v167; // [rsp+70h] [rbp-3F0h]
  unsigned int v168; // [rsp+78h] [rbp-3E8h]
  bool v169; // [rsp+78h] [rbp-3E8h]
  __int64 v170; // [rsp+78h] [rbp-3E8h]
  void *v171; // [rsp+80h] [rbp-3E0h]
  __int64 v172; // [rsp+88h] [rbp-3D8h]
  __int64 v173; // [rsp+F8h] [rbp-368h] BYREF
  void *v174; // [rsp+100h] [rbp-360h] BYREF
  void *v175; // [rsp+108h] [rbp-358h] BYREF
  void *v176; // [rsp+110h] [rbp-350h] BYREF
  __int128 v177; // [rsp+118h] [rbp-348h] BYREF
  char v178; // [rsp+128h] [rbp-338h]
  void *v179; // [rsp+130h] [rbp-330h]
  __int64 v180[3]; // [rsp+140h] [rbp-320h] BYREF
  char v181; // [rsp+158h] [rbp-308h]
  __int64 v182; // [rsp+160h] [rbp-300h]
  __m128i v183; // [rsp+170h] [rbp-2F0h] BYREF
  __m128i v184; // [rsp+180h] [rbp-2E0h] BYREF
  __int64 v185; // [rsp+190h] [rbp-2D0h]
  void *s2; // [rsp+1A0h] [rbp-2C0h] BYREF
  __int64 v187; // [rsp+1A8h] [rbp-2B8h]
  char v188[8]; // [rsp+1B0h] [rbp-2B0h] BYREF
  char v189; // [rsp+1B8h] [rbp-2A8h]
  __int64 v190; // [rsp+1C0h] [rbp-2A0h]
  void *v191; // [rsp+1D0h] [rbp-290h] BYREF
  __int64 v192; // [rsp+1D8h] [rbp-288h]
  char v193; // [rsp+1E0h] [rbp-280h] BYREF
  _QWORD *v194; // [rsp+1E8h] [rbp-278h]
  _QWORD *v195; // [rsp+1F0h] [rbp-270h]
  __int64 v196; // [rsp+200h] [rbp-260h]
  char *v197; // [rsp+210h] [rbp-250h] BYREF
  __int64 v198; // [rsp+218h] [rbp-248h]
  _BYTE v199[16]; // [rsp+220h] [rbp-240h] BYREF
  char *v200; // [rsp+230h] [rbp-230h]
  __int64 v201; // [rsp+240h] [rbp-220h]
  _BYTE *v202; // [rsp+250h] [rbp-210h] BYREF
  __int64 v203; // [rsp+258h] [rbp-208h]
  _BYTE v204[64]; // [rsp+260h] [rbp-200h] BYREF
  __int64 v205; // [rsp+2A0h] [rbp-1C0h] BYREF
  __int64 v206; // [rsp+2A8h] [rbp-1B8h]
  __int64 *v207; // [rsp+2B0h] [rbp-1B0h] BYREF
  unsigned int v208; // [rsp+2B8h] [rbp-1A8h]
  _BYTE v209[48]; // [rsp+430h] [rbp-30h] BYREF

  v3 = *(_BYTE *)(a1 + 40);
  v205 = 0;
  v206 = 1;
  v154 = v3;
  v4 = v3 == 0;
  v202 = v204;
  v203 = 0x800000000LL;
  v5 = (__int64 *)&v207;
  if ( !v4 )
  {
    do
    {
      *v5 = 0;
      v5 += 12;
      *((_BYTE *)v5 - 72) = 0;
      *(v5 - 8) = 0;
    }
    while ( v5 != (__int64 *)v209 );
    v141 = a1 + 48;
    v147 = *(_QWORD *)(a1 + 56);
    if ( v147 != a1 + 48 )
    {
      while ( 1 )
      {
        if ( !v147 )
          BUG();
        v6 = *(_QWORD *)(v147 + 40);
        if ( v6 )
        {
          v7 = (_QWORD *)sub_B14240(v6);
          for ( i = v8; v8 != v7; v7 = (_QWORD *)v7[1] )
          {
            if ( !*((_BYTE *)v7 + 32) )
              break;
          }
        }
        else
        {
          i = &qword_4F81430[1];
          v7 = &qword_4F81430[1];
        }
        v191 = v7;
        a2 = (__int64)&v191;
        v192 = (__int64)i;
        v194 = i;
        v10 = &v191;
        v195 = i;
        sub_F333F0((__int64)&v197, (__int64 *)&v191);
        v11 = (char *)v198;
        v158 = v200;
        v12 = v197;
        if ( v200 != v197 )
          break;
LABEL_127:
        v147 = *(_QWORD *)(v147 + 8);
        if ( v141 == v147 )
        {
          v104 = (_QWORD **)v202;
          v2 = (_QWORD **)&v202[8 * (unsigned int)v203];
          v101 = v203;
          if ( v2 != (_QWORD **)v202 )
          {
            do
            {
              v105 = *v104++;
              sub_B14290(v105);
            }
            while ( v2 != v104 );
            goto LABEL_118;
          }
          goto LABEL_119;
        }
      }
      while ( 1 )
      {
        if ( !v12[64] )
          goto LABEL_125;
        v13 = (void *)*((_QWORD *)v12 + 3);
        v191 = v13;
        if ( v13 )
          sub_B96E90((__int64)v10, (__int64)v13, 1);
        v14 = sub_B10CD0((__int64)v10);
        v15 = *(_BYTE *)(v14 - 16);
        if ( (v15 & 2) != 0 )
        {
          if ( *(_DWORD *)(v14 - 24) != 2 )
            goto LABEL_16;
          v106 = *(_QWORD *)(v14 - 32);
        }
        else
        {
          if ( ((*(_WORD *)(v14 - 16) >> 6) & 0xF) != 2 )
          {
LABEL_16:
            v16 = 0;
            goto LABEL_17;
          }
          v106 = v14 - 16 - 8LL * ((v15 >> 2) & 0xF);
        }
        v16 = *(void **)(v106 + 8);
LABEL_17:
        v17 = sub_B12000((__int64)(v12 + 72));
        v178 = 0;
        v176 = (void *)v17;
        v179 = v16;
        if ( v191 )
          sub_B91220((__int64)v10, (__int64)v191);
        if ( (v206 & 1) != 0 )
        {
          v18 = (__int64 *)&v207;
          v19 = 3;
        }
        else
        {
          v107 = v208;
          v18 = v207;
          if ( !v208 )
            goto LABEL_176;
          v19 = v208 - 1;
        }
        v191 = 0;
        LOBYTE(v194) = 0;
        v195 = 0;
        LODWORD(v180[0]) = 0;
        if ( v178 )
        {
          LOWORD(v180[0]) = WORD4(v177);
          WORD1(v180[0]) = v177;
        }
        s2 = v179;
        v183.m128i_i64[0] = (__int64)v176;
        v160 = 1;
        v20 = v19 & sub_F11290(v183.m128i_i64, v180, (__int64 *)&s2);
        v150 = v19;
        v21 = v11;
        v168 = v20;
        v22 = (__int64)v176;
        while ( 1 )
        {
          v23 = (__int64)&v18[12 * v168];
          if ( *(_QWORD *)v23 == v22
            && v178 == *(_BYTE *)(v23 + 24)
            && (!v178 || v177 == *(_OWORD *)(v23 + 8))
            && v179 == *(void **)(v23 + 32) )
          {
            v25 = &v18[12 * v168];
            v11 = v21;
            goto LABEL_29;
          }
          if ( sub_F34140(v23, (__int64)v10) )
            break;
          v168 = v150 & (v160 + v168);
          ++v160;
        }
        v11 = v21;
        if ( (v206 & 1) != 0 )
        {
          v18 = (__int64 *)&v207;
          v24 = 48;
          goto LABEL_28;
        }
        v18 = v207;
        v107 = v208;
LABEL_176:
        v24 = 12 * v107;
LABEL_28:
        v25 = &v18[v24];
LABEL_29:
        v169 = v154;
        if ( v12[64] == 2 )
        {
          v166 = v25;
          v118 = sub_B13870((__int64)v12);
          v119 = sub_AE9410(v118);
          v25 = v166;
          v169 = v119 == v120;
        }
        v161 = (__int64)v25;
        sub_B129C0(v10, (__int64)v12);
        s2 = v188;
        v187 = 0x400000000LL;
        v27 = (__int64)v191;
        v172 = v192;
        v28 = v161;
        v171 = v191;
        v183.m128i_i64[0] = (__int64)v191;
        if ( (void *)v192 == v191 )
        {
          v31 = v188;
          v32 = 0;
          LODWORD(v29) = 0;
        }
        else
        {
          v29 = 0;
          do
          {
            while ( 1 )
            {
              v30 = v27 & 0xFFFFFFFFFFFFFFF8LL;
              if ( (v27 & 4) != 0 || !v30 )
                break;
              ++v29;
              v27 = v30 + 144;
              if ( v192 == v30 + 144 )
                goto LABEL_37;
            }
            ++v29;
            v27 = (v30 + 8) | 4;
          }
          while ( v192 != v27 );
LABEL_37:
          v31 = v188;
          v32 = 0;
          if ( v29 > 4 )
          {
            sub_C8D5F0((__int64)&s2, v188, v29, 8u, v26, v161);
            v32 = v187;
            v28 = v161;
            v31 = (char *)s2 + 8 * (unsigned int)v187;
          }
        }
        a2 = v172;
        v33 = (__int64)v171;
        v173 = v172;
        v175 = v171;
        v174 = v171;
        v180[0] = v172;
        v183.m128i_i64[0] = (__int64)v171;
        if ( (void *)v172 != v171 )
        {
          while ( 1 )
          {
            while ( 1 )
            {
              v31 += 8;
              v34 = (_QWORD *)(v33 & 0xFFFFFFFFFFFFFFF8LL);
              if ( (v33 & 4) == 0 )
                break;
              *((_QWORD *)v31 - 1) = *(_QWORD *)(*v34 + 136LL);
LABEL_42:
              v33 = (unsigned __int64)(v34 + 1) | 4;
              if ( v33 == v172 )
                goto LABEL_46;
            }
            *((_QWORD *)v31 - 1) = v34[17];
            v33 = (__int64)(v34 + 18);
            if ( !v34 )
              goto LABEL_42;
            if ( v34 + 18 == (_QWORD *)v172 )
            {
LABEL_46:
              v32 = v187;
              break;
            }
          }
        }
        v35 = v29 + v32;
        LODWORD(v187) = v35;
        if ( (v206 & 1) != 0 )
        {
          v36 = (__int64 *)&v207;
          v37 = 48;
        }
        else
        {
          v36 = v207;
          v37 = 12LL * v208;
        }
        v38 = (__int64)&v36[v37];
        if ( v28 == v38
          || (v38 = *(unsigned int *)(v28 + 48), v38 != v35)
          || (v39 = 8 * v38) != 0
          && (a2 = (__int64)s2, v162 = v28, v40 = memcmp(*(const void **)(v28 + 40), s2, v39), v28 = v162, v40)
          || (v41 = *(_QWORD *)(v28 + 88), v41 != sub_B11F60((__int64)(v12 + 80))) )
        {
          if ( v169 )
          {
            v45 = sub_B11F60((__int64)(v12 + 80));
            v191 = &v193;
            v192 = 0x400000000LL;
            if ( (_DWORD)v187 )
              sub_F33640((__int64)v10, (__int64)&s2, (unsigned int)v187, v42, v43, v44);
            v196 = v45;
            if ( (v206 & 1) != 0 )
            {
              v163 = 3;
              v46 = (__int64 *)&v207;
LABEL_59:
              v180[0] = 0;
              v181 = 0;
              v182 = 0;
              v183 = 0u;
              v184.m128i_i64[0] = 0;
              v184.m128i_i64[1] = 1;
              v185 = 0;
              LODWORD(v173) = 0;
              if ( v178 )
                LODWORD(v173) = WORD4(v177) | ((_DWORD)v177 << 16);
              v175 = v179;
              v174 = v176;
              v47 = sub_F11290((__int64 *)&v174, &v173, (__int64 *)&v175);
              v50 = 0;
              v51 = (__int64)v176;
              v151 = v163 & v47;
              v142 = 1;
              v146 = v12;
              v145 = v11;
              v52 = (__int64)v176;
              v144 = v10;
              v53 = 0;
              v54 = v46;
              while ( 1 )
              {
                v55 = (__int64)&v54[12 * v151];
                if ( *(_QWORD *)v55 == v52
                  && v178 == *(_BYTE *)(v55 + 24)
                  && (!v178 || v177 == *(_OWORD *)(v55 + 8))
                  && v179 == *(void **)(v55 + 32) )
                {
                  break;
                }
                if ( sub_F34140(v55, (__int64)v180) )
                {
                  v56 = v53;
                  v11 = v145;
                  v10 = v144;
                  if ( !v56 )
                    v56 = (__int64)&v54[12 * v151];
                  v57 = 12;
                  ++v205;
                  v58 = 4;
                  v180[0] = v56;
                  v59 = ((unsigned int)v206 >> 1) + 1;
                  if ( (v206 & 1) == 0 )
                  {
                    v60 = v208;
                    goto LABEL_161;
                  }
                  goto LABEL_162;
                }
                v139 = sub_F34140(v55, (__int64)&v183);
                if ( !v53 && v139 )
                  v53 = (__int64)&v54[12 * v151];
                v151 = v163 & (v142 + v151);
                ++v142;
              }
LABEL_189:
              v12 = v146;
              v11 = v145;
              v126 = v55 + 40;
              v10 = v144;
              goto LABEL_167;
            }
            v60 = v208;
            v46 = v207;
            if ( v208 )
            {
              v163 = v208 - 1;
              goto LABEL_59;
            }
            ++v205;
            v56 = 0;
            v180[0] = 0;
            v59 = ((unsigned int)v206 >> 1) + 1;
LABEL_161:
            v58 = v60;
            v57 = 3 * v60;
LABEL_162:
            if ( v57 <= 4 * v59 )
            {
              sub_F3B390((__int64)&v205, 2 * v58);
            }
            else
            {
              if ( v58 - (v59 + HIDWORD(v206)) > v58 >> 3 )
                goto LABEL_164;
              sub_F3B390((__int64)&v205, v58);
            }
            sub_F386A0((__int64)&v205, (__int64)&v176, v180);
            v56 = v180[0];
            v59 = ((unsigned int)v206 >> 1) + 1;
LABEL_164:
            v167 = (__m128i *)v56;
            v183.m128i_i64[0] = 0;
            v184.m128i_i8[8] = 0;
            LODWORD(v206) = (2 * v59) | v206 & 1;
            v185 = 0;
            v124 = sub_F34140(v56, (__int64)&v183);
            v50 = v167;
            if ( !v124 )
              --HIDWORD(v206);
LABEL_166:
            *v167 = _mm_loadu_si128((const __m128i *)&v176);
            v50[1] = _mm_loadu_si128((const __m128i *)((char *)&v177 + 8));
            v125 = (__int64)v179;
            v50[5].m128i_i64[1] = 0;
            v126 = (__int64)&v50[2].m128i_i64[1];
            v50[2].m128i_i64[0] = v125;
            v50[2].m128i_i64[1] = (__int64)&v50[3].m128i_i64[1];
            v50[3].m128i_i64[0] = 0x400000000LL;
LABEL_167:
            a2 = (__int64)v10;
            sub_F334E0(v126, (char **)v10, (__int64)v50, v48, v49, v51);
            *(_QWORD *)(v126 + 48) = v196;
            if ( v191 != &v193 )
              _libc_free(v191, v10);
            goto LABEL_169;
          }
          v191 = &v193;
          v192 = 0x400000000LL;
          if ( (_DWORD)v187 )
            sub_F33640((__int64)v10, (__int64)&s2, v38, v37 * 8, v26, v28);
          v196 = 0;
          if ( (v206 & 1) != 0 )
          {
            v165 = 3;
            v108 = (__int64 *)&v207;
LABEL_140:
            v180[0] = 0;
            v181 = 0;
            v182 = 0;
            v183 = 0u;
            v184.m128i_i64[0] = 0;
            v184.m128i_i64[1] = 1;
            v185 = 0;
            LODWORD(v173) = 0;
            if ( v178 )
              LODWORD(v173) = WORD4(v177) | ((_DWORD)v177 << 16);
            v175 = v179;
            v174 = v176;
            v109 = sub_F11290((__int64 *)&v174, &v173, (__int64 *)&v175);
            v50 = 0;
            v51 = (__int64)v176;
            v153 = v165 & v109;
            v143 = 1;
            v146 = v12;
            v145 = v11;
            v110 = v108;
            v144 = v10;
            v111 = 0;
            v112 = (__int64)v176;
            while ( 1 )
            {
              v55 = (__int64)&v110[12 * v153];
              if ( v112 == *(_QWORD *)v55 && v178 == *(_BYTE *)(v55 + 24) && (!v178 || v177 == *(_OWORD *)(v55 + 8)) )
              {
                if ( v179 == *(void **)(v55 + 32) )
                  goto LABEL_189;
                if ( sub_F34140(v55, (__int64)v180) )
                {
LABEL_145:
                  v113 = v111;
                  v11 = v145;
                  v10 = v144;
                  if ( !v113 )
                    v113 = v55;
                  v114 = 12;
                  ++v205;
                  v115 = 4;
                  v180[0] = v113;
                  v116 = ((unsigned int)v206 >> 1) + 1;
                  if ( (v206 & 1) == 0 )
                  {
                    v117 = v208;
                    goto LABEL_179;
                  }
                  goto LABEL_180;
                }
              }
              else if ( sub_F34140(v55, (__int64)v180) )
              {
                goto LABEL_145;
              }
              v128 = sub_F34140(v55, (__int64)&v183);
              if ( !v111 && v128 )
                v111 = (__int64)&v110[12 * v153];
              v153 = v165 & (v143 + v153);
              ++v143;
            }
          }
          v117 = v208;
          v108 = v207;
          if ( v208 )
          {
            v165 = v208 - 1;
            goto LABEL_140;
          }
          ++v205;
          v113 = 0;
          v180[0] = 0;
          v116 = ((unsigned int)v206 >> 1) + 1;
LABEL_179:
          v115 = v117;
          v114 = 3 * v117;
LABEL_180:
          if ( 4 * v116 >= v114 )
          {
            sub_F3B390((__int64)&v205, 2 * v115);
          }
          else
          {
            if ( v115 - (v116 + HIDWORD(v206)) > v115 >> 3 )
              goto LABEL_182;
            sub_F3B390((__int64)&v205, v115);
          }
          sub_F386A0((__int64)&v205, (__int64)&v176, v180);
          v113 = v180[0];
          v116 = ((unsigned int)v206 >> 1) + 1;
LABEL_182:
          v167 = (__m128i *)v113;
          v183.m128i_i64[0] = 0;
          v184.m128i_i8[8] = 0;
          LODWORD(v206) = (2 * v116) | v206 & 1;
          v185 = 0;
          v127 = sub_F34140(v113, (__int64)&v183);
          v50 = v167;
          if ( !v127 )
            --HIDWORD(v206);
          goto LABEL_166;
        }
        if ( v169 )
        {
          v121 = (unsigned int)v203;
          v122 = (unsigned int)v203 + 1LL;
          if ( v122 > HIDWORD(v203) )
          {
            a2 = (__int64)v204;
            sub_C8D5F0((__int64)&v202, v204, v122, 8u, v26, v28);
            v121 = (unsigned int)v203;
          }
          *(_QWORD *)&v202[8 * v121] = v12;
          v123 = s2;
          LODWORD(v203) = v203 + 1;
          if ( s2 == v188 )
            goto LABEL_125;
LABEL_159:
          _libc_free(v123, a2);
          goto LABEL_125;
        }
LABEL_169:
        v123 = s2;
        if ( s2 != v188 )
          goto LABEL_159;
        do
LABEL_125:
          v12 = (char *)*((_QWORD *)v12 + 1);
        while ( v12 != v11 && v12[32] );
        if ( v12 == v158 )
          goto LABEL_127;
      }
    }
LABEL_208:
    v101 = 0;
    goto LABEL_119;
  }
  do
  {
    *v5 = 0;
    v5 += 12;
    *((_BYTE *)v5 - 72) = 0;
    *(v5 - 8) = 0;
  }
  while ( v5 != (__int64 *)v209 );
  v170 = a1 + 48;
  if ( *(_QWORD *)(a1 + 56) == a1 + 48 )
    goto LABEL_208;
  v61 = *(_QWORD *)(a1 + 56);
  do
  {
    while ( 1 )
    {
      if ( !v61 )
        BUG();
      if ( *(_BYTE *)(v61 - 24) != 85 )
        goto LABEL_71;
      v62 = *(_QWORD *)(v61 - 56);
      if ( !v62 )
        goto LABEL_71;
      if ( *(_BYTE *)v62 )
        goto LABEL_71;
      if ( *(_QWORD *)(v62 + 24) != *(_QWORD *)(v61 + 56) )
        goto LABEL_71;
      if ( (*(_BYTE *)(v62 + 33) & 0x20) == 0 )
        goto LABEL_71;
      v164 = *(_DWORD *)(v62 + 36) == 68 || *(_DWORD *)(v62 + 36) == 71;
      if ( !v164 )
        goto LABEL_71;
      v63 = sub_B10CD0(v61 + 24);
      v64 = *(_BYTE *)(v63 - 16);
      if ( (v64 & 2) != 0 )
      {
        if ( *(_DWORD *)(v63 - 24) != 2 )
          goto LABEL_81;
        v129 = *(_QWORD *)(v63 - 32);
      }
      else
      {
        if ( ((*(_WORD *)(v63 - 16) >> 6) & 0xF) != 2 )
        {
LABEL_81:
          v65 = 0;
          goto LABEL_82;
        }
        v129 = v63 - 16 - 8LL * ((v64 >> 2) & 0xF);
      }
      v65 = *(void **)(v129 + 8);
LABEL_82:
      v66 = v61 - 24;
      v67 = *(void **)(*(_QWORD *)(v61 - 24 + 32 * (1LL - (*(_DWORD *)(v61 - 20) & 0x7FFFFFF))) + 24LL);
      v184.m128i_i8[8] = 0;
      v185 = (__int64)v65;
      v183.m128i_i64[0] = (__int64)v67;
      if ( (v206 & 1) != 0 )
      {
        v68 = (__int64 *)&v207;
        v69 = 3;
      }
      else
      {
        v130 = v208;
        v68 = v207;
        if ( !v208 )
          goto LABEL_231;
        v69 = v208 - 1;
      }
      v197 = 0;
      v199[8] = 0;
      v200 = 0;
      LODWORD(v180[0]) = 0;
      v191 = v65;
      s2 = v67;
      v155 = 1;
      v148 = v69;
      for ( j = v69 & sub_F11290((__int64 *)&s2, v180, (__int64 *)&v191); ; j = v140 & v148 )
      {
        v71 = (__int64)&v68[12 * j];
        if ( sub_F34140((__int64)&v183, v71) )
          break;
        if ( sub_F34140(v71, (__int64)&v197) )
          goto LABEL_236;
        v140 = j + v155++;
      }
      v72 = &v68[12 * j];
      if ( v71 )
        goto LABEL_87;
LABEL_236:
      if ( (v206 & 1) != 0 )
      {
        v68 = (__int64 *)&v207;
        v138 = 48;
        goto LABEL_232;
      }
      v68 = v207;
      v130 = v208;
LABEL_231:
      v138 = 12 * v130;
LABEL_232:
      v72 = &v68[v138];
LABEL_87:
      v73 = *(_QWORD *)(v61 - 56);
      if ( !v73 || *(_BYTE *)v73 || *(_QWORD *)(v73 + 24) != *(_QWORD *)(v61 + 56) )
        BUG();
      if ( *(_DWORD *)(v73 + 36) == 68 )
      {
        v159 = v72;
        v74 = sub_AE9410(*(_QWORD *)(*(_QWORD *)(v66 + 32 * (3LL - (*(_DWORD *)(v61 - 20) & 0x7FFFFFF))) + 24LL));
        v72 = v159;
        v164 = v75 == v74;
      }
      v152 = (__int64)v72;
      sub_B58E30(&v197, v66);
      a2 = (__int64)v180;
      v191 = &v193;
      v192 = 0x400000000LL;
      s2 = (void *)v198;
      v180[0] = (__int64)v197;
      sub_F388A0((__int64)&v191, v180, &s2, v76, v77, v78);
      v80 = v152;
      if ( (v206 & 1) != 0 )
      {
        v81 = (__int64 *)&v207;
        v82 = 384;
      }
      else
      {
        v81 = v207;
        v82 = 96LL * v208;
      }
      v83 = (unsigned int)v192;
      if ( (__int64 *)v152 != (__int64 *)((char *)v81 + v82) )
      {
        v82 = *(unsigned int *)(v152 + 48);
        if ( v82 == (unsigned int)v192 )
        {
          v84 = 8 * v82;
          v79 = (char *)v191;
          if ( !v84
            || (a2 = (__int64)v191,
                v149 = v192,
                v156 = (char *)v191,
                v85 = memcmp(*(const void **)(v152 + 40), v191, v84),
                v79 = v156,
                v80 = v152,
                v83 = v149,
                !v85) )
          {
            v82 = *(_DWORD *)(v61 - 20) & 0x7FFFFFF;
            if ( *(_QWORD *)(v80 + 88) == *(_QWORD *)(*(_QWORD *)(v66 + 32 * (2 - v82)) + 24LL) )
              break;
          }
        }
      }
      if ( !v164 )
      {
        v197 = v199;
        v198 = 0x400000000LL;
        if ( (_DWORD)v83 )
          sub_F33640((__int64)&v197, (__int64)&v191, v82, v83, (__int64)v79, v80);
        v201 = 0;
        if ( (unsigned __int8)sub_F386A0((__int64)&v205, (__int64)&v183, (__int64 *)&v176) )
        {
LABEL_212:
          v98 = (__int64)v176 + 40;
          goto LABEL_110;
        }
        v133 = (__int64)v176;
        ++v205;
        v180[0] = (__int64)v176;
        v134 = ((unsigned int)v206 >> 1) + 1;
        if ( (v206 & 1) != 0 )
        {
          v136 = 12;
          v135 = 4;
        }
        else
        {
          v135 = v208;
          v136 = 3 * v208;
        }
        if ( v136 <= 4 * v134 )
        {
          v135 *= 2;
        }
        else if ( v135 - (v134 + HIDWORD(v206)) > v135 >> 3 )
        {
          goto LABEL_227;
        }
        sub_F3B390((__int64)&v205, v135);
        sub_F386A0((__int64)&v205, (__int64)&v183, v180);
        v133 = v180[0];
        v134 = ((unsigned int)v206 >> 1) + 1;
LABEL_227:
        v157 = (__m128i *)v133;
        s2 = 0;
        v189 = 0;
        LODWORD(v206) = v206 & 1 | (2 * v134);
        v190 = 0;
        v137 = sub_F34140(v133, (__int64)&s2);
        v88 = v157;
        if ( !v137 )
          --HIDWORD(v206);
        goto LABEL_109;
      }
      v86 = *(_DWORD *)(v61 - 20) & 0x7FFFFFF;
      v87 = *(_QWORD *)(*(_QWORD *)(v66 + 32 * (2 - v86)) + 24LL);
      v197 = v199;
      v198 = 0x400000000LL;
      if ( (_DWORD)v83 )
        sub_F33640((__int64)&v197, (__int64)&v191, v86, v83, (__int64)v79, v80);
      v201 = v87;
      if ( (unsigned __int8)sub_F386A0((__int64)&v205, (__int64)&v183, (__int64 *)&v176) )
        goto LABEL_212;
      v92 = (__int64)v176;
      ++v205;
      v180[0] = (__int64)v176;
      v93 = ((unsigned int)v206 >> 1) + 1;
      if ( (v206 & 1) != 0 )
      {
        v95 = 12;
        v94 = 4;
      }
      else
      {
        v94 = v208;
        v95 = 3 * v208;
      }
      if ( v95 <= 4 * v93 )
      {
        v94 *= 2;
      }
      else if ( v94 - (v93 + HIDWORD(v206)) > v94 >> 3 )
      {
        goto LABEL_107;
      }
      sub_F3B390((__int64)&v205, v94);
      sub_F386A0((__int64)&v205, (__int64)&v183, v180);
      v92 = v180[0];
      v93 = ((unsigned int)v206 >> 1) + 1;
LABEL_107:
      v157 = (__m128i *)v92;
      s2 = 0;
      v189 = 0;
      LODWORD(v206) = v206 & 1 | (2 * v93);
      v190 = 0;
      v96 = sub_F34140(v92, (__int64)&s2);
      v88 = v157;
      if ( !v96 )
        --HIDWORD(v206);
LABEL_109:
      *v157 = _mm_loadu_si128(&v183);
      v88[1] = _mm_loadu_si128(&v184);
      v97 = v185;
      v88[5].m128i_i64[1] = 0;
      v98 = (__int64)&v88[2].m128i_i64[1];
      v88[2].m128i_i64[0] = v97;
      v88[2].m128i_i64[1] = (__int64)&v88[3].m128i_i64[1];
      v88[3].m128i_i64[0] = 0x400000000LL;
LABEL_110:
      a2 = (__int64)&v197;
      sub_F334E0(v98, &v197, (__int64)v88, v89, v90, v91);
      *(_QWORD *)(v98 + 48) = v201;
      if ( v197 != v199 )
        _libc_free(v197, &v197);
      v79 = (char *)v191;
LABEL_113:
      if ( v79 != &v193 )
      {
        v99 = v79;
        goto LABEL_115;
      }
LABEL_71:
      v61 = *(_QWORD *)(v61 + 8);
      if ( v170 == v61 )
        goto LABEL_116;
    }
    if ( !v164 )
      goto LABEL_113;
    v131 = (unsigned int)v203;
    v132 = (unsigned int)v203 + 1LL;
    if ( v132 > HIDWORD(v203) )
    {
      a2 = (__int64)v204;
      sub_C8D5F0((__int64)&v202, v204, v132, 8u, (__int64)v79, v80);
      v131 = (unsigned int)v203;
    }
    *(_QWORD *)&v202[8 * v131] = v66;
    v99 = v191;
    LODWORD(v203) = v203 + 1;
    if ( v191 == &v193 )
      goto LABEL_71;
LABEL_115:
    _libc_free(v99, a2);
    v61 = *(_QWORD *)(v61 + 8);
  }
  while ( v170 != v61 );
LABEL_116:
  v100 = (_QWORD **)v202;
  v2 = (_QWORD **)&v202[8 * (unsigned int)v203];
  v101 = v203;
  if ( v2 != (_QWORD **)v202 )
  {
    do
    {
      v102 = *v100++;
      sub_B43D60(v102);
    }
    while ( v2 != v100 );
LABEL_118:
    v101 = v203;
  }
LABEL_119:
  LOBYTE(v2) = v101 != 0;
  sub_F38610((__int64)&v205, a2);
  if ( (v206 & 1) == 0 )
  {
    a2 = 96LL * v208;
    sub_C7D6A0((__int64)v207, a2, 8);
  }
  if ( v202 != v204 )
    _libc_free(v202, a2);
  return (unsigned int)v2;
}
