// Function: sub_29C57C0
// Address: 0x29c57c0
//
__int64 __fastcall sub_29C57C0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 a6,
        void *a7,
        size_t a8,
        _DWORD *a9)
{
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rdx
  unsigned __int8 v12; // al
  __int64 v13; // rax
  __int64 v14; // rdx
  _QWORD *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned __int8 v19; // al
  __int64 v20; // rax
  __int64 v21; // rdx
  _QWORD *v22; // rax
  unsigned int v23; // r13d
  unsigned int v24; // r13d
  __int64 v25; // rbx
  _QWORD *v26; // r13
  __int64 v27; // r12
  _QWORD *i; // r14
  __int64 v29; // r15
  __int64 v30; // rsi
  __int64 v31; // rsi
  _QWORD *v32; // rax
  _QWORD *v33; // r14
  _QWORD *j; // rbx
  __int64 v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // r12
  __int64 v38; // rbx
  __int64 v39; // r15
  _QWORD *v40; // rax
  __int64 v41; // rax
  unsigned __int8 v42; // dl
  __int64 v43; // rdi
  __int64 v44; // rdx
  __int64 v45; // rsi
  int v46; // r13d
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  unsigned __int64 v50; // r10
  __int64 v51; // rdx
  unsigned __int64 v52; // r11
  bool v53; // cl
  __int64 *v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 *v59; // rax
  __int64 *v60; // rax
  __int64 v61; // rax
  int v62; // eax
  _QWORD *v63; // r13
  __int64 v64; // rax
  unsigned __int8 v65; // dl
  __int64 v66; // rdi
  __int64 v67; // rdx
  __int64 v68; // rsi
  int v69; // r15d
  __int64 v70; // rdx
  __int64 v71; // rax
  unsigned __int64 v72; // r12
  __int64 v73; // rdx
  char v74; // cl
  unsigned __int64 v75; // r8
  __int64 *v76; // rax
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 *v81; // rax
  __int64 *v82; // rax
  __int64 v83; // rax
  int v84; // eax
  __int64 v85; // rax
  unsigned __int8 v86; // dl
  __int64 v87; // rax
  _BYTE *v88; // rdi
  __int64 v89; // rax
  unsigned int v90; // esi
  __int64 v91; // rax
  unsigned int v93; // esi
  __int64 v94; // rax
  int v96; // r13d
  __int64 *v97; // r12
  __int64 *n; // rbx
  __int64 v99; // rdi
  int v100; // r13d
  char *v101; // r12
  char *v102; // rbx
  __int64 v103; // rdi
  _QWORD *v104; // r12
  void *v105; // rdi
  void *v106; // rdi
  const char *v107; // rsi
  __int64 v108; // rdi
  __int64 v109; // rdi
  _BYTE *v110; // rax
  int v113; // r12d
  __int64 m; // rdi
  __m128i *v115; // rdx
  __m128i v116; // xmm0
  unsigned int v117; // r12d
  __int64 v118; // rdi
  _BYTE *v119; // rax
  unsigned int v120; // r10d
  unsigned int v121; // esi
  int v122; // r12d
  unsigned __int64 v123; // r8
  __int64 v124; // rdx
  unsigned __int64 v125; // r8
  int v129; // r12d
  __int64 k; // rdi
  __m128i *v131; // rdx
  __m128i si128; // xmm0
  unsigned int v133; // r12d
  __int64 v134; // rdi
  _BYTE *v135; // rax
  unsigned int v136; // r10d
  unsigned int v137; // esi
  int v138; // r12d
  unsigned __int64 v139; // r8
  __int64 v140; // rdx
  unsigned __int64 v141; // r8
  int v144; // eax
  void *v145; // rdi
  void *v146; // r8
  unsigned __int8 *v147; // rax
  size_t v148; // rdx
  __int64 v149; // r8
  void *v150; // rdi
  _BYTE *v151; // rsi
  void *v152; // rdi
  __int64 v153; // rax
  unsigned __int8 v154; // dl
  __int64 v155; // rax
  _BYTE *v156; // rdi
  __int64 v157; // rax
  __int64 *v158; // rax
  __int64 v159; // rax
  __int64 v160; // rax
  __int64 *v161; // rax
  __int64 v162; // rax
  int v164; // [rsp+1Ch] [rbp-154h]
  int v165; // [rsp+28h] [rbp-148h]
  unsigned __int8 v166; // [rsp+2Eh] [rbp-142h]
  bool v167; // [rsp+2Fh] [rbp-141h]
  _QWORD *v169; // [rsp+38h] [rbp-138h]
  __int64 v171; // [rsp+48h] [rbp-128h]
  unsigned __int64 v172; // [rsp+48h] [rbp-128h]
  unsigned __int64 v173; // [rsp+48h] [rbp-128h]
  __int64 v174; // [rsp+50h] [rbp-120h]
  unsigned __int64 v175; // [rsp+58h] [rbp-118h]
  unsigned __int64 v176; // [rsp+58h] [rbp-118h]
  bool v177; // [rsp+58h] [rbp-118h]
  char v178; // [rsp+58h] [rbp-118h]
  _QWORD *v179; // [rsp+60h] [rbp-110h]
  __int64 v180; // [rsp+60h] [rbp-110h]
  unsigned __int64 v181; // [rsp+60h] [rbp-110h]
  unsigned __int64 v182; // [rsp+60h] [rbp-110h]
  __int64 v183; // [rsp+60h] [rbp-110h]
  unsigned __int64 *v184; // [rsp+68h] [rbp-108h]
  __int64 v185; // [rsp+68h] [rbp-108h]
  size_t v186; // [rsp+68h] [rbp-108h]
  __m128i v187; // [rsp+70h] [rbp-100h] BYREF
  __int64 v188; // [rsp+80h] [rbp-F0h]
  __int64 v189; // [rsp+88h] [rbp-E8h]
  __int64 v190; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v191; // [rsp+98h] [rbp-D8h]
  void *v192; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 v193; // [rsp+A8h] [rbp-C8h]
  _BYTE v194[48]; // [rsp+B0h] [rbp-C0h] BYREF
  int v195; // [rsp+E0h] [rbp-90h]
  void *v196; // [rsp+F0h] [rbp-80h] BYREF
  __int64 v197; // [rsp+F8h] [rbp-78h]
  _BYTE v198[48]; // [rsp+100h] [rbp-70h] BYREF
  int v199; // [rsp+130h] [rbp-40h]

  v187.m128i_i64[0] = a4;
  v187.m128i_i64[1] = a5;
  v166 = a6;
  v9 = sub_BA8DC0((__int64)a1, (__int64)"llvm.debugify", 13);
  if ( !v9 )
  {
    v161 = sub_29C0AE0();
    v162 = sub_A51340((__int64)v161, a7, a8);
    sub_904010(v162, ": Skipping module without debugify metadata\n");
    return 0;
  }
  v10 = v9;
  v11 = sub_B91A10(v9, 0);
  v12 = *(_BYTE *)(v11 - 16);
  if ( (v12 & 2) != 0 )
    v13 = *(_QWORD *)(v11 - 32);
  else
    v13 = v11 - 8LL * ((v12 >> 2) & 0xF) - 16;
  v14 = *(_QWORD *)(*(_QWORD *)v13 + 136LL);
  v15 = *(_QWORD **)(v14 + 24);
  if ( *(_DWORD *)(v14 + 32) > 0x40u )
    v15 = (_QWORD *)*v15;
  v164 = (int)v15;
  v16 = sub_B91A10(v10, 1u);
  v19 = *(_BYTE *)(v16 - 16);
  if ( (v19 & 2) != 0 )
    v20 = *(_QWORD *)(v16 - 32);
  else
    v20 = v16 - 8LL * ((v19 >> 2) & 0xF) - 16;
  v21 = *(_QWORD *)(*(_QWORD *)v20 + 136LL);
  v22 = *(_QWORD **)(v21 + 24);
  if ( *(_DWORD *)(v21 + 32) > 0x40u )
    v22 = (_QWORD *)*v22;
  v165 = (int)v22;
  if ( a9 )
  {
    if ( v187.m128i_i64[1] )
      a9 = (_DWORD *)sub_29C5520((__int64)a9, &v187);
    else
      a9 = 0;
  }
  v23 = (unsigned int)(v164 + 63) >> 6;
  v192 = v194;
  v193 = 0x600000000LL;
  if ( v23 > 6 )
  {
    sub_C8D5F0((__int64)&v192, v194, v23, 8u, v17, v18);
    memset(v192, 255, 8LL * v23);
    LODWORD(v193) = (unsigned int)(v164 + 63) >> 6;
  }
  else
  {
    if ( v23 && 8LL * v23 )
      memset(v194, 255, 8LL * v23);
    LODWORD(v193) = (unsigned int)(v164 + 63) >> 6;
  }
  v195 = v164;
  if ( (v164 & 0x3F) != 0 )
    *((_QWORD *)v192 + (unsigned int)v193 - 1) &= ~(-1LL << (v164 & 0x3F));
  v24 = (unsigned int)(v165 + 63) >> 6;
  v196 = v198;
  v197 = 0x600000000LL;
  if ( v24 > 6 )
  {
    sub_C8D5F0((__int64)&v196, v198, v24, 8u, v17, v18);
    memset(v196, 255, 8LL * v24);
  }
  else if ( v24 && 8LL * v24 )
  {
    memset(v198, 255, 8LL * v24);
  }
  LODWORD(v197) = (unsigned int)(v165 + 63) >> 6;
  v199 = v165;
  if ( (v165 & 0x3F) != 0 )
    *((_QWORD *)v196 + (unsigned int)v197 - 1) &= ~(-1LL << (v165 & 0x3F));
  v174 = a2;
  v167 = 0;
  if ( a3 != a2 )
  {
    v184 = (unsigned __int64 *)&v190;
    while ( 1 )
    {
      v25 = v174 - 56;
      if ( !v174 )
        v25 = 0;
      if ( !sub_B2FC80(v25) && !sub_B2FC80(v25) && !(unsigned __int8)sub_B2FC00((_BYTE *)v25) )
      {
        v26 = *(_QWORD **)(v25 + 80);
        v27 = v25 + 72;
        if ( (_QWORD *)(v25 + 72) == v26 )
        {
          v29 = (__int64)v184;
          i = 0;
        }
        else
        {
          if ( !v26 )
            BUG();
          while ( 1 )
          {
            i = (_QWORD *)v26[4];
            if ( i != v26 + 3 )
              break;
            v26 = (_QWORD *)v26[1];
            if ( (_QWORD *)v27 == v26 )
              break;
            if ( !v26 )
              BUG();
          }
          v29 = (__int64)v184;
        }
        while ( v26 != (_QWORD *)v27 )
        {
          if ( !i )
            BUG();
          if ( *((_BYTE *)i - 24) != 85
            || (v83 = *(i - 7)) == 0
            || *(_BYTE *)v83
            || *(_QWORD *)(v83 + 24) != i[7]
            || (*(_BYTE *)(v83 + 33) & 0x20) == 0
            || (v84 = *(_DWORD *)(v83 + 36), v84 != 68) && v84 != 71 )
          {
            v30 = i[3];
            v190 = v30;
            if ( v30 )
            {
              sub_B96E90(v29, v30, 1);
              if ( v190 )
              {
                if ( (unsigned int)sub_B10CE0(v29) )
                {
                  v144 = sub_B10CE0(v29);
                  *((_QWORD *)v192 + ((unsigned int)(v144 - 1) >> 6)) &= ~(1LL << ((unsigned __int8)v144 - 1));
                  v31 = v190;
                  if ( !v190 )
                    goto LABEL_50;
                  goto LABEL_49;
                }
                v31 = v190;
                if ( *((_BYTE *)i - 24) != 84 )
                {
                  if ( v190 )
                  {
LABEL_49:
                    sub_B91220(v29, v31);
                    goto LABEL_50;
                  }
LABEL_230:
                  if ( (_BYTE)qword_5008FC8 )
                    v145 = sub_CB7330();
                  else
                    v145 = sub_CB72A0();
                  sub_904010((__int64)v145, "WARNING: Instruction with empty DebugLoc in function ");
                  if ( (_BYTE)qword_5008FC8 )
                    v146 = sub_CB7330();
                  else
                    v146 = sub_CB72A0();
                  v185 = (__int64)v146;
                  v147 = (unsigned __int8 *)sub_BD5D20(v25);
                  v149 = v185;
                  v150 = *(void **)(v185 + 32);
                  if ( *(_QWORD *)(v185 + 24) - (_QWORD)v150 < v148 )
                  {
                    v149 = sub_CB6200(v185, v147, v148);
                  }
                  else if ( v148 )
                  {
                    v183 = v185;
                    v186 = v148;
                    memcpy(v150, v147, v148);
                    v149 = v183;
                    *(_QWORD *)(v183 + 32) += v186;
                  }
                  sub_904010(v149, " --");
                  if ( (_BYTE)qword_5008FC8 )
                    v151 = sub_CB7330();
                  else
                    v151 = sub_CB72A0();
                  sub_A69870((__int64)(i - 3), v151, 0);
                  if ( (_BYTE)qword_5008FC8 )
                    v152 = sub_CB7330();
                  else
                    v152 = sub_CB72A0();
                  sub_904010((__int64)v152, "\n");
LABEL_107:
                  v31 = v190;
                }
                if ( !v31 )
                  goto LABEL_50;
                goto LABEL_49;
              }
            }
            if ( *((_BYTE *)i - 24) != 84 )
              goto LABEL_230;
            goto LABEL_107;
          }
LABEL_50:
          for ( i = (_QWORD *)i[1]; ; i = (_QWORD *)v26[4] )
          {
            v32 = v26 - 3;
            if ( !v26 )
              v32 = 0;
            if ( i != v32 + 6 )
              break;
            v26 = (_QWORD *)v26[1];
            if ( (_QWORD *)v27 == v26 )
              break;
            if ( !v26 )
              BUG();
          }
        }
        v33 = *(_QWORD **)(v25 + 80);
        v184 = (unsigned __int64 *)v29;
        if ( (_QWORD *)v27 == v33 )
        {
          j = 0;
        }
        else
        {
          if ( !v33 )
            BUG();
          while ( 1 )
          {
            j = (_QWORD *)v33[4];
            if ( j != v33 + 3 )
              break;
            v33 = (_QWORD *)v33[1];
            if ( (_QWORD *)v27 == v33 )
              goto LABEL_28;
            if ( !v33 )
              BUG();
          }
        }
        if ( (_QWORD *)v27 != v33 )
          break;
      }
LABEL_28:
      v174 = *(_QWORD *)(v174 + 8);
      if ( a3 == v174 )
        goto LABEL_146;
    }
    v169 = (_QWORD *)v27;
    while ( 1 )
    {
      if ( !j )
        BUG();
      v35 = j[5];
      if ( !v35 )
        goto LABEL_78;
      v37 = sub_B14240(v35);
      if ( v37 == v36 )
        goto LABEL_78;
      while ( *(_BYTE *)(v37 + 32) )
      {
        v37 = *(_QWORD *)(v37 + 8);
        if ( v37 == v36 )
          goto LABEL_78;
      }
      if ( v36 == v37 )
        goto LABEL_78;
      v179 = j;
      v38 = v36;
      v39 = v37;
      if ( (unsigned __int8)(*(_BYTE *)(v37 + 64) - 1) > 1u )
        goto LABEL_76;
LABEL_88:
      v41 = sub_B12000(v39 + 72);
      v42 = *(_BYTE *)(v41 - 16);
      if ( (v42 & 2) != 0 )
      {
        v43 = *(_QWORD *)(*(_QWORD *)(v41 - 32) + 8LL);
        if ( v43 )
          goto LABEL_90;
      }
      else
      {
        v43 = *(_QWORD *)(v41 - 16 - 8LL * ((v42 >> 2) & 0xF) + 8);
        if ( v43 )
        {
LABEL_90:
          v43 = sub_B91420(v43);
          v45 = v44;
          goto LABEL_91;
        }
      }
      v45 = 0;
LABEL_91:
      if ( sub_C93C90(v43, v45, 0xAu, v184) || (v46 = v190, v190 != (unsigned int)v190) )
        v46 = -1;
      v47 = sub_B11F60(v39 + 80);
      if ( (unsigned int)((__int64)(*(_QWORD *)(v47 + 24) - *(_QWORD *)(v47 + 16)) >> 3)
        || (v48 = sub_B12A50(v39, 0)) == 0
        || (v171 = *(_QWORD *)(v48 + 8),
            v175 = sub_29C1BE0((__int64)a1, v171),
            v49 = sub_B13000(v39),
            v50 = v175,
            v190 = v49,
            v191 = v51,
            !v175)
        || !(_BYTE)v191 )
      {
LABEL_95:
        *((_QWORD *)v196 + ((unsigned int)(v46 - 1) >> 6)) &= ~(1LL << ((unsigned __int8)v46 - 1));
        goto LABEL_76;
      }
      v52 = v190;
      v53 = v175 != v190;
      if ( *(_BYTE *)(v171 + 8) == 12 )
      {
        v173 = v190;
        v85 = sub_B12000(v39 + 72);
        v86 = *(_BYTE *)(v85 - 16);
        if ( (v86 & 2) != 0 )
          v87 = *(_QWORD *)(v85 - 32);
        else
          v87 = v85 - 16 - 8LL * ((v86 >> 2) & 0xF);
        v88 = *(_BYTE **)(v87 + 24);
        if ( *v88 != 12 )
          goto LABEL_95;
        v89 = sub_AF2C80((__int64)v88);
        v188 = v89;
        if ( !BYTE4(v89) || (_DWORD)v89 )
          goto LABEL_95;
        v50 = v175;
        v52 = v173;
        v53 = v175 < v173;
      }
      if ( !v53 )
        goto LABEL_95;
      v167 = v53;
      v172 = v52;
      v176 = v50;
      v54 = sub_29C0AE0();
      v55 = sub_904010((__int64)v54, "ERROR: dbg.value operand has size ");
      v56 = sub_CB59D0(v55, v176);
      v57 = sub_904010(v56, ", but its variable has size ");
      v58 = sub_CB59D0(v57, v172);
      sub_904010(v58, ": ");
      v59 = sub_29C0AE0();
      sub_A691F0(v39, (__int64)v59, 0);
      v60 = sub_29C0AE0();
      sub_904010((__int64)v60, "\n");
LABEL_76:
      while ( 1 )
      {
        v39 = *(_QWORD *)(v39 + 8);
        if ( v39 == v38 )
          break;
        if ( !*(_BYTE *)(v39 + 32) )
        {
          if ( v38 == v39 )
            break;
          if ( (unsigned __int8)(*(_BYTE *)(v39 + 64) - 1) <= 1u )
            goto LABEL_88;
        }
      }
      j = v179;
LABEL_78:
      if ( *((_BYTE *)j - 24) != 85 )
        goto LABEL_79;
      v61 = *(j - 7);
      if ( !v61 )
        goto LABEL_79;
      if ( *(_BYTE *)v61 )
        goto LABEL_79;
      if ( *(_QWORD *)(v61 + 24) != j[7] )
        goto LABEL_79;
      if ( (*(_BYTE *)(v61 + 33) & 0x20) == 0 )
        goto LABEL_79;
      v62 = *(_DWORD *)(v61 + 36);
      if ( v62 != 68 && v62 != 71 )
        goto LABEL_79;
      v63 = j - 3;
      v64 = *(_QWORD *)(j[4 * (1LL - (*((_DWORD *)j - 5) & 0x7FFFFFF)) - 3] + 24LL);
      v65 = *(_BYTE *)(v64 - 16);
      if ( (v65 & 2) != 0 )
      {
        v66 = *(_QWORD *)(*(_QWORD *)(v64 - 32) + 8LL);
        if ( v66 )
          goto LABEL_118;
      }
      else
      {
        v66 = *(_QWORD *)(v64 - 16 - 8LL * ((v65 >> 2) & 0xF) + 8);
        if ( v66 )
        {
LABEL_118:
          v66 = sub_B91420(v66);
          v68 = v67;
          goto LABEL_119;
        }
      }
      v68 = 0;
LABEL_119:
      if ( sub_C93C90(v66, v68, 0xAu, v184) || (v69 = v190, v190 != (unsigned int)v190) )
        v69 = -1;
      v70 = *(_QWORD *)(v63[4 * (2LL - (*((_DWORD *)j - 5) & 0x7FFFFFF))] + 24LL);
      if ( (unsigned int)((__int64)(*(_QWORD *)(v70 + 24) - *(_QWORD *)(v70 + 16)) >> 3)
        || (v71 = sub_B58EB0((__int64)(j - 3), 0)) == 0
        || (v180 = *(_QWORD *)(v71 + 8),
            v72 = sub_29C1BE0((__int64)a1, v180),
            v190 = sub_B59530((__int64)(j - 3)),
            v191 = v73,
            !v72)
        || (v74 = v191) == 0 )
      {
LABEL_123:
        *((_QWORD *)v196 + ((unsigned int)(v69 - 1) >> 6)) &= ~(1LL << ((unsigned __int8)v69 - 1));
        goto LABEL_79;
      }
      v75 = v190;
      if ( *(_BYTE *)(v180 + 8) == 12 )
      {
        v153 = *(_QWORD *)(v63[4 * (1LL - (*((_DWORD *)j - 5) & 0x7FFFFFF))] + 24LL);
        v154 = *(_BYTE *)(v153 - 16);
        if ( (v154 & 2) != 0 )
          v155 = *(_QWORD *)(v153 - 32);
        else
          v155 = v153 - 16 - 8LL * ((v154 >> 2) & 0xF);
        v156 = *(_BYTE **)(v155 + 24);
        v178 = v191;
        v182 = v190;
        if ( *v156 != 12 )
          goto LABEL_123;
        v157 = sub_AF2C80((__int64)v156);
        v189 = v157;
        if ( !BYTE4(v157) )
          goto LABEL_123;
        if ( (_DWORD)v157 )
          goto LABEL_123;
        v75 = v182;
        v74 = v178;
        if ( v72 >= v182 )
          goto LABEL_123;
      }
      else if ( v72 == v190 )
      {
        goto LABEL_123;
      }
      v177 = v74;
      v181 = v75;
      v76 = sub_29C0AE0();
      v77 = sub_904010((__int64)v76, "ERROR: dbg.value operand has size ");
      v78 = sub_CB59D0(v77, v72);
      v79 = sub_904010(v78, ", but its variable has size ");
      v80 = sub_CB59D0(v79, v181);
      sub_904010(v80, ": ");
      v81 = sub_29C0AE0();
      sub_A69870((__int64)(j - 3), v81, 0);
      v82 = sub_29C0AE0();
      sub_904010((__int64)v82, "\n");
      v167 = v177;
LABEL_79:
      for ( j = (_QWORD *)j[1]; ; j = (_QWORD *)v33[4] )
      {
        v40 = v33 - 3;
        if ( !v33 )
          v40 = 0;
        if ( j != v40 + 6 )
          break;
        v33 = (_QWORD *)v33[1];
        if ( v169 == v33 )
          goto LABEL_28;
        if ( !v33 )
          BUG();
      }
      if ( v169 == v33 )
        goto LABEL_28;
    }
  }
LABEL_146:
  if ( v195 )
  {
    v90 = (unsigned int)(v195 - 1) >> 6;
    v91 = 0;
    while ( 1 )
    {
      _RDX = *((_QWORD *)v192 + v91);
      if ( v90 == (_DWORD)v91 )
        _RDX = (0xFFFFFFFFFFFFFFFFLL >> -(char)v195) & *((_QWORD *)v192 + v91);
      if ( _RDX )
        break;
      if ( v90 + 1 == ++v91 )
        goto LABEL_152;
    }
    __asm { tzcnt   rdx, rdx }
    v129 = _RDX + ((_DWORD)v91 << 6);
    if ( v129 != -1 )
    {
      if ( !(_BYTE)qword_5008FC8 )
        goto LABEL_225;
LABEL_207:
      for ( k = (__int64)sub_CB7330(); ; k = (__int64)sub_CB72A0() )
      {
        v131 = *(__m128i **)(k + 32);
        if ( *(_QWORD *)(k + 24) - (_QWORD)v131 <= 0x15u )
        {
          k = sub_CB6200(k, "WARNING: Missing line ", 0x16u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_439AB30);
          v131[1].m128i_i32[0] = 1852402720;
          v131[1].m128i_i16[2] = 8293;
          *v131 = si128;
          *(_QWORD *)(k + 32) += 22LL;
        }
        v133 = v129 + 1;
        v134 = sub_CB59D0(k, v133);
        v135 = *(_BYTE **)(v134 + 32);
        if ( *(_BYTE **)(v134 + 24) == v135 )
        {
          sub_CB6200(v134, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v135 = 10;
          ++*(_QWORD *)(v134 + 32);
        }
        if ( v133 == v195 )
          break;
        v136 = v133 >> 6;
        v137 = (unsigned int)(v195 - 1) >> 6;
        if ( v133 >> 6 > v137 )
          break;
        v138 = v133 & 0x3F;
        v139 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v138);
        if ( v138 == 0 )
          v139 = 0;
        v140 = v136;
        v141 = ~v139;
        while ( 1 )
        {
          _RAX = *((_QWORD *)v192 + v140);
          if ( v136 == (_DWORD)v140 )
            _RAX = v141 & *((_QWORD *)v192 + v140);
          if ( v137 == (_DWORD)v140 )
            _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v195;
          if ( _RAX )
            break;
          if ( v137 < (unsigned int)++v140 )
            goto LABEL_152;
        }
        __asm { tzcnt   rax, rax }
        v129 = ((_DWORD)v140 << 6) + _RAX;
        if ( v129 == -1 )
          break;
        if ( (_BYTE)qword_5008FC8 )
          goto LABEL_207;
LABEL_225:
        ;
      }
    }
  }
LABEL_152:
  if ( v199 )
  {
    v93 = (unsigned int)(v199 - 1) >> 6;
    v94 = 0;
    while ( 1 )
    {
      _RDX = *((_QWORD *)v196 + v94);
      if ( v93 == (_DWORD)v94 )
        _RDX = (0xFFFFFFFFFFFFFFFFLL >> -(char)v199) & *((_QWORD *)v196 + v94);
      if ( _RDX )
        break;
      if ( v93 + 1 == ++v94 )
        goto LABEL_158;
    }
    __asm { tzcnt   rdx, rdx }
    v113 = _RDX + ((_DWORD)v94 << 6);
    if ( v113 != -1 )
    {
      if ( !(_BYTE)qword_5008FC8 )
        goto LABEL_202;
LABEL_184:
      for ( m = (__int64)sub_CB7330(); ; m = (__int64)sub_CB72A0() )
      {
        v115 = *(__m128i **)(m + 32);
        if ( *(_QWORD *)(m + 24) - (_QWORD)v115 <= 0x19u )
        {
          m = sub_CB6200(m, "WARNING: Missing variable ", 0x1Au);
        }
        else
        {
          v116 = _mm_load_si128((const __m128i *)&xmmword_439AB30);
          qmemcpy(&v115[1], " variable ", 10);
          *v115 = v116;
          *(_QWORD *)(m + 32) += 26LL;
        }
        v117 = v113 + 1;
        v118 = sub_CB59D0(m, v117);
        v119 = *(_BYTE **)(v118 + 32);
        if ( *(_BYTE **)(v118 + 24) == v119 )
        {
          sub_CB6200(v118, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v119 = 10;
          ++*(_QWORD *)(v118 + 32);
        }
        if ( v117 == v199 )
          break;
        v120 = v117 >> 6;
        v121 = (unsigned int)(v199 - 1) >> 6;
        if ( v117 >> 6 > v121 )
          break;
        v122 = v117 & 0x3F;
        v123 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v122);
        if ( v122 == 0 )
          v123 = 0;
        v124 = v120;
        v125 = ~v123;
        while ( 1 )
        {
          _RAX = *((_QWORD *)v196 + v124);
          if ( v120 == (_DWORD)v124 )
            _RAX = v125 & *((_QWORD *)v196 + v124);
          if ( v121 == (_DWORD)v124 )
            _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v199;
          if ( _RAX )
            break;
          if ( v121 < (unsigned int)++v124 )
            goto LABEL_158;
        }
        __asm { tzcnt   rax, rax }
        v113 = ((_DWORD)v124 << 6) + _RAX;
        if ( v113 == -1 )
          break;
        if ( (_BYTE)qword_5008FC8 )
          goto LABEL_184;
LABEL_202:
        ;
      }
    }
  }
LABEL_158:
  if ( a9 )
  {
    v96 = 0;
    v97 = (__int64 *)v192;
    a9[3] += v164;
    for ( n = &v97[(unsigned int)v193]; n != v97; v96 += sub_39FAC40(v99) )
      v99 = *v97++;
    a9[2] += v96;
    v100 = 0;
    a9[1] += v165;
    v101 = (char *)v196;
    v102 = (char *)v196 + 8 * (unsigned int)v197;
    if ( v196 != v102 )
    {
      do
      {
        v103 = *(_QWORD *)v101;
        v101 += 8;
        v100 += sub_39FAC40(v103);
      }
      while ( v101 != v102 );
    }
    *a9 += v100;
  }
  if ( (_BYTE)qword_5008FC8 )
    v104 = sub_CB7330();
  else
    v104 = sub_CB72A0();
  v105 = (void *)v104[4];
  if ( a8 > v104[3] - (_QWORD)v105 )
  {
    sub_CB6200((__int64)v104, (unsigned __int8 *)a7, a8);
  }
  else if ( a8 )
  {
    memcpy(v105, a7, a8);
    v104[4] += a8;
  }
  if ( v187.m128i_i64[1] )
  {
    v158 = sub_29C0AE0();
    v159 = sub_904010((__int64)v158, " [");
    v160 = sub_A51340(v159, (const void *)v187.m128i_i64[0], v187.m128i_u64[1]);
    sub_904010(v160, "]");
  }
  if ( (_BYTE)qword_5008FC8 )
    v106 = sub_CB7330();
  else
    v106 = sub_CB72A0();
  v107 = "FAIL";
  v108 = sub_904010((__int64)v106, ": ");
  if ( !v167 )
    v107 = "PASS";
  v109 = sub_904010(v108, v107);
  v110 = *(_BYTE **)(v109 + 32);
  if ( (unsigned __int64)v110 >= *(_QWORD *)(v109 + 24) )
  {
    sub_CB5D20(v109, 10);
    if ( !a6 )
      goto LABEL_177;
LABEL_259:
    v166 = sub_29C1CB0(a1);
    goto LABEL_177;
  }
  *(_QWORD *)(v109 + 32) = v110 + 1;
  *v110 = 10;
  if ( a6 )
    goto LABEL_259;
LABEL_177:
  if ( v196 != v198 )
    _libc_free((unsigned __int64)v196);
  if ( v192 != v194 )
    _libc_free((unsigned __int64)v192);
  return v166;
}
