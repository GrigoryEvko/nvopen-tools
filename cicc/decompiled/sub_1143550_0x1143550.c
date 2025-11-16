// Function: sub_1143550
// Address: 0x1143550
//
unsigned __int8 *__fastcall sub_1143550(__m128i *a1, __int64 a2, char *a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v6; // r14
  __int64 *v8; // r12
  __int64 v9; // rbx
  char v10; // dl
  __int64 v11; // rdx
  __int64 v12; // r9
  __int64 v13; // rax
  _BYTE *v14; // rbx
  _BYTE *v15; // r13
  unsigned int v16; // eax
  __int64 v17; // rdx
  __int64 v18; // r9
  _BYTE *v19; // r11
  __int64 v20; // rcx
  __int64 v21; // r8
  bool v22; // dl
  __int64 v23; // rcx
  unsigned __int8 *result; // rax
  __int64 v25; // rsi
  __m128i v26; // xmm1
  unsigned __int64 v27; // xmm2_8
  __int64 v28; // rax
  __m128i v29; // xmm3
  __int64 *v30; // r10
  __int64 v31; // rdx
  bool v32; // r15
  char v33; // al
  int v34; // r11d
  __int64 v35; // r15
  unsigned int v36; // r12d
  unsigned __int64 v37; // rax
  __int16 v38; // r12
  __int64 v39; // r14
  __m128i v40; // xmm5
  unsigned __int64 v41; // xmm6_8
  __int64 v42; // rax
  __m128i v43; // xmm7
  __int64 v44; // r12
  _BYTE *v45; // rax
  bool v46; // r15
  __int64 v47; // rax
  __int64 v48; // rdi
  unsigned int v49; // eax
  unsigned int v50; // r11d
  __int64 *v51; // rcx
  __int64 v52; // rdx
  __int64 **v53; // r8
  unsigned int **v54; // rdi
  __int64 v55; // rdx
  unsigned int v56; // eax
  __int64 v57; // rax
  __int64 v58; // rdi
  char v59; // al
  __int64 *v60; // rcx
  char v61; // dl
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 **v65; // r8
  unsigned int v66; // r11d
  unsigned int **v67; // r13
  unsigned int **v68; // rdi
  __int64 v69; // rax
  unsigned __int8 *v70; // rax
  unsigned int **v71; // r14
  __int64 v72; // r15
  unsigned int v73; // ebx
  bool v74; // bl
  char v75; // bl
  bool v76; // al
  char v77; // al
  unsigned int v78; // r15d
  bool v79; // cl
  char v80; // cl
  char v81; // al
  __int64 *v82; // r10
  __int64 *v83; // rcx
  char v84; // dl
  __int64 v85; // rax
  __int64 v86; // rax
  _BYTE *v87; // rax
  int v88; // eax
  _BYTE *v89; // rax
  __int64 v90; // rax
  __int64 v91; // r8
  bool v92; // al
  bool v93; // cl
  __int64 v94; // rdi
  __int64 *v95; // rcx
  char v96; // dl
  __int64 v97; // rax
  __int64 v98; // rax
  __int64 v99; // rdi
  char v100; // al
  __int64 *v101; // rcx
  char v102; // dl
  __int64 v103; // rax
  __int64 v104; // rax
  __int64 v105; // r12
  __int64 v106; // rdx
  _BYTE *v107; // rax
  bool v108; // al
  bool v109; // dl
  int v110; // eax
  unsigned int v111; // r12d
  unsigned __int64 v112; // rax
  __int64 *v113; // rax
  unsigned __int8 *v114; // r8
  char v115; // dl
  bool v116; // al
  __int64 v117; // r15
  _BYTE *v118; // rax
  unsigned int v119; // ebx
  __int64 v120; // r12
  unsigned __int8 *v121; // rax
  __int64 v122; // rbx
  _BYTE *v123; // rax
  int v124; // eax
  char v125; // r12
  unsigned int v126; // ebx
  unsigned __int8 *v127; // r13
  bool v128; // r8
  __int64 v129; // rax
  int v130; // eax
  unsigned int v131; // ebx
  unsigned __int8 *v132; // r15
  unsigned __int8 *v133; // rax
  unsigned int v134; // ebx
  char v135; // dl
  _BYTE *v136; // rax
  int v137; // eax
  bool v138; // cl
  unsigned int v139; // r12d
  unsigned __int8 *v140; // rbx
  __int64 v141; // r15
  unsigned __int8 *v142; // rax
  bool v143; // [rsp+3h] [rbp-FDh]
  int v144; // [rsp+4h] [rbp-FCh]
  unsigned int v145; // [rsp+8h] [rbp-F8h]
  __int64 v146; // [rsp+8h] [rbp-F8h]
  __int64 **v147; // [rsp+10h] [rbp-F0h]
  __int64 *v148; // [rsp+10h] [rbp-F0h]
  __int64 **v149; // [rsp+10h] [rbp-F0h]
  __int64 v150; // [rsp+10h] [rbp-F0h]
  __int64 v151; // [rsp+10h] [rbp-F0h]
  _BYTE *v152; // [rsp+18h] [rbp-E8h]
  __int64 *v153; // [rsp+18h] [rbp-E8h]
  unsigned int v154; // [rsp+18h] [rbp-E8h]
  unsigned int v155; // [rsp+18h] [rbp-E8h]
  __int64 *v156; // [rsp+18h] [rbp-E8h]
  char v157; // [rsp+18h] [rbp-E8h]
  bool v158; // [rsp+18h] [rbp-E8h]
  __int64 *v159; // [rsp+18h] [rbp-E8h]
  char v160; // [rsp+18h] [rbp-E8h]
  bool v161; // [rsp+18h] [rbp-E8h]
  _BYTE *v162; // [rsp+20h] [rbp-E0h]
  unsigned int v163; // [rsp+20h] [rbp-E0h]
  __int64 v164; // [rsp+20h] [rbp-E0h]
  unsigned int v165; // [rsp+20h] [rbp-E0h]
  __int64 v166; // [rsp+20h] [rbp-E0h]
  unsigned int v167; // [rsp+20h] [rbp-E0h]
  __int64 **v168; // [rsp+20h] [rbp-E0h]
  __int64 v169; // [rsp+20h] [rbp-E0h]
  unsigned int v170; // [rsp+20h] [rbp-E0h]
  unsigned int v171; // [rsp+20h] [rbp-E0h]
  unsigned int v172; // [rsp+20h] [rbp-E0h]
  __int64 v173; // [rsp+20h] [rbp-E0h]
  int v174; // [rsp+20h] [rbp-E0h]
  unsigned int v175; // [rsp+20h] [rbp-E0h]
  int v176; // [rsp+20h] [rbp-E0h]
  char v177; // [rsp+20h] [rbp-E0h]
  int v178; // [rsp+20h] [rbp-E0h]
  __int64 v179; // [rsp+28h] [rbp-D8h]
  __int64 v180; // [rsp+28h] [rbp-D8h]
  __int64 *v181; // [rsp+28h] [rbp-D8h]
  unsigned __int8 *v182; // [rsp+28h] [rbp-D8h]
  __int64 v183; // [rsp+28h] [rbp-D8h]
  unsigned __int8 *v184; // [rsp+28h] [rbp-D8h]
  unsigned int v185; // [rsp+28h] [rbp-D8h]
  unsigned __int8 *v186; // [rsp+28h] [rbp-D8h]
  __int64 v187; // [rsp+28h] [rbp-D8h]
  __int64 *v188; // [rsp+28h] [rbp-D8h]
  __int64 v189; // [rsp+28h] [rbp-D8h]
  __int64 v190; // [rsp+28h] [rbp-D8h]
  __int64 v191; // [rsp+28h] [rbp-D8h]
  __int64 v192; // [rsp+28h] [rbp-D8h]
  int v193; // [rsp+28h] [rbp-D8h]
  char *v194; // [rsp+28h] [rbp-D8h]
  __int64 v195; // [rsp+30h] [rbp-D0h] BYREF
  unsigned int v196; // [rsp+38h] [rbp-C8h]
  __int64 *v197; // [rsp+40h] [rbp-C0h] BYREF
  int v198; // [rsp+48h] [rbp-B8h]
  __int64 *v199; // [rsp+50h] [rbp-B0h] BYREF
  __int64 *v200; // [rsp+58h] [rbp-A8h]
  __int16 v201; // [rsp+70h] [rbp-90h]
  __m128i v202; // [rsp+80h] [rbp-80h] BYREF
  __m128i v203; // [rsp+90h] [rbp-70h]
  unsigned __int64 v204; // [rsp+A0h] [rbp-60h]
  __int64 v205; // [rsp+A8h] [rbp-58h]
  __m128i v206; // [rsp+B0h] [rbp-50h]
  __int64 v207; // [rsp+C0h] [rbp-40h]

  v6 = a2;
  v8 = (__int64 *)a1;
  v9 = (__int64)a3;
  v10 = *a3;
  if ( (unsigned __int8)v10 <= 0x1Cu )
  {
    if ( v10 != 5 || *(_WORD *)(v9 + 2) != 34 )
      goto LABEL_49;
LABEL_21:
    v23 = a2;
    a2 = v9;
    result = sub_1142B00((__int64)a1, v9, (unsigned __int8 *)a4, v23, a5);
    if ( result )
      return result;
    v10 = *(_BYTE *)v9;
    if ( *(_BYTE *)v9 != 86 )
      goto LABEL_4;
    goto LABEL_24;
  }
  if ( v10 == 63 )
    goto LABEL_21;
  if ( v10 != 86 )
    goto LABEL_4;
LABEL_24:
  a2 = v6;
  result = sub_111DB70(a1, v6, v9, (unsigned __int8 *)a4, a5);
  if ( result )
    return result;
  v10 = *(_BYTE *)v9;
LABEL_4:
  if ( v10 == 85 )
  {
    v11 = *(_QWORD *)(v9 - 32);
    if ( !v11 )
      goto LABEL_26;
    if ( *(_BYTE *)v11 || *(_QWORD *)(v11 + 24) != *(_QWORD *)(v9 + 80) || (*(_BYTE *)(v11 + 33) & 0x20) == 0 )
    {
LABEL_8:
      if ( v11 )
      {
        if ( !*(_BYTE *)v11 && *(_QWORD *)(v11 + 24) == *(_QWORD *)(v9 + 80) && *(_DWORD *)(v11 + 36) == 1 )
        {
          v12 = *(_QWORD *)(v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF));
          if ( v12 )
          {
            v13 = 32 * (1LL - (*(_DWORD *)(v9 + 4) & 0x7FFFFFF));
            if ( **(_BYTE **)(v9 + v13) <= 0x15u && a4 == v12 )
            {
              v162 = *(_BYTE **)(v9 + v13);
              v179 = *(_QWORD *)(v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF));
              v14 = (_BYTE *)sub_AD6530(*(_QWORD *)(v12 + 8), a2);
              v15 = (_BYTE *)sub_AD62B0(*(_QWORD *)(v179 + 8));
              v16 = sub_BCB060(*(_QWORD *)(v179 + 8));
              v18 = v179;
              v19 = v162;
              v196 = v16;
              if ( v16 > 0x40 )
              {
                a2 = 0;
                v152 = v162;
                v166 = v179;
                v185 = v16;
                sub_C43690((__int64)&v195, 0, 0);
                v18 = v166;
                v19 = v152;
                v20 = v185 - 1;
                v21 = 1LL << ((unsigned __int8)v185 - 1);
                if ( v196 > 0x40 )
                {
                  v17 = v195;
                  *(_QWORD *)(v195 + 8LL * ((v185 - 1) >> 6)) |= v21;
LABEL_18:
                  v180 = v18;
                  v22 = sub_AD7930(v19, a2, v17, v20, v21);
                  switch ( (int)v6 )
                  {
                    case ' ':
                    case '#':
                    case ')':
                      v71 = (unsigned int **)a1[2].m128i_i64[0];
                      if ( v22 )
                      {
                        LOWORD(v204) = 257;
                        v69 = sub_92B530(v71, 0x26u, v180, v15, (__int64)&v202);
LABEL_105:
                        v70 = sub_F162A0((__int64)v8, a5, v69);
                      }
                      else
                      {
                        LOWORD(v204) = 257;
                        sub_9865C0((__int64)&v197, (__int64)&v195);
                        sub_C46A40((__int64)&v197, 1);
                        v88 = v198;
                        v198 = 0;
                        LODWORD(v200) = v88;
                        v199 = v197;
                        v89 = (_BYTE *)sub_AD8D80(*(_QWORD *)(v180 + 8), (__int64)&v199);
                        v90 = sub_92B530(v71, 0x24u, v180, v89, (__int64)&v202);
                        v186 = sub_F162A0((__int64)a1, a5, v90);
                        sub_969240((__int64 *)&v199);
                        sub_969240((__int64 *)&v197);
                        v70 = v186;
                      }
                      v184 = v70;
                      sub_969240(&v195);
                      result = v184;
                      break;
                    case '!':
                    case '$':
                    case '&':
                      v67 = (unsigned int **)a1[2].m128i_i64[0];
                      if ( v22 )
                      {
                        v68 = (unsigned int **)a1[2].m128i_i64[0];
                        v201 = 257;
                        v69 = sub_92B530(v68, 0x28u, v180, v14, (__int64)&v199);
                      }
                      else
                      {
                        LOWORD(v204) = 257;
                        v87 = (_BYTE *)sub_AD8D80(*(_QWORD *)(v180 + 8), (__int64)&v195);
                        v69 = sub_92B530(v67, 0x22u, v180, v87, (__int64)&v202);
                      }
                      goto LABEL_105;
                    case '"':
                    case '(':
                      v69 = sub_AD6450(*(_QWORD *)(a5 + 8));
                      goto LABEL_105;
                    case '%':
                    case '\'':
                      v69 = sub_AD6400(*(_QWORD *)(a5 + 8));
                      goto LABEL_105;
                    default:
                      BUG();
                  }
                  return result;
                }
              }
              else
              {
                v195 = 0;
                v20 = v16 - 1;
                v21 = 1LL << ((unsigned __int8)v16 - 1);
              }
              v195 |= v21;
              goto LABEL_18;
            }
          }
        }
      }
LABEL_26:
      v25 = (unsigned int)(v6 - 32);
      v26 = _mm_loadu_si128(a1 + 7);
      v27 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
      v28 = a1[10].m128i_i64[0];
      v29 = _mm_loadu_si128(a1 + 9);
      v202 = _mm_loadu_si128(a1 + 6);
      v204 = v27;
      v207 = v28;
      v205 = a5;
      v203 = v26;
      v206 = v29;
      switch ( (int)v6 )
      {
        case ' ':
        case '#':
          v34 = 37;
          v10 = 85;
          goto LABEL_35;
        case '$':
          v10 = 85;
          goto LABEL_65;
        case '\'':
          v10 = 85;
          goto LABEL_79;
        case '(':
          v10 = 85;
          goto LABEL_77;
        default:
          v30 = &v195;
          goto LABEL_28;
      }
    }
    v56 = *(_DWORD *)(v11 + 36);
    if ( v56 > 0x14A )
    {
      if ( v56 - 365 > 1 )
        goto LABEL_8;
    }
    else if ( v56 <= 0x148 )
    {
      goto LABEL_8;
    }
    a2 = a5;
    result = sub_112D9E0(a1, a5, v9, (_BYTE *)a4, v6);
    if ( result )
      return result;
    v10 = *(_BYTE *)v9;
    v203.m128i_i8[0] = 0;
    v202.m128i_i64[0] = (__int64)&v197;
    v202.m128i_i64[1] = (__int64)&v199;
    if ( v10 != 42 )
      goto LABEL_55;
  }
  else
  {
    v203.m128i_i8[0] = 0;
    v202.m128i_i64[0] = (__int64)&v197;
    v202.m128i_i64[1] = (__int64)&v199;
    if ( v10 != 42 )
      goto LABEL_49;
  }
  if ( *(_QWORD *)(v9 - 64) )
  {
    a2 = *(_QWORD *)(v9 - 32);
    v197 = *(__int64 **)(v9 - 64);
    if ( (unsigned __int8)sub_991580((__int64)&v202.m128i_i64[1], a2) && v197 == (__int64 *)a4 )
      return (unsigned __int8 *)sub_1114370((__int64)a1, a4, (__int64)v199, v6);
    v10 = *(_BYTE *)v9;
LABEL_55:
    if ( v10 == 85 )
    {
      v11 = *(_QWORD *)(v9 - 32);
      goto LABEL_8;
    }
    goto LABEL_49;
  }
  v10 = 42;
LABEL_49:
  v25 = (unsigned int)(v6 - 32);
  v40 = _mm_loadu_si128(a1 + 7);
  v41 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v42 = a1[10].m128i_i64[0];
  v43 = _mm_loadu_si128(a1 + 9);
  v202 = _mm_loadu_si128(a1 + 6);
  v204 = v41;
  v207 = v42;
  v205 = a5;
  v203 = v40;
  v206 = v43;
  switch ( (int)v6 )
  {
    case ' ':
    case '#':
      v34 = 37;
      if ( v10 == 57 )
        goto LABEL_67;
      goto LABEL_35;
    case '$':
LABEL_65:
      v34 = 34;
      goto LABEL_66;
    case '\'':
LABEL_79:
      v34 = 41;
      goto LABEL_66;
    case '(':
LABEL_77:
      v34 = 38;
LABEL_66:
      if ( v10 != 57 )
        goto LABEL_39;
LABEL_67:
      v47 = *(_QWORD *)(v9 - 64);
      v48 = *(_QWORD *)(v9 - 32);
      if ( a4 == v47 && v48 )
      {
        v197 = *(__int64 **)(v9 - 32);
LABEL_71:
        v165 = v34;
        v195 = a4;
        if ( !sub_1131D50(v48, 0, (__int64)&v202, 0) )
        {
          v25 = 1;
          if ( sub_1131D50(v195, 1, (__int64)&v202, 0) )
          {
            v76 = sub_B532B0(v6);
            v50 = v165;
            if ( !v76 )
              goto LABEL_73;
            v25 = (__int64)&v202;
            v77 = sub_9B6260(v195, &v202, 0);
            v50 = v165;
            if ( v77 )
              goto LABEL_73;
          }
          goto LABEL_38;
        }
        LOBYTE(v49) = sub_B532B0(v6);
        v50 = v165;
        if ( !(_BYTE)v49 )
        {
LABEL_73:
          v51 = v197;
          v52 = v195;
          v53 = &v199;
          goto LABEL_74;
        }
        v91 = (__int64)v197;
        if ( *(_BYTE *)v197 == 17 )
        {
          v155 = v165;
          v169 = (__int64)v197;
          v92 = sub_986C60(v197 + 3, *((_DWORD *)v197 + 8) - 1);
          v91 = v169;
          v50 = v155;
          v93 = !v92;
          goto LABEL_148;
        }
        v157 = v49;
        v173 = v197[1];
        if ( (unsigned int)*(unsigned __int8 *)(v173 + 8) - 17 <= 1 && *(_BYTE *)v197 <= 0x15u )
        {
          v145 = v50;
          v150 = (__int64)v197;
          v113 = (__int64 *)sub_AD7630((__int64)v197, 0, v49);
          v114 = (unsigned __int8 *)v150;
          v50 = v145;
          v115 = v157;
          if ( v113 && *(_BYTE *)v113 == 17 )
          {
            v116 = sub_986C60(v113 + 3, *((_DWORD *)v113 + 8) - 1);
            v91 = (__int64)v197;
            v50 = v145;
            v93 = !v116;
            goto LABEL_148;
          }
          if ( *(_BYTE *)(v173 + 8) == 17 )
          {
            v124 = *(_DWORD *)(v173 + 32);
            v175 = v145;
            v159 = v8;
            v125 = v115;
            v144 = v124;
            v151 = v9;
            v126 = 0;
            v146 = a4;
            v127 = v114;
            v128 = 0;
            while ( v144 != v126 )
            {
              v143 = v128;
              v129 = sub_AD69F0(v127, v126);
              v128 = v143;
              if ( !v129 )
                goto LABEL_167;
              if ( *(_BYTE *)v129 != 13 )
              {
                if ( *(_BYTE *)v129 != 17 || sub_986C60((__int64 *)(v129 + 24), *(_DWORD *)(v129 + 32) - 1) )
                {
LABEL_167:
                  v50 = v175;
                  v8 = v159;
                  v9 = v151;
                  a4 = v146;
                  goto LABEL_168;
                }
                v128 = v125;
              }
              ++v126;
            }
            v93 = v128;
            v50 = v175;
            v8 = v159;
            v9 = v151;
            a4 = v146;
            v91 = (__int64)v197;
LABEL_148:
            if ( v93 )
            {
              v51 = (__int64 *)v91;
              v52 = v195;
              v53 = &v199;
              goto LABEL_74;
            }
            goto LABEL_169;
          }
LABEL_168:
          v91 = (__int64)v197;
        }
LABEL_169:
        v25 = (__int64)&v202;
        v172 = v50;
        if ( (unsigned __int8)sub_9AC470(v91, &v202, 0) )
        {
          v51 = v197;
          v52 = v195;
          v53 = &v199;
          v50 = v172;
LABEL_74:
          v54 = (unsigned int **)v8[4];
          v25 = v50;
          v201 = 257;
          v55 = sub_92B530(v54, v50, v52, v51, (__int64)v53);
          if ( v55 )
            return sub_F162A0((__int64)v8, a5, v55);
        }
LABEL_38:
        v10 = *(_BYTE *)v9;
        goto LABEL_39;
      }
      v10 = 57;
      if ( a4 == v48 && v47 )
      {
        v197 = *(__int64 **)(v9 - 64);
        v48 = v47;
        goto LABEL_71;
      }
LABEL_35:
      if ( (unsigned int)v25 > 1 )
      {
LABEL_39:
        v30 = &v195;
        goto LABEL_40;
      }
      v25 = a4;
      v163 = v34;
      v199 = 0;
      if ( (unsigned __int8)sub_995B10(&v199, a4) )
      {
        v57 = *(_QWORD *)(v9 + 16);
        if ( v57 )
        {
          if ( !*(_QWORD *)(v57 + 8) && *(_BYTE *)v9 == 58 )
          {
            if ( *(_QWORD *)(v9 - 64) )
            {
              v58 = *(_QWORD *)(v9 - 32);
              v195 = *(_QWORD *)(v9 - 64);
              if ( v58 )
              {
                v197 = (__int64 *)v58;
                v59 = sub_1131D50(v58, 0, (__int64)&v202, 0);
                v50 = v163;
                if ( v59 )
                {
                  v60 = (__int64 *)v8[4];
                  v61 = 0;
                  v62 = *(_QWORD *)(v195 + 16);
                  if ( v62 )
                    v61 = *(_QWORD *)(v62 + 8) == 0;
                  LOBYTE(v199) = 0;
                  v63 = sub_F13D80(v8, v195, v61, v60, &v199, 0);
                  v53 = &v199;
                  v50 = v163;
                  v52 = v63;
                  if ( v63 )
                    goto LABEL_97;
                }
                v94 = v195;
                v25 = 0;
                v170 = v50;
                v195 = (__int64)v197;
                v197 = (__int64 *)v94;
                if ( sub_1131D50(v94, 0, (__int64)&v202, 0) )
                {
                  v25 = v195;
                  v95 = (__int64 *)v8[4];
                  v96 = 0;
                  v97 = *(_QWORD *)(v195 + 16);
                  if ( v97 )
                    v96 = *(_QWORD *)(v97 + 8) == 0;
                  LOBYTE(v199) = 0;
                  v98 = sub_F13D80(v8, v195, v96, v95, &v199, 0);
                  v53 = &v199;
                  v50 = v170;
                  v52 = v98;
                  if ( v98 )
                  {
LABEL_97:
                    v195 = v52;
                    v51 = v197;
                    goto LABEL_74;
                  }
                }
                goto LABEL_38;
              }
            }
          }
        }
      }
      if ( !(unsigned __int8)sub_1112510(a4) )
        goto LABEL_38;
      v30 = &v195;
      v10 = *(_BYTE *)v9;
      v200 = (__int64 *)&v197;
      v64 = *(_QWORD *)(v9 + 16);
      v199 = &v195;
      if ( !v64 || *(_QWORD *)(v64 + 8) || v10 != 57 )
        goto LABEL_40;
      v25 = v9;
      if ( !(unsigned __int8)sub_11108F0(&v199, v9) )
        goto LABEL_102;
      v147 = v65;
      v153 = v30;
      v167 = v66;
      v81 = sub_1131D50((__int64)v197, 1, (__int64)&v202, 0);
      v50 = v167;
      v82 = v153;
      v53 = v147;
      if ( v81 )
      {
        v83 = (__int64 *)v8[4];
        v84 = 0;
        v85 = v197[2];
        if ( v85 )
          v84 = *(_QWORD *)(v85 + 8) == 0;
        v148 = v153;
        v154 = v167;
        v168 = v53;
        LOBYTE(v199) = 0;
        v86 = sub_F13D80(v8, (__int64)v197, v84, v83, v53, 0);
        v53 = v168;
        v50 = v154;
        v82 = v148;
        v51 = (__int64 *)v86;
        if ( v86 )
          goto LABEL_143;
      }
      v99 = v195;
      v25 = 1;
      v149 = v53;
      v156 = v82;
      v171 = v50;
      v195 = (__int64)v197;
      v197 = (__int64 *)v99;
      v100 = sub_1131D50(v99, 1, (__int64)&v202, 0);
      v30 = v156;
      if ( v100 )
      {
        v25 = (__int64)v197;
        v101 = (__int64 *)v8[4];
        v102 = 0;
        v103 = v197[2];
        if ( v103 )
          v102 = *(_QWORD *)(v103 + 8) == 0;
        LOBYTE(v199) = 0;
        v104 = sub_F13D80(v8, (__int64)v197, v102, v101, v149, 0);
        v53 = v149;
        v50 = v171;
        v51 = (__int64 *)v104;
        if ( v104 )
        {
LABEL_143:
          v197 = v51;
          v52 = v195;
          goto LABEL_74;
        }
        v10 = *(_BYTE *)v9;
        v30 = v156;
      }
      else
      {
LABEL_102:
        v10 = *(_BYTE *)v9;
      }
LABEL_40:
      if ( v10 != 48 || a4 != *(_QWORD *)(v9 - 64) )
        goto LABEL_28;
      v35 = *(_QWORD *)(v9 - 32);
      if ( *(_BYTE *)v35 == 17 )
      {
        v36 = *(_DWORD *)(v35 + 32);
        if ( v36 > 0x40 )
        {
          v188 = v30;
          v110 = sub_C444A0(v35 + 24);
          v30 = v188;
          if ( v36 - v110 > 0x40 )
            goto LABEL_46;
          v37 = **(_QWORD **)(v35 + 24);
        }
        else
        {
          v37 = *(_QWORD *)(v35 + 24);
        }
        if ( v37 > 1 )
          goto LABEL_46;
      }
      else
      {
        v105 = *(_QWORD *)(v35 + 8);
        v106 = (unsigned int)*(unsigned __int8 *)(v105 + 8) - 17;
        if ( (unsigned int)v106 <= 1 && *(_BYTE *)v35 <= 0x15u )
        {
          v25 = 0;
          v187 = (__int64)v30;
          v107 = sub_AD7630(*(_QWORD *)(v9 - 32), 0, v106);
          v30 = (__int64 *)v187;
          if ( v107 && *v107 == 17 )
          {
            v25 = (__int64)(v107 + 24);
            v108 = sub_1110BA0(v187, (__int64)(v107 + 24));
            v30 = (__int64 *)v187;
            v109 = v108;
            goto LABEL_165;
          }
          if ( *(_BYTE *)(v105 + 8) == 17 )
          {
            v174 = *(_DWORD *)(v105 + 32);
            if ( v174 )
            {
              v190 = v9;
              v109 = 0;
              v119 = 0;
              v120 = (__int64)v30;
              do
              {
                v25 = v119;
                v158 = v109;
                v121 = (unsigned __int8 *)sub_AD69F0((unsigned __int8 *)v35, v119);
                if ( !v121
                  || (v25 = *v121, v109 = v158, (_BYTE)v25 != 13)
                  && ((_BYTE)v25 != 17 || (v25 = (__int64)(v121 + 24), !(v109 = sub_1110BA0(v120, (__int64)(v121 + 24))))) )
                {
                  v9 = v190;
                  v30 = (__int64 *)v120;
                  goto LABEL_28;
                }
                ++v119;
              }
              while ( v174 != v119 );
              v9 = v190;
              v30 = (__int64 *)v120;
LABEL_165:
              if ( v109 )
                goto LABEL_46;
            }
          }
        }
      }
LABEL_28:
      v181 = v30;
      v32 = sub_B532A0(v6);
      if ( v32 )
        goto LABEL_63;
      v33 = *(_BYTE *)v9;
      if ( *(_BYTE *)v9 != 49 )
        goto LABEL_30;
      if ( a4 != *(_QWORD *)(v9 - 64) )
        goto LABEL_31;
      v31 = *(_QWORD *)(v9 - 32);
      if ( *(_BYTE *)v31 == 17 )
      {
        v111 = *(_DWORD *)(v31 + 32);
        if ( v111 > 0x40 )
        {
          v191 = *(_QWORD *)(v9 - 32);
          if ( v111 - (unsigned int)sub_C444A0(v31 + 24) > 0x40 )
            goto LABEL_46;
          v31 = v191;
          v112 = **(_QWORD **)(v191 + 24);
        }
        else
        {
          v112 = *(_QWORD *)(v31 + 24);
        }
        if ( v112 <= 1 )
          goto LABEL_31;
LABEL_46:
        v38 = sub_B52F50(v6);
        v39 = sub_AD6530(*(_QWORD *)(a4 + 8), v25);
        v201 = 257;
        result = (unsigned __int8 *)sub_BD2C40(72, unk_3F10FD0);
        if ( result )
        {
          v182 = result;
          sub_1113300((__int64)result, v38, a4, v39, (__int64)&v199);
          return v182;
        }
        return result;
      }
      v44 = *(_QWORD *)(v31 + 8);
      v164 = (__int64)v181;
      if ( (unsigned int)*(unsigned __int8 *)(v44 + 8) - 17 > 1 || *(_BYTE *)v31 > 0x15u )
        goto LABEL_31;
      v25 = 0;
      v183 = *(_QWORD *)(v9 - 32);
      v45 = sub_AD7630(v31, 0, v31);
      v31 = v183;
      if ( v45 && *v45 == 17 )
      {
        v25 = (__int64)(v45 + 24);
        v46 = sub_1110BA0(v164, (__int64)(v45 + 24));
      }
      else
      {
        if ( *(_BYTE *)(v44 + 8) != 17 )
          goto LABEL_63;
        v137 = *(_DWORD *)(v44 + 32);
        v194 = (char *)v9;
        v138 = v32;
        v139 = 0;
        v140 = (unsigned __int8 *)v31;
        v141 = v164;
        v178 = v137;
        while ( v178 != v139 )
        {
          v25 = v139;
          v161 = v138;
          v142 = (unsigned __int8 *)sub_AD69F0(v140, v139);
          v138 = v161;
          if ( !v142
            || (v25 = *v142, (_BYTE)v25 != 13)
            && ((_BYTE)v25 != 17 || (v25 = (__int64)(v142 + 24), !(v138 = sub_1110BA0(v141, (__int64)(v142 + 24))))) )
          {
            v9 = (__int64)v194;
            v33 = *v194;
            goto LABEL_30;
          }
          ++v139;
        }
        v9 = (__int64)v194;
        v46 = v138;
      }
      if ( v46 )
        goto LABEL_46;
LABEL_63:
      v33 = *(_BYTE *)v9;
LABEL_30:
      if ( v33 != 55 || a4 != *(_QWORD *)(v9 - 64) )
        goto LABEL_31;
      v31 = *(_QWORD *)(v9 - 32);
      if ( *(_BYTE *)v31 == 17 )
      {
        v78 = *(_DWORD *)(v31 + 32);
        if ( v78 <= 0x40 )
          v79 = *(_QWORD *)(v31 + 24) == 0;
        else
          v79 = v78 == (unsigned int)sub_C444A0(v31 + 24);
        v80 = !v79;
LABEL_134:
        if ( !v80 )
          goto LABEL_31;
        goto LABEL_46;
      }
      v117 = *(_QWORD *)(v31 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v117 + 8) - 17 <= 1 && *(_BYTE *)v31 <= 0x15u )
      {
        v25 = 0;
        v189 = *(_QWORD *)(v9 - 32);
        v118 = sub_AD7630(v189, 0, v31);
        v31 = v189;
        if ( v118 && *v118 == 17 )
        {
          v25 = (__int64)(v118 + 24);
          v80 = sub_1110B60((__int64)&v197, (__int64)(v118 + 24));
          goto LABEL_134;
        }
        if ( *(_BYTE *)(v117 + 8) == 17 )
        {
          v130 = *(_DWORD *)(v117 + 32);
          v192 = v9;
          v80 = 0;
          v131 = 0;
          v132 = (unsigned __int8 *)v31;
          v176 = v130;
          while ( v176 != v131 )
          {
            v25 = v131;
            v160 = v80;
            v133 = (unsigned __int8 *)sub_AD69F0(v132, v131);
            if ( !v133
              || (v25 = *v133, v80 = v160, (_BYTE)v25 != 13)
              && ((_BYTE)v25 != 17
               || (v25 = (__int64)(v133 + 24), (v80 = sub_1110B60((__int64)&v197, (__int64)(v133 + 24))) == 0)) )
            {
              v9 = v192;
              goto LABEL_31;
            }
            ++v131;
          }
          v9 = v192;
          goto LABEL_134;
        }
      }
LABEL_31:
      if ( (unsigned int)(v6 - 39) > 1 || *(_BYTE *)v9 != 56 || a4 != *(_QWORD *)(v9 - 64) )
        return 0;
      v72 = *(_QWORD *)(v9 - 32);
      if ( *(_BYTE *)v72 == 17 )
      {
        v73 = *(_DWORD *)(v72 + 32);
        if ( v73 <= 0x40 )
          v74 = *(_QWORD *)(v72 + 24) == 0;
        else
          v74 = v73 == (unsigned int)sub_C444A0(v72 + 24);
        v75 = !v74;
        goto LABEL_117;
      }
      v122 = *(_QWORD *)(v72 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v122 + 8) - 17 > 1 || *(_BYTE *)v72 > 0x15u )
        return 0;
      v25 = 0;
      v123 = sub_AD7630(v72, 0, v31);
      if ( !v123 || *v123 != 17 )
      {
        if ( *(_BYTE *)(v122 + 8) == 17 )
        {
          v193 = *(_DWORD *)(v122 + 32);
          v134 = 0;
          v135 = 0;
          while ( v193 != v134 )
          {
            v25 = v134;
            v177 = v135;
            v136 = (_BYTE *)sub_AD69F0((unsigned __int8 *)v72, v134);
            if ( !v136 )
              return 0;
            v135 = v177;
            if ( *v136 != 13 )
            {
              if ( *v136 != 17 )
                return 0;
              v25 = (__int64)(v136 + 24);
              v135 = sub_1110B60((__int64)&v197, (__int64)(v136 + 24));
              if ( !v135 )
                return 0;
            }
            ++v134;
          }
          v75 = v135;
          goto LABEL_117;
        }
        return 0;
      }
      v25 = (__int64)(v123 + 24);
      v75 = sub_1110B60((__int64)&v197, (__int64)(v123 + 24));
LABEL_117:
      if ( v75 )
        goto LABEL_46;
      return 0;
    default:
      goto LABEL_39;
  }
}
