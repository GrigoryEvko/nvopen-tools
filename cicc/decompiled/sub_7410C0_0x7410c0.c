// Function: sub_7410C0
// Address: 0x7410c0
//
unsigned __int64 __fastcall sub_7410C0(
        __m128i *a1,
        __m128i *a2,
        __int64 a3,
        _QWORD *a4,
        _DWORD *a5,
        int a6,
        int *a7,
        __int64 *a8,
        __m128i *a9,
        _QWORD *a10)
{
  const __m128i *v12; // r12
  __int64 v14; // rdx
  char v15; // al
  int *v16; // rax
  unsigned __int64 v17; // r15
  int v19; // eax
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  _QWORD *v23; // rax
  int v24; // ecx
  __int64 v25; // rdx
  unsigned int v26; // r15d
  __m128i **v27; // rax
  __m128i *v28; // rdi
  __m128i *v29; // rax
  __int64 v30; // r12
  __m128i *v31; // r13
  __m128i *v32; // rcx
  __m128i *v33; // rdx
  __m128i *v34; // rax
  __m128i v35; // xmm0
  __int64 v36; // rax
  __int64 v37; // rbx
  _QWORD *v38; // rax
  __int64 v39; // rbx
  _QWORD *v40; // rax
  __int8 v41; // dl
  __int64 v42; // rax
  __int64 *v43; // r14
  int v44; // eax
  __int64 v45; // rbx
  _QWORD *v46; // rax
  char v47; // r11
  __int64 v48; // rdi
  _QWORD *v49; // rax
  int v50; // r15d
  __int64 v51; // r14
  unsigned int v52; // ebx
  __int64 v53; // rax
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 v56; // rcx
  __m128i **v57; // rax
  __m128i *v58; // rsi
  __int64 v59; // r15
  __int64 v60; // r14
  __m128i *v61; // rcx
  __m128i *v62; // rdx
  __m128i *v63; // rax
  __m128i v64; // xmm2
  __int64 v65; // r15
  __int64 v66; // rdx
  const __m128i *v67; // rbx
  const __m128i *v68; // r12
  __m128i *v69; // rdx
  const __m128i *v70; // rax
  __m128i *v71; // rax
  int v72; // eax
  _QWORD *v73; // r12
  __int64 v74; // rbx
  __int8 v75; // al
  __int64 v76; // r15
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 *v79; // rax
  unsigned __int8 v80; // r11
  __int64 v81; // r15
  __int64 *v82; // rax
  __int64 v83; // rdi
  __int64 *v84; // rax
  int v85; // eax
  int v86; // eax
  int v87; // eax
  int v88; // eax
  __int64 v89; // r12
  _BOOL4 v90; // eax
  __int64 v91; // rcx
  __int64 v92; // r8
  __int64 v93; // r9
  char v94; // r11
  __int64 v95; // r15
  int v96; // eax
  __int64 *v97; // rdi
  _QWORD *v98; // rbx
  __int64 *v99; // rax
  __int64 *v100; // rdx
  __int64 v101; // r12
  int v102; // eax
  __m128i *v103; // r8
  __m128i *v104; // rcx
  __int64 v105; // r15
  __int64 v106; // rax
  const __m128i *v107; // rdi
  const __m128i *v108; // rbx
  const __m128i *v109; // rsi
  __m128i *v110; // rax
  const __m128i *v111; // rdx
  __m128i *v112; // rsi
  __int8 *v113; // rax
  char v114; // dl
  __m128i *v115; // rax
  int v116; // eax
  __int64 v117; // rax
  const __m128i *v118; // rsi
  __m128i *v119; // rax
  const __m128i *v120; // rdx
  __int64 v121; // rdi
  _BOOL4 v122; // eax
  __int64 v123; // r8
  _BOOL4 v124; // eax
  _BOOL4 v125; // eax
  int v126; // eax
  int v127; // eax
  int v128; // eax
  int v129; // eax
  __int64 v130; // r12
  __int64 v131; // rax
  char v132; // r11
  __int64 v133; // rax
  __int64 v134; // rax
  __m128i *v135; // rax
  int v136; // eax
  __int64 v137; // rax
  __int64 v138; // rax
  __m128i *v139; // rax
  __int64 v140; // rsi
  __m128i *v141; // r9
  const __m128i *v142; // rdi
  __m128i *v143; // r8
  char v144; // dl
  __int64 v145; // rax
  __int64 v146; // rax
  __m128i *v147; // rax
  int v148; // eax
  __int64 v149; // rax
  __int64 v150; // rcx
  __m128i *v151; // rax
  int v152; // eax
  __int64 v153; // rax
  _QWORD *v154; // rax
  const __m128i *v155; // rdi
  __m128i *v156; // r12
  __int64 v157; // rdi
  __m128i *v158; // rax
  int v159; // eax
  __int64 v160; // rax
  __m128i *v161; // rax
  int v162; // eax
  __int64 v163; // rax
  __int64 v164; // rax
  __m128i *v165; // rsi
  __m128i *v166; // rdi
  __int64 i; // rcx
  __m128i *v168; // rax
  int v169; // eax
  __int64 v170; // rax
  int v171; // eax
  int v172; // eax
  __int64 v173; // r8
  int v174; // eax
  __int64 v175; // rax
  _QWORD *v176; // rax
  int v177; // eax
  __int64 v178; // rdi
  _BOOL4 v179; // eax
  const __m128i *v180; // rax
  __m128i *v181; // rsi
  __m128i *v182; // rdi
  __int64 j; // rcx
  bool v184; // zf
  __int64 v185; // [rsp+8h] [rbp-138h]
  __int64 v186; // [rsp+8h] [rbp-138h]
  unsigned __int8 v187; // [rsp+8h] [rbp-138h]
  char v188; // [rsp+8h] [rbp-138h]
  unsigned __int8 v189; // [rsp+18h] [rbp-128h]
  char v190; // [rsp+18h] [rbp-128h]
  char v191; // [rsp+18h] [rbp-128h]
  char v192; // [rsp+18h] [rbp-128h]
  char v193; // [rsp+18h] [rbp-128h]
  char v194; // [rsp+18h] [rbp-128h]
  char v195; // [rsp+18h] [rbp-128h]
  unsigned __int8 v196; // [rsp+18h] [rbp-128h]
  unsigned __int8 v197; // [rsp+18h] [rbp-128h]
  __int64 v198; // [rsp+18h] [rbp-128h]
  __int64 v199; // [rsp+18h] [rbp-128h]
  __int64 v200; // [rsp+28h] [rbp-118h]
  unsigned int v201; // [rsp+28h] [rbp-118h]
  unsigned __int8 v202; // [rsp+30h] [rbp-110h]
  __int64 v203; // [rsp+30h] [rbp-110h]
  char v204; // [rsp+38h] [rbp-108h]
  __int64 v205; // [rsp+38h] [rbp-108h]
  char v206; // [rsp+38h] [rbp-108h]
  char v207; // [rsp+38h] [rbp-108h]
  char v208; // [rsp+38h] [rbp-108h]
  __int64 v209; // [rsp+40h] [rbp-100h]
  __int64 v210; // [rsp+48h] [rbp-F8h]
  __int64 v211; // [rsp+48h] [rbp-F8h]
  const __m128i *v212; // [rsp+48h] [rbp-F8h]
  __int64 v213; // [rsp+48h] [rbp-F8h]
  const __m128i *v214; // [rsp+48h] [rbp-F8h]
  __int64 v215; // [rsp+48h] [rbp-F8h]
  int v217; // [rsp+50h] [rbp-F0h]
  __int64 v218; // [rsp+50h] [rbp-F0h]
  unsigned __int8 v220; // [rsp+58h] [rbp-E8h]
  unsigned __int8 v221; // [rsp+58h] [rbp-E8h]
  unsigned __int8 v222; // [rsp+58h] [rbp-E8h]
  unsigned __int8 v223; // [rsp+58h] [rbp-E8h]
  char v224; // [rsp+58h] [rbp-E8h]
  __int64 v225; // [rsp+58h] [rbp-E8h]
  char v226; // [rsp+58h] [rbp-E8h]
  char v227; // [rsp+58h] [rbp-E8h]
  char v228; // [rsp+58h] [rbp-E8h]
  char v229; // [rsp+58h] [rbp-E8h]
  bool v230; // [rsp+58h] [rbp-E8h]
  int v231; // [rsp+6Ch] [rbp-D4h] BYREF
  int v232; // [rsp+70h] [rbp-D0h] BYREF
  _BOOL4 v233; // [rsp+74h] [rbp-CCh] BYREF
  __m128i *v234; // [rsp+78h] [rbp-C8h] BYREF
  __m128i *v235; // [rsp+80h] [rbp-C0h] BYREF
  __m128i *v236; // [rsp+88h] [rbp-B8h] BYREF
  __int64 *v237; // [rsp+90h] [rbp-B0h] BYREF
  __int64 *v238; // [rsp+98h] [rbp-A8h] BYREF
  __int64 *v239; // [rsp+A0h] [rbp-A0h] BYREF
  __m128i *v240; // [rsp+A8h] [rbp-98h] BYREF
  __m128i *v241; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v242; // [rsp+B8h] [rbp-88h]
  __int64 v243; // [rsp+C0h] [rbp-80h]
  __m128i v244; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v245; // [rsp+E0h] [rbp-60h]
  __m128i *v246; // [rsp+E8h] [rbp-58h]
  __int64 v247; // [rsp+F0h] [rbp-50h]
  int v248; // [rsp+F8h] [rbp-48h]
  __int64 *v249; // [rsp+100h] [rbp-40h]
  char v250; // [rsp+108h] [rbp-38h]

  v12 = a2;
  v234 = (__m128i *)sub_724DC0();
  v235 = (__m128i *)sub_724DC0();
  v236 = (__m128i *)sub_724DC0();
  *a10 = 0;
  if ( unk_4F07734 && (unsigned int)sub_696370((__int64)a1) )
  {
    sub_6E3D60((__int64)&v244);
    v246 = a2;
    v248 = a6;
    v247 = a3;
    v249 = a8;
    v17 = sub_6DD8E0(a1->m128i_i64, a4, (__int64)&v244, a9);
    v16 = a7;
    if ( !v250 )
    {
LABEL_11:
      v19 = *v16;
      goto LABEL_12;
    }
    goto LABEL_6;
  }
  switch ( a1[1].m128i_i8[8] )
  {
    case 1:
      v47 = a1[3].m128i_i8[8];
      if ( v47 == 116 )
      {
        v17 = sub_7432A0(a1[4].m128i_i64[1], (_DWORD)a2, a3, (_DWORD)a4, (_DWORD)a5, a6, (__int64)a7, (__int64)a8);
        v19 = *a7;
        goto LABEL_12;
      }
      switch ( v47 )
      {
        case 0:
        case 5:
        case 25:
        case 28:
        case 29:
        case 53:
        case 54:
        case 55:
        case 56:
        case 57:
        case 71:
        case 72:
        case 87:
        case 88:
        case 103:
          break;
        case 26:
        case 27:
        case 39:
        case 40:
        case 41:
        case 42:
        case 43:
        case 58:
        case 59:
        case 60:
        case 61:
        case 62:
        case 63:
          v75 = a1[3].m128i_i8[9];
          if ( v75 != 2 && v75 != 14 )
            goto LABEL_5;
          break;
        default:
          goto LABEL_5;
      }
      v238 = 0;
      v76 = a1[4].m128i_i64[1];
      v239 = 0;
      v211 = 0;
      v77 = *(_QWORD *)(v76 + 16);
      v241 = 0;
      v218 = v77;
      v78 = a1->m128i_i64[0];
      v244.m128i_i64[0] = 0;
      v209 = v78;
      if ( v47 == 5 )
      {
        v106 = sub_8A2270(v78, (_DWORD)a2, a3, (_DWORD)a5, a6, (_DWORD)a7, (__int64)a8);
        v47 = 5;
        v211 = v106;
        if ( *a7 )
          goto LABEL_7;
      }
      v204 = v47;
      v79 = (__int64 *)sub_7410C0(
                         v76,
                         (_DWORD)a2,
                         a3,
                         v211,
                         (_DWORD)a5,
                         a6,
                         (__int64)a7,
                         (__int64)a8,
                         (__int64)v234,
                         (__int64)&v240);
      v80 = v204;
      v237 = v79;
      if ( *a7 )
      {
        v81 = 0;
        if ( v218 )
          goto LABEL_99;
LABEL_175:
        v203 = 0;
LABEL_113:
        v88 = *a7;
        goto LABEL_114;
      }
      if ( v79 )
      {
        v81 = *v79;
      }
      else
      {
        v115 = v240;
        if ( !v240 )
          v115 = v234;
        v81 = v115[8].m128i_i64[0];
        v116 = sub_8D32E0(v81);
        v80 = v204;
        if ( v116 )
        {
          v117 = sub_8D46C0(v81);
          v80 = v204;
          v81 = v117;
        }
      }
      if ( !v218 )
      {
        if ( v81 )
        {
          v223 = v80;
          v102 = sub_8D2660(v81);
          v80 = v223;
          if ( v102 )
          {
            if ( v223 != 103 )
            {
              v200 = 0;
              v205 = 0;
              v203 = 0;
              goto LABEL_161;
            }
          }
        }
        goto LABEL_175;
      }
LABEL_99:
      v202 = v80;
      v82 = (__int64 *)sub_7410C0(
                         v218,
                         (_DWORD)a2,
                         a3,
                         0,
                         (_DWORD)a5,
                         a6,
                         (__int64)a7,
                         (__int64)a8,
                         (__int64)v235,
                         (__int64)&v241);
      v205 = 0;
      v238 = v82;
      v80 = v202;
      if ( !*a7 )
      {
        if ( v82 )
        {
          v205 = *v82;
        }
        else
        {
          v147 = v241;
          if ( !v241 )
            v147 = v235;
          v205 = v147[8].m128i_i64[0];
          v148 = sub_8D32E0(v205);
          v80 = v202;
          if ( v148 )
          {
            v149 = sub_8D46C0(v205);
            v80 = v202;
            v205 = v149;
          }
        }
      }
      v200 = 0;
      v83 = *(_QWORD *)(v218 + 16);
      v203 = v83;
      if ( v83 )
      {
        v189 = v80;
        v84 = (__int64 *)sub_7410C0(
                           v83,
                           (_DWORD)a2,
                           a3,
                           0,
                           (_DWORD)a5,
                           a6,
                           (__int64)a7,
                           (__int64)a8,
                           (__int64)v236,
                           (__int64)&v244);
        v80 = v189;
        v239 = v84;
        if ( !*a7 )
        {
          if ( v84 )
          {
            v200 = *v84;
          }
          else
          {
            v158 = (__m128i *)v244.m128i_i64[0];
            if ( !v244.m128i_i64[0] )
              v158 = v236;
            v200 = v158[8].m128i_i64[0];
            v159 = sub_8D32E0(v200);
            v80 = v189;
            if ( v159 )
            {
              v160 = sub_8D46C0(v200);
              v80 = v189;
              v200 = v160;
            }
          }
        }
      }
      if ( !v81 || (v220 = v80, v85 = sub_8D2660(v81), v80 = v220, !v85) || v220 == 103 )
      {
        if ( !v205 || (v221 = v80, v86 = sub_8D2660(v205), v80 = v221, !v86) )
        {
          if ( !v200 )
            goto LABEL_113;
          v222 = v80;
          v87 = sub_8D2660(v200);
          v80 = v222;
          if ( !v87 )
            goto LABEL_113;
        }
      }
LABEL_161:
      v103 = 0;
      if ( !v237 )
      {
        v103 = v240;
        if ( !v240 )
          v103 = v234;
      }
      a2 = 0;
      if ( !v238 )
      {
        a2 = v241;
        if ( !v241 )
          a2 = v235;
      }
      v104 = 0;
      if ( !v239 )
      {
        v104 = (__m128i *)v244.m128i_i64[0];
        if ( !v244.m128i_i64[0] )
          v104 = v236;
      }
      if ( *a7 )
        goto LABEL_7;
      if ( v80 > 0x3Fu )
      {
        if ( v80 != 103 )
        {
          if ( v80 > 0x67u )
            goto LABEL_5;
          if ( v80 > 0x58u )
          {
            if ( v80 != 91 )
              goto LABEL_5;
          }
          else if ( v80 <= 0x56u )
          {
            goto LABEL_5;
          }
          goto LABEL_115;
        }
        v224 = 103;
        if ( !(unsigned int)sub_728620(v205, (__int64)a2, v200, (__int64)v104) )
          goto LABEL_5;
LABEL_178:
        v80 = v224;
        v88 = *a7;
LABEL_114:
        if ( v88 )
          goto LABEL_7;
        goto LABEL_115;
      }
      if ( v80 > 0x39u )
      {
        v150 = (__int64)a2;
        a2 = v103;
        v224 = v80;
        if ( !(unsigned int)sub_728620(v81, (__int64)v103, v205, v150) )
          goto LABEL_5;
        goto LABEL_178;
      }
      if ( v80 > 0x1Du || ((1LL << v80) & 0x22000021) == 0 )
        goto LABEL_5;
LABEL_115:
      v201 = v80;
      if ( v237 )
      {
        v89 = *v237;
      }
      else
      {
        v135 = v240;
        if ( !v240 )
          v135 = v234;
        v89 = v135[8].m128i_i64[0];
        v196 = v80;
        v136 = sub_8D32E0(v89);
        v80 = v196;
        if ( v136 )
        {
          v137 = sub_8D46C0(v89);
          v80 = v196;
          v89 = v137;
        }
      }
      while ( *(_BYTE *)(v89 + 140) == 12 )
        v89 = *(_QWORD *)(v89 + 160);
      if ( v218 )
      {
        if ( v238 )
        {
          v95 = *v238;
        }
        else
        {
          v161 = v241;
          if ( !v241 )
            v161 = v235;
          v95 = v161[8].m128i_i64[0];
          v197 = v80;
          v162 = sub_8D32E0(v95);
          v80 = v197;
          if ( v162 )
          {
            v163 = sub_8D46C0(v95);
            v80 = v197;
            v95 = v163;
          }
        }
        while ( *(_BYTE *)(v95 + 140) == 12 )
          v95 = *(_QWORD *)(v95 + 160);
        v123 = 0;
        if ( v203 )
        {
          if ( v239 )
          {
            v123 = *v239;
          }
          else
          {
            v168 = (__m128i *)v244.m128i_i64[0];
            if ( !v244.m128i_i64[0] )
              v168 = v236;
            v187 = v80;
            v198 = v168[8].m128i_i64[0];
            v169 = sub_8D32E0(v198);
            v123 = v198;
            v80 = v187;
            if ( v169 )
            {
              v170 = sub_8D46C0(v198);
              v80 = v187;
              v123 = v170;
            }
          }
          while ( *(_BYTE *)(v123 + 140) == 12 )
            v123 = *(_QWORD *)(v123 + 160);
        }
        v186 = v123;
        v191 = v80;
        v124 = sub_7306C0(v89);
        v94 = v191;
        v92 = v186;
        if ( v124 || (v125 = sub_7306C0(v95), v94 = v191, v92 = v186, v125) )
        {
          if ( !v94 )
          {
LABEL_126:
            if ( v203 )
              goto LABEL_127;
            goto LABEL_233;
          }
          goto LABEL_123;
        }
        if ( !v203 )
        {
LABEL_233:
          switch ( v94 )
          {
            case '\'':
            case '(':
            case ')':
            case '*':
            case '+':
            case '7':
            case '8':
            case '9':
              goto LABEL_237;
            case ',':
            case '-':
              sub_721090();
            case '5':
            case '6':
              v195 = v94;
              v209 = sub_8D6540(v89);
              sub_73E000((__int64 *)&v237, v234, (const __m128i **)&v240, v209, a5);
              v132 = v195;
              goto LABEL_242;
            case ':':
            case ';':
            case '<':
            case '=':
            case '>':
            case '?':
              v194 = v94;
              v128 = sub_8D2660(v89);
              v94 = v194;
              if ( !v128 )
              {
                v129 = sub_8D2660(v95);
                v94 = v194;
                if ( !v129 )
                {
                  v130 = sub_6E8B20(v89, v95);
                  sub_73E000((__int64 *)&v237, v234, (const __m128i **)&v240, v130, a5);
                  a2 = v235;
                  sub_73E000((__int64 *)&v238, v235, (const __m128i **)&v241, v130, a5);
                  v131 = sub_72C390();
                  v94 = v194;
                  v209 = v131;
                }
              }
              goto LABEL_129;
            case 'G':
            case 'H':
              v192 = v94;
              v126 = sub_8D2660(v89);
              v94 = v192;
              if ( !v126 )
              {
                v127 = sub_8D2660(v95);
                v94 = v192;
                if ( !v127 )
                {
LABEL_237:
                  v193 = v94;
                  v209 = sub_6E8B20(v89, v95);
                  sub_73E000((__int64 *)&v237, v234, (const __m128i **)&v240, v209, a5);
                  a2 = v235;
                  sub_73E000((__int64 *)&v238, v235, (const __m128i **)&v241, v209, a5);
                  v94 = v193;
                }
              }
              break;
            case 'J':
            case 'K':
            case 'L':
            case 'M':
            case 'N':
            case 'Q':
            case 'R':
            case 'S':
              v206 = v94;
              a2 = v235;
              v209 = sub_6E8B20(v89, v95);
              sub_73E000((__int64 *)&v238, v235, (const __m128i **)&v241, v209, a5);
              v94 = v206;
              break;
            case 'O':
            case 'P':
              v208 = v94;
              v134 = sub_8D6540(v89);
              v132 = v208;
              v209 = v134;
LABEL_242:
              v207 = v132;
              v133 = sub_8D6540(v95);
              a2 = v235;
              sub_73E000((__int64 *)&v238, v235, (const __m128i **)&v241, v133, a5);
              v94 = v207;
              break;
            default:
              goto LABEL_129;
          }
          goto LABEL_129;
        }
        v178 = v186;
        v188 = v191;
        v199 = v92;
        v179 = sub_7306C0(v178);
        v92 = v199;
        v94 = v188;
        if ( !v179 )
        {
LABEL_127:
          if ( v95 == v92 || v94 != 103 )
            goto LABEL_129;
          a2 = (__m128i *)v92;
          v215 = v92;
          v171 = sub_8DED30(v95, v92, 1);
          v92 = v215;
          v94 = 103;
          if ( !v171 )
          {
            v172 = sub_8D2660(v95);
            v173 = v215;
            if ( v172 || (v174 = sub_8D2660(v215), v173 = v215, v174) )
            {
              if ( (unsigned int)sub_8D2690(v173) )
                v175 = sub_72C4C0();
              else
                v175 = sub_72C570();
              v209 = v175;
            }
            else
            {
              v209 = sub_6E8B20(v95, v215);
            }
            sub_73E000((__int64 *)&v238, v235, (const __m128i **)&v241, v209, a5);
            a2 = v236;
            sub_73E000((__int64 *)&v239, v236, (const __m128i **)&v244, v209, a5);
            v94 = 103;
          }
LABEL_271:
          if ( *a7 )
            goto LABEL_7;
LABEL_132:
          v97 = v237;
          if ( v237 )
          {
            if ( v94 || (*((_BYTE *)v237 + 25) & 3) == 0 )
              goto LABEL_135;
            a2 = a9;
            if ( (unsigned int)sub_717510(v237, a9, 1, v91, v92, v93) )
            {
              v17 = 0;
              *a10 = 0;
              goto LABEL_18;
            }
            goto LABEL_340;
          }
LABEL_286:
          v155 = v240;
          if ( v238 || (v17 = (unsigned __int64)v239) != 0 )
          {
LABEL_287:
            v156 = v234;
LABEL_288:
            if ( v155 )
            {
              v98 = sub_730690((__int64)v155);
              v157 = v155[8].m128i_i64[0];
            }
            else
            {
              v176 = sub_73A720(v156, (__int64)a2);
              v157 = v156[8].m128i_i64[0];
              v98 = v176;
            }
            if ( (unsigned int)sub_8D32E0(v157) )
              *((_BYTE *)v98 + 25) |= 1u;
LABEL_136:
            v237 = v98;
            v99 = v238;
            if ( !v218 )
            {
LABEL_144:
              v98[2] = v99;
              if ( v99 )
                v99[2] = (__int64)v239;
              v17 = (unsigned __int64)sub_730FF0(a1);
              sub_73D8E0(v17, v201, v209, a1[1].m128i_i8[9] & 1, (__int64)v237);
              sub_730580((__int64)a1, v17);
              goto LABEL_18;
            }
LABEL_137:
            v99 = v238;
            if ( !v238 )
              v99 = sub_73A880(v235, (__int64)v241);
            v238 = v99;
            if ( v203 )
            {
              v100 = v239;
              if ( !v239 )
              {
                v100 = sub_73A880(v236, v244.m128i_i64[0]);
                v99 = v238;
              }
              v239 = v100;
            }
            v98 = v237;
            goto LABEL_144;
          }
          if ( v240 )
          {
            a2 = v234;
            v227 = v94;
            sub_72A510(v240, v234);
            v94 = v227;
          }
          if ( v241 )
          {
            a2 = v235;
            v228 = v94;
            sub_72A510(v241, v235);
            v94 = v228;
          }
          if ( v244.m128i_i64[0] )
          {
            a2 = v236;
            v229 = v94;
            sub_72A510((const __m128i *)v244.m128i_i64[0], v236);
            v94 = v229;
          }
          v156 = v234;
          if ( v234[10].m128i_i8[13] == 12 && v94 != 5 )
            goto LABEL_340;
          if ( v218 )
          {
            if ( v235[10].m128i_i8[13] == 12 )
            {
              v155 = v240;
              if ( v237 )
                goto LABEL_137;
              goto LABEL_288;
            }
            if ( v203 )
            {
              if ( v236[10].m128i_i8[13] != 12 )
              {
                if ( (unsigned int)sub_711520((__int64)v234, (__int64)a2, (__int64)v235, v91, v92) )
                {
                  v164 = v244.m128i_i64[0];
                  *a10 = v244.m128i_i64[0];
                  if ( !v164 )
                  {
                    v165 = v236;
                    v166 = a9;
                    for ( i = 52; i; --i )
                    {
                      v166->m128i_i32[0] = v165->m128i_i32[0];
                      v165 = (__m128i *)((char *)v165 + 4);
                      v166 = (__m128i *)((char *)v166 + 4);
                    }
                  }
                }
                else
                {
                  v180 = v241;
                  *a10 = v241;
                  if ( !v180 )
                  {
                    v181 = v235;
                    v182 = a9;
                    for ( j = 52; j; --j )
                    {
                      v182->m128i_i32[0] = v181->m128i_i32[0];
                      v181 = (__m128i *)((char *)v181 + 4);
                      v182 = (__m128i *)((char *)v182 + 4);
                    }
                  }
                }
                goto LABEL_18;
              }
              goto LABEL_340;
            }
            sub_713ED0(v201, v234, v235, v209, (__int64)a9, 1u, 1, &v232, &v233, &v231, a5);
            v184 = v231 == 0;
            *a10 = 0;
            if ( v184 )
              goto LABEL_18;
LABEL_354:
            *a7 = 1;
            goto LABEL_18;
          }
          if ( v203 && v236[10].m128i_i8[13] == 12 )
          {
LABEL_340:
            v98 = v237;
            if ( v237 )
              goto LABEL_136;
            v155 = v240;
            goto LABEL_287;
          }
          if ( v94 == 5 )
          {
            v230 = (a1[1].m128i_i8[11] & 2) != 0;
            if ( !(unsigned int)sub_728A90((__int64)v234, v209, (a1[1].m128i_i8[11] & 2) == 0, (a6 & 0x80) != 0, &v233) )
              goto LABEL_354;
            v233 = (a1[3].m128i_i8[10] & 2) != 0;
            sub_72A510(v234, a9);
            sub_7115B0(a9, v209, v230, 1, 1, 1, 0, 0, 1u, v233, 0, &v232, &v231, a5);
            v184 = v231 == 0;
            *a10 = 0;
            if ( !v184 )
              goto LABEL_354;
          }
          else if ( v94 == 25 )
          {
            sub_72A510(v234, a9);
            *a10 = 0;
          }
          else
          {
            sub_712770(v201, v234, v209, (__int64)a9, 1, 1, &v232, &v233, &v231, a5);
            v184 = v231 == 0;
            *a10 = 0;
            if ( !v184 )
              goto LABEL_354;
          }
LABEL_18:
          v16 = a7;
          goto LABEL_11;
        }
      }
      else
      {
        v190 = v80;
        v90 = sub_7306C0(v89);
        v94 = v190;
        if ( v90 )
        {
          if ( !v190 )
            goto LABEL_129;
          v92 = 0;
          v95 = 0;
          goto LABEL_123;
        }
        if ( !v203 || (v95 = 0, v122 = sub_7306C0(0), v92 = 0, v94 = v190, !v122) )
        {
          if ( (unsigned __int8)v94 > 0x1Bu )
          {
            if ( v94 != 28 )
              goto LABEL_269;
          }
          else
          {
            if ( (unsigned __int8)v94 <= 0x19u )
              goto LABEL_129;
            v144 = *(_BYTE *)(v89 + 140);
            if ( v144 == 12 )
            {
              v145 = v89;
              do
              {
                v145 = *(_QWORD *)(v145 + 160);
                v144 = *(_BYTE *)(v145 + 140);
              }
              while ( v144 == 12 );
            }
            if ( v144 != 2 )
            {
LABEL_129:
              if ( *a7 )
                goto LABEL_7;
              if ( v94 != 5 )
              {
                if ( v94 != 25 )
                  goto LABEL_132;
                v97 = v237;
                if ( v237 )
                {
                  v209 = *v237;
                  goto LABEL_135;
                }
                v151 = v234;
                if ( !v234 )
                  v151 = v240;
                v211 = v151[8].m128i_i64[0];
                v152 = sub_8D32E0(v211);
                v94 = 25;
                if ( v152 )
                {
                  v153 = sub_8D46C0(v211);
                  v94 = 25;
                  v211 = v153;
                }
              }
              v209 = v211;
              if ( !v237 )
                goto LABEL_286;
              v97 = v237;
LABEL_135:
              v98 = v97;
              goto LABEL_136;
            }
          }
          v226 = v94;
          a2 = v234;
          v209 = sub_8D6540(v89);
          sub_73E000((__int64 *)&v237, v234, (const __m128i **)&v240, v209, a5);
          v94 = v226;
          goto LABEL_129;
        }
      }
      if ( !v94 )
        goto LABEL_125;
LABEL_123:
      if ( v94 == 5 )
      {
        v185 = v92;
        v96 = sub_8D2EF0(v89);
        v94 = 5;
        v92 = v185;
        if ( v96 || (v177 = sub_8D3D10(v89), v94 = 5, v92 = v185, v177) )
        {
LABEL_125:
          if ( v218 )
            goto LABEL_126;
LABEL_269:
          if ( v94 != 29 )
            goto LABEL_129;
          v146 = sub_72C390();
          v94 = 29;
          v209 = v146;
          goto LABEL_271;
        }
      }
LABEL_5:
      v16 = a7;
LABEL_6:
      *v16 = 1;
LABEL_7:
      v17 = (unsigned __int64)sub_7305B0();
      *a10 = 0;
LABEL_8:
      sub_724E30((__int64)&v234);
      sub_724E30((__int64)&v235);
      sub_724E30((__int64)&v236);
      return v17;
    case 2:
      v48 = a1[3].m128i_i64[1];
      goto LABEL_52;
    case 3:
    case 0x14:
      if ( a4 )
      {
        if ( !(unsigned int)sub_8D32B0(a4) || !(unsigned int)sub_717510(a1, a9, 1, v20, v21, v22) )
          goto LABEL_16;
        v73 = (_QWORD *)a1->m128i_i64[0];
        if ( (unsigned int)sub_8D32E0(a4) )
        {
          v74 = (__int64)a4;
          if ( (a1[1].m128i_i8[9] & 1) != 0 )
          {
            a9[8].m128i_i64[0] = sub_72D600(v73);
            v73 = (_QWORD *)sub_72D2E0(v73);
            v154 = (_QWORD *)sub_8D46C0(a4);
            v74 = sub_72D2E0(v154);
          }
        }
        else
        {
          v74 = (__int64)a4;
          v73 = (_QWORD *)sub_6EEB30((__int64)v73, 0);
          a9[8].m128i_i64[0] = (__int64)v73;
        }
        if ( !(unsigned int)sub_8E1010(
                              (_DWORD)v73,
                              1,
                              0,
                              a1[1].m128i_i8[8] == 20,
                              0,
                              (_DWORD)a9,
                              v74,
                              0,
                              0,
                              1,
                              0,
                              (__int64)&v244,
                              0)
          || !(unsigned int)sub_8DD690(&v244, v73, 1, a9, v74, 0) )
        {
LABEL_16:
          v23 = sub_73B8B0(a1, 0);
          v17 = (unsigned __int64)v23;
          if ( (*((_BYTE *)v23 + 25) & 1) == 0 )
            goto LABEL_18;
          if ( *((_BYTE *)v23 + 24) != 3 )
            goto LABEL_18;
          if ( (unsigned int)sub_8D32E0(a4) )
            goto LABEL_18;
          v107 = (const __m128i *)sub_6EA7C0(*(_QWORD *)(v17 + 56));
          if ( !v107 )
            goto LABEL_18;
          sub_72A510(v107, a9);
          a9[9].m128i_i64[0] = 0;
        }
        v17 = 0;
        *a10 = 0;
        v19 = *a7;
        goto LABEL_12;
      }
      v17 = (unsigned __int64)sub_73B8B0(a1, 0);
      v19 = *a7;
      goto LABEL_12;
    case 5:
      v14 = a1[3].m128i_i64[1];
      v15 = *(_BYTE *)(v14 + 48);
      if ( v15 != 1 )
      {
        if ( v15 != 2 )
          goto LABEL_5;
        v48 = *(_QWORD *)(v14 + 56);
LABEL_52:
        v17 = 0;
        *a10 = sub_743600(v48, (_DWORD)a2, a3, (_DWORD)a4, (_DWORD)a5, a6, (__int64)a7, (__int64)a8, (__int64)a9);
        v19 = *a7;
        goto LABEL_12;
      }
      v101 = sub_8A2270(a1->m128i_i64[0], (_DWORD)a2, a3, (_DWORD)a5, a6, (_DWORD)a7, (__int64)a8);
      if ( !*a7 && !sub_7306C0(v101) && !(unsigned int)sub_8D3D40(v101) )
      {
        v17 = 0;
        sub_72BB40(v101, a9);
        v19 = *a7;
        goto LABEL_12;
      }
      goto LABEL_5;
    case 0x16:
      v36 = sub_8A2270(a1[3].m128i_i64[1], (_DWORD)a2, a3, (_DWORD)a5, a6, (_DWORD)a7, (__int64)a8);
      goto LABEL_36;
    case 0x17:
      v39 = sub_7433F0(a1[4].m128i_i64[0], (_DWORD)a2, a3, (_DWORD)a5, a6, (_DWORD)a7, (__int64)a8);
      if ( *a7 )
        goto LABEL_7;
      v40 = sub_730FF0(a1);
      v40[8] = v39;
      v17 = (unsigned __int64)v40;
      sub_7197C0((__int64)v40, a9, 0, 0, &v244);
      v41 = a9[10].m128i_i8[13];
      v16 = a7;
      if ( !v41 )
        goto LABEL_6;
      v19 = *a7;
      if ( v41 != 12 )
        v17 = 0;
      goto LABEL_12;
    case 0x18:
      v45 = sub_8A2270(a1->m128i_i64[0], (_DWORD)a2, a3, (_DWORD)a5, a6, (_DWORD)a7, (__int64)a8);
      if ( *a7 )
        goto LABEL_7;
      v46 = sub_730FF0(a1);
      *v46 = v45;
      v17 = (unsigned __int64)v46;
      v19 = *a7;
      goto LABEL_12;
    case 0x19:
      v36 = sub_7433F0(a1[3].m128i_i64[1], (_DWORD)a2, a3, (_DWORD)a5, a6, (_DWORD)a7, (__int64)a8);
LABEL_36:
      v37 = v36;
      if ( *a7 )
        goto LABEL_7;
      v38 = sub_730FF0(a1);
      v38[7] = v37;
      v17 = (unsigned __int64)v38;
      v19 = *a7;
      goto LABEL_12;
    case 0x20:
      v42 = sub_690FF0(0, a1[4].m128i_i64[0], 0, (int)a2, a3, (__int64)a5, a6, (__int64)a7, (__int64)a8);
      v43 = (__int64 *)v42;
      if ( (a6 & 0x4000) == 0 && v42 && !(unsigned int)sub_89A370(v42) )
      {
        v244 = 0u;
        v44 = sub_6F1C10((__int64)a1, (__int64)a2, a3, &v244, 0, a8, a7, (unsigned int *)a7);
        sub_72C470(v44, (__int64)a9);
        v17 = 0;
        sub_67E3D0(&v244);
        sub_725130(v43);
        v19 = *a7;
        goto LABEL_12;
      }
      if ( *a7 )
        goto LABEL_7;
      v49 = sub_730FF0(a1);
      v49[8] = v43;
      v17 = (unsigned __int64)v49;
      v19 = *a7;
      goto LABEL_12;
    case 0x21:
      if ( (a6 & 0x4000) != 0 || (unsigned int)sub_89A370(a2) )
      {
        if ( (a6 & 0x86000) == 0 )
          goto LABEL_5;
        v24 = *((_DWORD *)qword_4D03BF8 + 2);
        v25 = *qword_4D03BF8;
        v243 = 0;
        v26 = v24 & ((unsigned __int64)a1 >> 3);
        v210 = v25;
        v217 = v24;
        v242 = 0;
        v241 = (__m128i *)sub_823970(0);
        v27 = (__m128i **)(v210 + 32LL * v26);
        v28 = *v27;
        if ( a1 == *v27 )
        {
LABEL_170:
          v105 = (__int64)v27[3];
          if ( v105 != v243 )
          {
            if ( v105 > 0 )
            {
              v109 = v27[3];
              v212 = v27[1];
              v243 = 0;
              sub_738450((const __m128i **)&v241, v109);
              v110 = v241;
              v111 = v212;
              v112 = (__m128i *)((char *)v241 + 24 * v105);
              do
              {
                if ( v110 )
                {
                  *v110 = _mm_loadu_si128(v111);
                  v110[1].m128i_i64[0] = v111[1].m128i_i64[0];
                }
                v110 = (__m128i *)((char *)v110 + 24);
                v111 = (const __m128i *)((char *)v111 + 24);
              }
              while ( v110 != v112 );
            }
            v243 = v105;
          }
        }
        else
        {
          while ( v28 )
          {
            v26 = v217 & (v26 + 1);
            v27 = (__m128i **)(v210 + 32LL * v26);
            v28 = *v27;
            if ( a1 == *v27 )
              goto LABEL_170;
          }
        }
        v17 = (unsigned __int64)sub_730FF0(a1);
        v29 = sub_72F240(v12);
        v30 = v243;
        v31 = v29;
        if ( v243 == v242 )
          sub_738390((const __m128i **)&v241);
        v32 = v241;
        if ( v30 > 0 )
        {
          v33 = (__m128i *)((char *)v241 + 24 * v30);
          do
          {
            v34 = v33;
            v33 = (__m128i *)((char *)v33 - 24);
            if ( v34 )
            {
              v35 = _mm_loadu_si128((__m128i *)((char *)v34 - 24));
              v34[1].m128i_i64[0] = v34[-1].m128i_i64[1];
              *v34 = v35;
            }
          }
          while ( v33 != v32 );
        }
        if ( v32 )
        {
          v32->m128i_i64[1] = (__int64)v31;
          v32->m128i_i64[0] = a3;
          v32[1].m128i_i8[0] = v32[1].m128i_i8[0] & 0xF0 | (8 * ((a6 & 0x80000) != 0)) | (4 * ((a6 & 0x2000) != 0) + 1);
        }
        v243 = v30 + 1;
        sub_738500(v244.m128i_i64, (__int64)qword_4D03BF8, v17, (const __m128i **)&v241, v17 >> 3);
        sub_823A00(v244.m128i_i64[0], 24 * v244.m128i_i64[1]);
        sub_823A00(v241, 24 * v242);
        v19 = *a7;
LABEL_12:
        if ( !v19 )
          goto LABEL_8;
        goto LABEL_7;
      }
      if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) & 0x30) == 0x10 )
      {
        sub_89F7D0(&v241);
      }
      else
      {
        v243 = 0;
        v242 = 1;
        v241 = (__m128i *)sub_823970(24);
      }
      v50 = *((_DWORD *)qword_4D03BF8 + 2);
      v51 = *qword_4D03BF8;
      v244 = 0u;
      v52 = v50 & ((unsigned __int64)a1 >> 3);
      v245 = 0;
      v53 = sub_823970(0);
      v244 = (__m128i)(unsigned __int64)v53;
      v56 = v53;
      v57 = (__m128i **)(v51 + 32LL * v52);
      v58 = *v57;
      if ( a1 == *v57 )
      {
LABEL_188:
        v108 = v57[1];
        v59 = (__int64)v57[3];
        if ( v59 != v245 )
        {
          if ( v59 > 0 )
          {
            v118 = v57[3];
            v245 = 0;
            sub_738450((const __m128i **)&v244, v118);
            v119 = (__m128i *)v244.m128i_i64[0];
            v56 = 3 * v59;
            v120 = v108;
            v121 = v244.m128i_i64[0] + 24 * v59;
            do
            {
              if ( v119 )
              {
                *v119 = _mm_loadu_si128(v120);
                v56 = v120[1].m128i_i64[0];
                v119[1].m128i_i64[0] = v56;
              }
              v119 = (__m128i *)((char *)v119 + 24);
              v120 = (const __m128i *)((char *)v120 + 24);
            }
            while ( (__m128i *)v121 != v119 );
          }
          v245 = v59;
          goto LABEL_64;
        }
        if ( v59 <= 0 )
        {
LABEL_64:
          v60 = v243;
          if ( v59 )
          {
            if ( v243 == v242 )
              sub_738390((const __m128i **)&v241);
            v61 = v241;
            if ( v60 > 0 )
            {
              v62 = (__m128i *)((char *)v241 + 24 * v60);
              do
              {
                v63 = v62;
                v62 = (__m128i *)((char *)v62 - 24);
                if ( v63 )
                {
                  v64 = _mm_loadu_si128((__m128i *)((char *)v63 - 24));
                  v63[1].m128i_i64[0] = v63[-1].m128i_i64[1];
                  *v63 = v64;
                }
              }
              while ( v61 != v62 );
            }
            if ( v61 )
            {
              v61->m128i_i64[1] = (__int64)v12;
              v61->m128i_i64[0] = a3;
              v61[1].m128i_i8[0] = v61[1].m128i_i8[0] & 0xF0 | 2;
            }
            v65 = v245;
            v66 = v60 + 1;
            v67 = (const __m128i *)v244.m128i_i64[0];
            v243 = v60 + 1;
            v56 = v60 + 1 + v245;
            v68 = v241;
            if ( v56 > v242 )
            {
              v213 = v242;
              v225 = v60 + 1 + v245;
              v139 = (__m128i *)sub_823970(24 * v56);
              v140 = v213;
              v141 = v139;
              if ( v60 + 1 > 0 )
              {
                v142 = v68;
                v143 = (__m128i *)((char *)v139 + 24 * v60 + 24);
                do
                {
                  if ( v139 )
                  {
                    *v139 = _mm_loadu_si128(v142);
                    v139[1].m128i_i64[0] = v142[1].m128i_i64[0];
                  }
                  v139 = (__m128i *)((char *)v139 + 24);
                  v142 = (const __m128i *)((char *)v142 + 24);
                }
                while ( v143 != v139 );
              }
              v214 = v141;
              sub_823A00(v68, 24 * v140);
              v55 = (__int64)v214;
              v56 = v225;
              v66 = v60 + 1;
              v241 = (__m128i *)v214;
              v68 = v214;
              v242 = v225;
            }
            if ( v66 > 1 )
            {
              v69 = (__m128i *)((char *)v68 + 24 * v65 + 24 * v60);
              v70 = (const __m128i *)((char *)v68 + 24 * v60);
              do
              {
                if ( v69 )
                {
                  *v69 = _mm_loadu_si128(v70);
                  v56 = v70[1].m128i_i64[0];
                  v69[1].m128i_i64[0] = v56;
                }
                v70 = (const __m128i *)((char *)v70 - 24);
                v69 = (__m128i *)((char *)v69 - 24);
              }
              while ( v68 != v70 );
            }
            if ( v65 > 0 )
            {
              v71 = (__m128i *)&v68[1].m128i_u64[1];
              v56 = (__int64)&v68[1].m128i_i64[3 * v65 + 1];
              do
              {
                if ( v71 )
                {
                  *v71 = _mm_loadu_si128(v67);
                  v71[1].m128i_i64[0] = v67[1].m128i_i64[0];
                }
                v71 = (__m128i *)((char *)v71 + 24);
                v67 = (const __m128i *)((char *)v67 + 24);
              }
              while ( (__m128i *)v56 != v71 );
            }
            v243 += v65;
          }
          else
          {
            if ( v243 == v242 )
              sub_738390((const __m128i **)&v241);
            v113 = &v241->m128i_i8[24 * v60];
            if ( v113 )
            {
              v114 = v113[16];
              *((_QWORD *)v113 + 1) = v12;
              *(_QWORD *)v113 = a3;
              v113[16] = v114 & 0xF0 | 2;
            }
            v243 = v60 + 1;
          }
          v17 = 0;
          v72 = sub_6F1E90((__int64)a1, (__int64)&v241, (__int64)a8, v56, v54, v55);
          sub_72C470(v72, (__int64)a9);
          sub_823A00(v244.m128i_i64[0], 24 * v244.m128i_i64[1]);
          sub_823A00(v241, 24 * v242);
          v19 = *a7;
          goto LABEL_12;
        }
        v138 = 0;
        do
        {
          *(__m128i *)(v56 + v138 * 8) = _mm_loadu_si128((const __m128i *)((char *)v108 + v138 * 8));
          *(_QWORD *)(v56 + v138 * 8 + 16) = v108[1].m128i_i64[v138];
          v138 += 3;
        }
        while ( 3 * v59 != v138 );
      }
      else
      {
        while ( v58 )
        {
          v52 = v50 & (v52 + 1);
          v57 = (__m128i **)(v51 + 32LL * v52);
          v58 = *v57;
          if ( a1 == *v57 )
            goto LABEL_188;
        }
      }
      v59 = v245;
      goto LABEL_64;
    default:
      goto LABEL_5;
  }
}
