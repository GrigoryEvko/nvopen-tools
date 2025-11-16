// Function: sub_1282050
// Address: 0x1282050
//
__int64 __fastcall sub_1282050(
        __int64 *a1,
        _DWORD *a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        int a8,
        int a9,
        __int64 a10,
        _BYTE *a11,
        __int64 a12,
        __int64 a13,
        __int64 a14,
        __int64 a15)
{
  int v15; // r13d
  __int64 v17; // r14
  __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // rax
  unsigned int v21; // r13d
  _QWORD *v22; // r14
  unsigned __int64 v23; // rax
  __int64 v24; // r13
  __int64 v25; // rax
  __int64 v26; // r15
  __int64 v27; // r14
  __int64 v28; // r8
  __int64 v29; // r9
  unsigned __int64 v30; // rdi
  int v31; // eax
  int v32; // edx
  int v33; // eax
  __int64 v34; // rax
  bool v35; // cf
  unsigned __int64 v36; // rax
  __int64 v37; // rdi
  _BOOL8 v38; // r13
  unsigned int v39; // r14d
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 result; // rax
  __int64 v43; // rax
  unsigned __int64 v44; // r14
  __int64 v45; // rax
  __int64 v46; // rdi
  __int64 v47; // r13
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // r13
  __int64 v51; // rdi
  __int64 v52; // rsi
  __int64 v53; // rdi
  __int64 v54; // rax
  _QWORD *v55; // r15
  __int64 v56; // rdi
  unsigned __int64 v57; // rsi
  __int64 v58; // rax
  __int64 v59; // rsi
  _QWORD *v60; // rdx
  char *v61; // rsi
  __int64 v62; // rdi
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // r15
  __int64 v66; // rax
  _QWORD *v67; // r13
  __int64 v68; // rdi
  unsigned __int64 *v69; // r15
  __int64 v70; // rax
  unsigned __int64 v71; // rcx
  __int64 v72; // rsi
  char *v73; // rsi
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // rdi
  __int64 v77; // r13
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // r12
  unsigned __int64 v82; // r15
  __int64 v83; // rax
  __int64 v84; // rax
  _QWORD *v85; // r15
  __int64 v86; // rdi
  unsigned __int64 *v87; // r13
  __int64 v88; // rax
  unsigned __int64 v89; // rcx
  __int64 v90; // rsi
  char *v91; // rsi
  __int64 v92; // rax
  __int64 v93; // rdi
  __int64 *v94; // r15
  __int64 v95; // rax
  __int64 v96; // rsi
  __int64 v97; // rsi
  unsigned __int64 v98; // rsi
  __int64 v99; // rdi
  __int64 v100; // rax
  _QWORD *v101; // r15
  __int64 v102; // rdi
  unsigned __int64 v103; // rsi
  __int64 v104; // rax
  __int64 v105; // rsi
  _QWORD *v106; // rdx
  char *v107; // rsi
  __int64 v108; // rdx
  __int64 v109; // rax
  __int64 v110; // rdx
  __int64 v111; // r15
  __int64 v112; // r13
  __int64 v113; // rax
  _QWORD *v114; // r13
  __int64 v115; // rdi
  unsigned __int64 *v116; // r15
  __int64 v117; // rax
  unsigned __int64 v118; // rcx
  __int64 v119; // rsi
  char *v120; // rsi
  __int64 v121; // rax
  __int64 v122; // rdi
  __int64 *v123; // r13
  __int64 v124; // rax
  __int64 v125; // rcx
  __int64 v126; // rsi
  unsigned __int64 v127; // rsi
  __int64 v128; // rax
  __int64 v129; // rdi
  __int64 *v130; // r15
  __int64 v131; // rax
  __int64 v132; // rsi
  __int64 v133; // rsi
  unsigned __int64 v134; // rsi
  __int64 v135; // rax
  __int64 v136; // rdi
  unsigned __int64 v137; // rsi
  __int64 v138; // rax
  __int64 v139; // rsi
  _QWORD *v140; // rdx
  unsigned __int64 v141; // rsi
  _QWORD *v142; // r14
  __int64 i; // rax
  unsigned __int64 v144; // rcx
  int v145; // r10d
  __int64 v146; // r13
  unsigned int v147; // eax
  unsigned int v148; // r8d
  unsigned __int64 v149; // rdx
  unsigned __int64 v150; // rcx
  __int64 v151; // rsi
  unsigned __int64 v152; // rdx
  __int64 v153; // rdi
  __int64 v154; // r15
  __int64 v155; // r14
  __int64 v156; // r13
  unsigned __int64 v157; // r15
  int v158; // ecx
  __int64 v159; // rax
  _QWORD *v160; // r13
  __int64 v161; // rdi
  unsigned __int64 *v162; // r14
  __int64 v163; // rax
  unsigned __int64 v164; // rcx
  __int64 v165; // rsi
  _QWORD *v166; // r12
  char *v167; // rsi
  __int64 v168; // rax
  unsigned __int64 *v169; // r14
  __int64 v170; // rax
  unsigned __int64 v171; // rcx
  __int64 v172; // rsi
  _BYTE *v173; // r14
  unsigned __int64 v174; // rsi
  __int64 v175; // rax
  __int64 *v176; // r13
  __int64 v177; // rax
  __int64 v178; // rcx
  __int64 v179; // rsi
  unsigned __int64 v180; // rsi
  __int64 v181; // rsi
  int v182; // r13d
  __int64 v183; // rdi
  __int64 v184; // rax
  __int64 *v185; // r13
  __int64 v186; // rax
  __int64 v187; // rcx
  __int64 v188; // rsi
  unsigned __int64 v189; // rsi
  __int64 v190; // rax
  __int64 v191; // rax
  __int64 v192; // rdi
  __int64 v193; // r13
  __int64 v194; // rax
  __int64 v195; // rax
  __int64 v196; // rdi
  __int64 v197; // rax
  __int64 v198; // rdi
  __int64 v199; // r13
  __int64 v200; // rax
  _QWORD *v201; // r14
  __int64 v202; // rax
  unsigned __int64 *v203; // r13
  __int64 v204; // rax
  unsigned __int64 v205; // rsi
  __int64 v206; // rsi
  char *v207; // rsi
  __int64 v208; // rsi
  __int64 v209; // rdx
  unsigned __int64 v210; // rax
  int v211; // edx
  __int64 v212; // rdi
  __int64 v213; // rax
  __int64 v214; // r13
  unsigned __int8 v215; // al
  __int64 v216; // rax
  _QWORD *v217; // r14
  __int64 v218; // rax
  unsigned __int64 *v219; // r13
  __int64 v220; // rax
  unsigned __int64 v221; // rcx
  __int64 v222; // rsi
  char *v223; // rsi
  __int64 v224; // rax
  __int64 *v225; // r13
  __int64 v226; // rax
  __int64 v227; // rcx
  __int64 v228; // rsi
  unsigned __int64 v229; // rsi
  bool v230; // al
  __int64 v231; // r13
  _QWORD *v232; // r14
  __int64 v233; // rax
  __int64 v234; // rax
  unsigned __int64 *v235; // r15
  __int64 v236; // rax
  unsigned __int64 v237; // rcx
  __int64 v238; // rsi
  unsigned __int64 v239; // rsi
  __int64 v240; // rax
  _QWORD *v241; // r14
  __int64 v242; // rax
  unsigned __int64 *v243; // r13
  __int64 v244; // rax
  unsigned __int64 v245; // rcx
  __int64 v246; // rsi
  __int64 v247; // rax
  __int64 *v248; // r14
  __int64 v249; // rax
  __int64 v250; // rcx
  __int64 v251; // rsi
  unsigned __int64 v252; // rsi
  __int64 v253; // rax
  __int64 *v254; // r14
  __int64 v255; // rax
  __int64 v256; // rcx
  __int64 v257; // rsi
  __int64 v258; // r14
  unsigned __int64 v259; // rsi
  __int64 v260; // [rsp+0h] [rbp-100h]
  __int64 v261; // [rsp+8h] [rbp-F8h]
  unsigned __int64 *v262; // [rsp+8h] [rbp-F8h]
  __int64 v263; // [rsp+8h] [rbp-F8h]
  __int64 v264; // [rsp+8h] [rbp-F8h]
  unsigned __int64 *v265; // [rsp+8h] [rbp-F8h]
  __int64 v266; // [rsp+8h] [rbp-F8h]
  unsigned __int64 v267; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v268; // [rsp+18h] [rbp-E8h]
  _BYTE *v269; // [rsp+20h] [rbp-E0h]
  unsigned __int8 v270; // [rsp+28h] [rbp-D8h]
  __int64 v271; // [rsp+28h] [rbp-D8h]
  __int64 v272; // [rsp+30h] [rbp-D0h]
  __int64 v273; // [rsp+30h] [rbp-D0h]
  unsigned __int64 *v274; // [rsp+30h] [rbp-D0h]
  unsigned __int64 v276; // [rsp+38h] [rbp-C8h]
  __int64 v277; // [rsp+40h] [rbp-C0h]
  unsigned int v278; // [rsp+40h] [rbp-C0h]
  __int64 *v279; // [rsp+48h] [rbp-B8h]
  unsigned int v280; // [rsp+50h] [rbp-B0h]
  unsigned __int64 v281; // [rsp+50h] [rbp-B0h]
  char v282; // [rsp+50h] [rbp-B0h]
  __int64 v284; // [rsp+58h] [rbp-A8h]
  unsigned int v285; // [rsp+58h] [rbp-A8h]
  __int64 v286; // [rsp+58h] [rbp-A8h]
  unsigned __int64 v287; // [rsp+68h] [rbp-98h] BYREF
  unsigned __int64 v288; // [rsp+70h] [rbp-90h] BYREF
  unsigned int v289; // [rsp+78h] [rbp-88h]
  __int16 v290; // [rsp+80h] [rbp-80h]
  char *v291; // [rsp+90h] [rbp-70h] BYREF
  unsigned int v292; // [rsp+98h] [rbp-68h]
  __int16 v293; // [rsp+A0h] [rbp-60h]
  char *v294; // [rsp+B0h] [rbp-50h] BYREF
  unsigned int v295; // [rsp+B8h] [rbp-48h]
  __int16 v296; // [rsp+C0h] [rbp-40h]

  v15 = 1;
  v269 = a11;
  v280 = *(unsigned __int8 *)(a13 + 137);
  v17 = sub_127A040(a1[4] + 8, *(_QWORD *)(a13 + 120));
  v18 = v17;
  v19 = *(_QWORD *)(a1[4] + 368);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v18 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v83 = *(_QWORD *)(v18 + 32);
        v18 = *(_QWORD *)(v18 + 24);
        v15 *= (_DWORD)v83;
        continue;
      case 1:
        LODWORD(v20) = 16;
        break;
      case 2:
        LODWORD(v20) = 32;
        break;
      case 3:
      case 9:
        LODWORD(v20) = 64;
        break;
      case 4:
        LODWORD(v20) = 80;
        break;
      case 5:
      case 6:
        LODWORD(v20) = 128;
        break;
      case 7:
        LODWORD(v20) = 8 * sub_15A9520(v19, 0);
        break;
      case 0xB:
        LODWORD(v20) = *(_DWORD *)(v18 + 8) >> 8;
        break;
      case 0xD:
        v20 = 8LL * *(_QWORD *)sub_15A9930(v19, v18);
        break;
      case 0xE:
        v81 = *(_QWORD *)(v18 + 32);
        v273 = *(_QWORD *)(v18 + 24);
        v82 = (unsigned int)sub_15A9FE0(v19, v273);
        v20 = 8 * v82 * v81 * ((v82 + ((unsigned __int64)(sub_127FA20(v19, v273) + 7) >> 3) - 1) / v82);
        break;
      case 0xF:
        LODWORD(v20) = 8 * sub_15A9520(v19, *(_DWORD *)(v18 + 8) >> 8);
        break;
    }
    break;
  }
  v21 = v20 * v15;
  v279 = a1 + 6;
  v291 = "tmp";
  v293 = 259;
  if ( v17 == *(_QWORD *)a7 )
  {
    v22 = (_QWORD *)a7;
  }
  else if ( *(_BYTE *)(a7 + 16) > 0x10u )
  {
    v296 = 257;
    v135 = sub_15FE0A0(a7, v17, 0, &v294, 0);
    v136 = a1[7];
    v22 = (_QWORD *)v135;
    if ( v136 )
    {
      v274 = (unsigned __int64 *)a1[8];
      sub_157E9D0(v136 + 40, v135);
      v137 = *v274;
      v138 = v22[3] & 7LL;
      v22[4] = v274;
      v137 &= 0xFFFFFFFFFFFFFFF8LL;
      v22[3] = v137 | v138;
      *(_QWORD *)(v137 + 8) = v22 + 3;
      *v274 = *v274 & 7 | (unsigned __int64)(v22 + 3);
    }
    sub_164B780(v22, &v291);
    v139 = a1[6];
    if ( v139 )
    {
      v288 = a1[6];
      sub_1623A60(&v288, v139, 2);
      v140 = v22 + 6;
      if ( v22[6] )
      {
        sub_161E7C0(v22 + 6);
        v140 = v22 + 6;
      }
      v141 = v288;
      v22[6] = v288;
      if ( v141 )
        sub_1623210(&v288, v141, v140);
    }
  }
  else
  {
    v22 = (_QWORD *)sub_15A4750(a7, v17, 0);
  }
  v295 = v21;
  if ( v21 > 0x40 )
    sub_16A4EF0(&v294, 0, 0);
  else
    v294 = 0;
  if ( v280 )
  {
    if ( v280 > 0x40 )
    {
      sub_16A5260(&v294, 0, v280);
    }
    else
    {
      v23 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v280);
      if ( v295 > 0x40 )
        *(_QWORD *)v294 |= v23;
      else
        v294 = (char *)((unsigned __int64)v294 | v23);
    }
  }
  v24 = sub_159C0E0(a1[5], &v294);
  if ( v295 > 0x40 && v294 )
    j_j___libc_free_0_0(v294);
  v294 = "bf.value";
  v296 = 259;
  v25 = sub_1281C00(v279, (__int64)v22, v24, (__int64)&v294);
  v272 = v25;
  if ( a3 )
  {
    v26 = *(_QWORD *)a7;
    v291 = "bf.reload.val";
    v293 = 259;
    if ( v26 == *(_QWORD *)v25 )
    {
      v27 = v25;
    }
    else if ( *(_BYTE *)(v25 + 16) > 0x10u )
    {
      v296 = 257;
      v27 = sub_15FE0A0(v25, v26, 0, &v294, 0);
      v184 = a1[7];
      if ( v184 )
      {
        v185 = (__int64 *)a1[8];
        sub_157E9D0(v184 + 40, v27);
        v186 = *(_QWORD *)(v27 + 24);
        v187 = *v185;
        *(_QWORD *)(v27 + 32) = v185;
        v187 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v27 + 24) = v187 | v186 & 7;
        *(_QWORD *)(v187 + 8) = v27 + 24;
        *v185 = *v185 & 7 | (v27 + 24);
      }
      sub_164B780(v27, &v291);
      v188 = a1[6];
      if ( v188 )
      {
        v288 = a1[6];
        sub_1623A60(&v288, v188, 2);
        if ( *(_QWORD *)(v27 + 48) )
          sub_161E7C0(v27 + 48);
        v189 = v288;
        *(_QWORD *)(v27 + 48) = v288;
        if ( v189 )
          sub_1623210(&v288, v189, v27 + 48);
      }
    }
    else
    {
      v27 = sub_15A4750(v25, v26, 0);
    }
    if ( (*(_BYTE *)(a13 + 144) & 8) != 0 )
    {
      v181 = v26;
      v182 = 1;
      v183 = *(_QWORD *)(a1[4] + 368);
      while ( 2 )
      {
        switch ( *(_BYTE *)(v181 + 8) )
        {
          case 1:
            LODWORD(v181) = 16;
            goto LABEL_262;
          case 2:
            LODWORD(v181) = 32;
            goto LABEL_262;
          case 3:
          case 9:
            LODWORD(v181) = 64;
            goto LABEL_262;
          case 4:
            LODWORD(v181) = 80;
            goto LABEL_262;
          case 5:
          case 6:
            LODWORD(v181) = 128;
            goto LABEL_262;
          case 7:
            LODWORD(v181) = 8 * sub_15A9520(v183, 0);
            goto LABEL_262;
          case 0xB:
            LODWORD(v181) = *(_DWORD *)(v181 + 8) >> 8;
            goto LABEL_262;
          case 0xD:
            v181 = 8LL * *(_QWORD *)sub_15A9930(v183, v181);
            goto LABEL_262;
          case 0xE:
            v260 = *(_QWORD *)(v181 + 24);
            v271 = *(_QWORD *)(v181 + 32);
            v268 = (unsigned int)sub_15A9FE0(v183, v260);
            v181 = 8 * v268 * v271 * ((v268 + ((unsigned __int64)(sub_127FA20(v183, v260) + 7) >> 3) - 1) / v268);
            goto LABEL_262;
          case 0xF:
            LODWORD(v181) = 8 * sub_15A9520(v183, *(_DWORD *)(v181 + 8) >> 8);
LABEL_262:
            v231 = sub_15A0680(v26, v182 * (_DWORD)v181 - v280, 0);
            v293 = 259;
            v291 = "bf.reload.sext";
            v290 = 257;
            if ( *(_BYTE *)(v27 + 16) > 0x10u || *(_BYTE *)(v231 + 16) > 0x10u )
            {
              v296 = 257;
              v232 = (_QWORD *)sub_15FB440(23, v27, v231, &v294, 0);
              v234 = a1[7];
              if ( v234 )
              {
                v235 = (unsigned __int64 *)a1[8];
                sub_157E9D0(v234 + 40, v232);
                v236 = v232[3];
                v237 = *v235;
                v232[4] = v235;
                v237 &= 0xFFFFFFFFFFFFFFF8LL;
                v232[3] = v237 | v236 & 7;
                *(_QWORD *)(v237 + 8) = v232 + 3;
                *v235 = *v235 & 7 | (unsigned __int64)(v232 + 3);
              }
              sub_164B780(v232, &v288);
              v238 = a1[6];
              if ( v238 )
              {
                v287 = a1[6];
                sub_1623A60(&v287, v238, 2);
                if ( v232[6] )
                  sub_161E7C0(v232 + 6);
                v239 = v287;
                v232[6] = v287;
                if ( v239 )
                  sub_1623210(&v287, v239, v232 + 6);
              }
            }
            else
            {
              v232 = (_QWORD *)sub_15A2D50(v27, v231, 0, 0);
            }
            v27 = sub_1281D90(v279, (__int64)v232, v231, (__int64)&v291, 0);
            break;
          case 0x10:
            v233 = *(_QWORD *)(v181 + 32);
            v181 = *(_QWORD *)(v181 + 24);
            v182 *= (_DWORD)v233;
            continue;
          default:
            BUG();
        }
        break;
      }
    }
    *a3 = v27;
  }
  v270 = a15 & 1;
  if ( (unsigned __int8)sub_127F680(a14, a13, (unsigned int)a12) )
  {
    v142 = sub_1281800(a1, a2, &v287, a15 & 1, v28, v29, a10, a11, a12, a13, a14, a15);
    for ( i = *(_QWORD *)(a13 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v144 = *(_QWORD *)(i + 128);
    v145 = *(unsigned __int8 *)(a13 + 137);
    v146 = *(unsigned __int8 *)(a13 + 136) + 8 * (*(_QWORD *)(a13 + 128) % v144);
    v147 = 8 * v144;
    v292 = 8 * v144;
    v148 = v145 + v146;
    if ( (unsigned int)(8 * v144) <= 0x40 )
    {
      v291 = 0;
      if ( (_DWORD)v146 == v148 )
      {
        v151 = -1;
        goto LABEL_161;
      }
      if ( (unsigned int)v146 <= 0x3F && v148 <= 0x40 )
      {
        v149 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v145) << v146;
        v150 = 0;
LABEL_160:
        v151 = ~(v150 | v149);
        goto LABEL_161;
      }
      goto LABEL_260;
    }
    v278 = v145 + v146;
    v282 = v145;
    sub_16A4EF0(&v291, 0, 0);
    v148 = v278;
    if ( (_DWORD)v146 != v278 )
    {
      if ( (unsigned int)v146 > 0x3F || v278 > 0x40 )
      {
LABEL_260:
        sub_16A5260(&v291, (unsigned int)v146, v148);
        goto LABEL_212;
      }
      v147 = v292;
      v149 = 0xFFFFFFFFFFFFFFFFLL >> (64 - v282) << v146;
      v150 = (unsigned __int64)v291;
      if ( v292 <= 0x40 )
        goto LABEL_160;
      *(_QWORD *)v291 |= v149;
    }
LABEL_212:
    v147 = v292;
    if ( v292 > 0x40 )
    {
      sub_16A8F40(&v291);
      v147 = v292;
      v152 = (unsigned __int64)v291;
      goto LABEL_162;
    }
    v151 = ~(unsigned __int64)v291;
LABEL_161:
    v152 = v151 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v147);
    v291 = (char *)v152;
LABEL_162:
    v153 = a1[5];
    v295 = v147;
    v294 = (char *)v152;
    v292 = 0;
    v154 = sub_159C0E0(v153, &v294);
    if ( v295 > 0x40 && v294 )
      j_j___libc_free_0_0(v294);
    if ( v292 > 0x40 && v291 )
      j_j___libc_free_0_0(v291);
    v294 = "bf.prev.cleared";
    v296 = 259;
    v155 = sub_1281C00(v279, (__int64)v142, v154, (__int64)&v294);
    v294 = "bf.newval.positioned";
    v296 = 259;
    v156 = sub_1281EE0(v279, v272, v146, (__int64)&v294, 0, 0);
    v293 = 259;
    v291 = "bf.finalcontainerval";
    if ( *(_BYTE *)(v156 + 16) <= 0x10u )
    {
      if ( (unsigned __int8)sub_1593BB0(v156) )
        goto LABEL_172;
      if ( *(_BYTE *)(v155 + 16) <= 0x10u )
      {
        v155 = sub_15A2D10(v155, v156);
        goto LABEL_172;
      }
    }
    v296 = 257;
    v155 = sub_15FB440(27, v155, v156, &v294, 0);
    v224 = a1[7];
    if ( v224 )
    {
      v225 = (__int64 *)a1[8];
      sub_157E9D0(v224 + 40, v155);
      v226 = *(_QWORD *)(v155 + 24);
      v227 = *v225;
      *(_QWORD *)(v155 + 32) = v225;
      v227 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v155 + 24) = v227 | v226 & 7;
      *(_QWORD *)(v227 + 8) = v155 + 24;
      *v225 = *v225 & 7 | (v155 + 24);
    }
    sub_164B780(v155, &v291);
    v228 = a1[6];
    if ( v228 )
    {
      v288 = a1[6];
      sub_1623A60(&v288, v228, 2);
      if ( *(_QWORD *)(v155 + 48) )
        sub_161E7C0(v155 + 48);
      v229 = v288;
      *(_QWORD *)(v155 + 48) = v288;
      if ( v229 )
        sub_1623210(&v288, v229, v155 + 48);
    }
LABEL_172:
    v157 = v287;
    v158 = 1;
    if ( !v270 )
    {
      v158 = unk_4D0463C;
      if ( unk_4D0463C )
      {
        v230 = sub_126A420(a1[4], v287);
        v157 = v287;
        v158 = v230;
      }
    }
    v285 = v158;
    v296 = 257;
    v159 = sub_1648A60(64, 2);
    v160 = (_QWORD *)v159;
    if ( v159 )
      sub_15F9650(v159, v155, v157, v285, 0);
    v161 = a1[7];
    if ( v161 )
    {
      v162 = (unsigned __int64 *)a1[8];
      sub_157E9D0(v161 + 40, v160);
      v163 = v160[3];
      v164 = *v162;
      v160[4] = v162;
      v164 &= 0xFFFFFFFFFFFFFFF8LL;
      v160[3] = v164 | v163 & 7;
      *(_QWORD *)(v164 + 8) = v160 + 3;
      *v162 = *v162 & 7 | (unsigned __int64)(v160 + 3);
    }
    result = sub_164B780(v160, &v294);
    v165 = a1[6];
    if ( !v165 )
      return result;
    v291 = (char *)a1[6];
    v166 = v160 + 6;
    result = sub_1623A60(&v291, v165, 2);
    if ( v160[6] )
      result = sub_161E7C0(v160 + 6);
    v167 = v291;
    v160[6] = v291;
    if ( !v167 )
      return result;
    return sub_1623210(&v291, v167, v166);
  }
  if ( !v270 && unk_4D0463C )
    v270 = sub_126A420(a1[4], (unsigned __int64)a11);
  v30 = *(_QWORD *)(a13 + 128);
  v31 = *(unsigned __int8 *)(a13 + 137) + *(unsigned __int8 *)(a13 + 136);
  v32 = v31 + 6;
  v33 = v31 - 1;
  v267 = v30;
  if ( v33 < 0 )
    v33 = v32;
  v34 = v33 >> 3;
  v35 = __CFADD__(v30, v34);
  v36 = v30 + v34;
  v37 = a1[5];
  v281 = v36;
  v38 = v35;
  v39 = *(_DWORD *)(*(_QWORD *)a11 + 8LL);
  v277 = v35;
  v293 = 257;
  v40 = sub_1643330(v37);
  v41 = sub_1646BA0(v40, v39 >> 8);
  if ( v41 != *(_QWORD *)a11 )
  {
    if ( a11[16] > 0x10u )
    {
      v296 = 257;
      v269 = (_BYTE *)sub_15FDBD0(47, a11, v41, &v294, 0);
      v168 = a1[7];
      if ( v168 )
      {
        v169 = (unsigned __int64 *)a1[8];
        sub_157E9D0(v168 + 40, v269);
        v170 = *((_QWORD *)v269 + 3);
        v171 = *v169;
        *((_QWORD *)v269 + 4) = v169;
        v171 &= 0xFFFFFFFFFFFFFFF8LL;
        *((_QWORD *)v269 + 3) = v171 | v170 & 7;
        *(_QWORD *)(v171 + 8) = v269 + 24;
        *v169 = *v169 & 7 | (unsigned __int64)(v269 + 24);
      }
      sub_164B780(v269, &v291);
      v172 = a1[6];
      if ( v172 )
      {
        v288 = a1[6];
        sub_1623A60(&v288, v172, 2);
        v173 = v269 + 48;
        if ( *((_QWORD *)v269 + 6) )
          sub_161E7C0(v173);
        v174 = v288;
        *((_QWORD *)v269 + 6) = v288;
        if ( v174 )
          sub_1623210(&v288, v174, v173);
      }
    }
    else
    {
      v269 = (_BYTE *)sub_15A46C0(47, a11, v41, 0);
    }
  }
  if ( v267 == v281 )
  {
    v190 = sub_1643350(a1[5]);
    v191 = sub_159C470(v190, v267, 0);
    v192 = a1[5];
    v193 = v191;
    v296 = 259;
    v294 = "bf_byte";
    v194 = sub_1643330(v192);
    v195 = sub_12815B0(v279, v194, v269, v193, (__int64)&v294);
    v196 = a1[5];
    v286 = v195;
    v291 = "bfval.trunc";
    v293 = 259;
    v197 = sub_1644900(v196, 8);
    if ( v197 != *(_QWORD *)v272 )
    {
      if ( *(_BYTE *)(v272 + 16) > 0x10u )
      {
        v296 = 257;
        v272 = sub_15FE0A0(v272, v197, 0, &v294, 0);
        v253 = a1[7];
        if ( v253 )
        {
          v254 = (__int64 *)a1[8];
          sub_157E9D0(v253 + 40, v272);
          v255 = *(_QWORD *)(v272 + 24);
          v256 = *v254;
          *(_QWORD *)(v272 + 32) = v254;
          v256 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v272 + 24) = v256 | v255 & 7;
          *(_QWORD *)(v256 + 8) = v272 + 24;
          *v254 = *v254 & 7 | (v272 + 24);
        }
        sub_164B780(v272, &v291);
        v257 = a1[6];
        if ( v257 )
        {
          v288 = a1[6];
          sub_1623A60(&v288, v257, 2);
          v258 = v272 + 48;
          if ( *(_QWORD *)(v272 + 48) )
            sub_161E7C0(v258);
          v259 = v288;
          *(_QWORD *)(v272 + 48) = v288;
          if ( v259 )
            sub_1623210(&v288, v259, v258);
        }
      }
      else
      {
        v272 = sub_15A4750(v272, v197, 0);
      }
    }
    if ( *(_BYTE *)(a13 + 137) != 8 )
    {
      v198 = a1[5];
      v294 = "oldval";
      v296 = 259;
      v199 = sub_1643330(v198);
      v200 = sub_1648A60(64, 1);
      v201 = (_QWORD *)v200;
      if ( v200 )
        sub_15F9210(v200, v199, v286, 0, v270, 0);
      v202 = a1[7];
      if ( v202 )
      {
        v203 = (unsigned __int64 *)a1[8];
        sub_157E9D0(v202 + 40, v201);
        v204 = v201[3];
        v205 = *v203;
        v201[4] = v203;
        v205 &= 0xFFFFFFFFFFFFFFF8LL;
        v201[3] = v205 | v204 & 7;
        *(_QWORD *)(v205 + 8) = v201 + 3;
        *v203 = *v203 & 7 | (unsigned __int64)(v201 + 3);
      }
      sub_164B780(v201, &v294);
      v206 = a1[6];
      if ( v206 )
      {
        v291 = (char *)a1[6];
        sub_1623A60(&v291, v206, 2);
        if ( v201[6] )
          sub_161E7C0(v201 + 6);
        v207 = v291;
        v201[6] = v291;
        if ( v207 )
          sub_1623210(&v291, v207, v201 + 6);
      }
      v295 = 8;
      v294 = 0;
      v208 = *(unsigned __int8 *)(a13 + 136);
      v209 = (unsigned int)v208 + *(unsigned __int8 *)(a13 + 137);
      if ( (_DWORD)v209 == (_DWORD)v208 )
      {
        v210 = 255;
      }
      else
      {
        if ( (unsigned int)v208 > 0x3F || (unsigned int)v209 > 0x40 )
        {
          sub_16A5260(&v294, v208, v209);
          v211 = v295;
          if ( v295 <= 0x40 )
          {
            v210 = ~(unsigned __int64)v294 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v295);
          }
          else
          {
            sub_16A8F40(&v294);
            v211 = v295;
            v210 = (unsigned __int64)v294;
          }
LABEL_233:
          v212 = a1[5];
          v289 = v211;
          v288 = v210;
          v213 = sub_159C0E0(v212, &v288);
          v296 = 259;
          v294 = "masked";
          v214 = sub_1281C00(v279, (__int64)v201, v213, (__int64)&v294);
          v215 = *(_BYTE *)(a13 + 136);
          if ( v215 )
          {
            v294 = "bf.position";
            v296 = 259;
            v272 = sub_1281EE0(v279, v272, v215, (__int64)&v294, 0, 0);
          }
          v291 = "bf0.merged";
          v293 = 259;
          if ( *(_BYTE *)(v272 + 16) <= 0x10u )
          {
            if ( (unsigned __int8)sub_1593BB0(v272) )
            {
LABEL_239:
              v296 = 257;
              v216 = sub_1648A60(64, 2);
              v217 = (_QWORD *)v216;
              if ( v216 )
                sub_15F9650(v216, v214, v286, v270, 0);
              v218 = a1[7];
              if ( v218 )
              {
                v219 = (unsigned __int64 *)a1[8];
                sub_157E9D0(v218 + 40, v217);
                v220 = v217[3];
                v221 = *v219;
                v217[4] = v219;
                v221 &= 0xFFFFFFFFFFFFFFF8LL;
                v217[3] = v221 | v220 & 7;
                *(_QWORD *)(v221 + 8) = v217 + 3;
                *v219 = *v219 & 7 | (unsigned __int64)(v217 + 3);
              }
              result = sub_164B780(v217, &v294);
              v222 = a1[6];
              if ( v222 )
              {
                v291 = (char *)a1[6];
                result = sub_1623A60(&v291, v222, 2);
                if ( v217[6] )
                  result = sub_161E7C0(v217 + 6);
                v223 = v291;
                v217[6] = v291;
                if ( v223 )
                  result = sub_1623210(&v291, v223, v217 + 6);
              }
              if ( v289 > 0x40 && v288 )
                return j_j___libc_free_0_0(v288);
              return result;
            }
            if ( *(_BYTE *)(v214 + 16) <= 0x10u )
            {
              v214 = sub_15A2D10(v214, v272);
              goto LABEL_239;
            }
          }
          v296 = 257;
          v214 = sub_15FB440(27, v214, v272, &v294, 0);
          v247 = a1[7];
          if ( v247 )
          {
            v248 = (__int64 *)a1[8];
            sub_157E9D0(v247 + 40, v214);
            v249 = *(_QWORD *)(v214 + 24);
            v250 = *v248;
            *(_QWORD *)(v214 + 32) = v248;
            v250 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v214 + 24) = v250 | v249 & 7;
            *(_QWORD *)(v250 + 8) = v214 + 24;
            *v248 = *v248 & 7 | (v214 + 24);
          }
          sub_164B780(v214, &v291);
          v251 = a1[6];
          if ( v251 )
          {
            v287 = a1[6];
            sub_1623A60(&v287, v251, 2);
            if ( *(_QWORD *)(v214 + 48) )
              sub_161E7C0(v214 + 48);
            v252 = v287;
            *(_QWORD *)(v214 + 48) = v287;
            if ( v252 )
              sub_1623210(&v287, v252, v214 + 48);
          }
          goto LABEL_239;
        }
        v210 = (unsigned __int8)~(0xFFFFFFFFFFFFFFFFLL >> (64 - *(_BYTE *)(a13 + 137)) << v208);
      }
      v211 = 8;
      goto LABEL_233;
    }
    v296 = 257;
    v240 = sub_1648A60(64, 2);
    v241 = (_QWORD *)v240;
    if ( v240 )
      sub_15F9650(v240, v272, v286, v270, 0);
    v242 = a1[7];
    if ( v242 )
    {
      v243 = (unsigned __int64 *)a1[8];
      sub_157E9D0(v242 + 40, v241);
      v244 = v241[3];
      v245 = *v243;
      v241[4] = v243;
      v245 &= 0xFFFFFFFFFFFFFFF8LL;
      v241[3] = v245 | v244 & 7;
      *(_QWORD *)(v245 + 8) = v241 + 3;
      *v243 = *v243 & 7 | (unsigned __int64)(v241 + 3);
    }
    result = sub_164B780(v241, &v294);
    v246 = a1[6];
    if ( !v246 )
      return result;
    v291 = (char *)a1[6];
    v166 = v241 + 6;
    result = sub_1623A60(&v291, v246, 2);
    if ( v241[6] )
      result = sub_161E7C0(v241 + 6);
    v167 = v291;
    v241[6] = v291;
    if ( !v167 )
      return result;
    return sub_1623210(&v291, v167, v166);
  }
  result = *(unsigned __int8 *)(a13 + 137);
  v276 = result;
  if ( !v38 )
  {
    v43 = sub_1643350(a1[5]);
    v44 = v267;
    v45 = sub_159C470(v43, v267, 0);
    v46 = a1[5];
    v47 = v45;
    v296 = 257;
    v48 = sub_1643330(v46);
    v49 = sub_12815B0(v279, v48, v269, v47, (__int64)&v294);
    v50 = v272;
    v284 = v49;
    while ( 1 )
    {
      v51 = a1[5];
      v291 = "trunc";
      v293 = 259;
      v52 = sub_1644900(v51, 8);
      if ( v52 == *(_QWORD *)v50 )
        goto LABEL_36;
      if ( *(_BYTE *)(v50 + 16) <= 0x10u )
        break;
      v296 = 257;
      v92 = sub_15FE0A0(v50, v52, 0, &v294, 0);
      v93 = a1[7];
      v50 = v92;
      if ( v93 )
      {
        v94 = (__int64 *)a1[8];
        sub_157E9D0(v93 + 40, v92);
        v95 = *(_QWORD *)(v50 + 24);
        v96 = *v94;
        *(_QWORD *)(v50 + 32) = v94;
        v96 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v50 + 24) = v96 | v95 & 7;
        *(_QWORD *)(v96 + 8) = v50 + 24;
        *v94 = *v94 & 7 | (v50 + 24);
      }
      sub_164B780(v50, &v291);
      v97 = a1[6];
      if ( !v97 )
        goto LABEL_36;
      v288 = a1[6];
      sub_1623A60(&v288, v97, 2);
      if ( *(_QWORD *)(v50 + 48) )
        sub_161E7C0(v50 + 48);
      v98 = v288;
      *(_QWORD *)(v50 + 48) = v288;
      if ( !v98 )
        goto LABEL_36;
      sub_1623210(&v288, v98, v50 + 48);
      if ( v267 != v44 )
      {
LABEL_37:
        if ( v281 == v44 && v276 <= 7 )
        {
          v53 = a1[5];
          v294 = "oldbyte";
          v296 = 259;
          v261 = sub_1643330(v53);
          v54 = sub_1648A60(64, 1);
          v55 = (_QWORD *)v54;
          if ( v54 )
            sub_15F9210(v54, v261, v284, 0, v270, 0);
          v56 = a1[7];
          if ( v56 )
          {
            v262 = (unsigned __int64 *)a1[8];
            sub_157E9D0(v56 + 40, v55);
            v57 = *v262;
            v58 = v55[3] & 7LL;
            v55[4] = v262;
            v57 &= 0xFFFFFFFFFFFFFFF8LL;
            v55[3] = v57 | v58;
            *(_QWORD *)(v57 + 8) = v55 + 3;
            *v262 = *v262 & 7 | (unsigned __int64)(v55 + 3);
          }
          sub_164B780(v55, &v294);
          v59 = a1[6];
          if ( v59 )
          {
            v291 = (char *)a1[6];
            sub_1623A60(&v291, v59, 2);
            v60 = v55 + 6;
            if ( v55[6] )
            {
              sub_161E7C0(v55 + 6);
              v60 = v55 + 6;
            }
            v61 = v291;
            v55[6] = v291;
            if ( v61 )
              sub_1623210(&v291, v61, v60);
          }
          v62 = a1[5];
          v295 = 8;
          v294 = (char *)(0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v276 + 56) << v276);
          v63 = sub_159C0E0(v62, &v294);
          v64 = v63;
          if ( v295 > 0x40 && v294 )
          {
            v263 = v63;
            j_j___libc_free_0_0(v294);
            v64 = v263;
          }
          v294 = "preserved";
          v296 = 259;
          v65 = sub_1281C00(v279, (__int64)v55, v64, (__int64)&v294);
          v293 = 259;
          v291 = "bf2.merge";
          if ( *(_BYTE *)(v50 + 16) > 0x10u )
            goto LABEL_127;
          if ( (unsigned __int8)sub_1593BB0(v50) )
            goto LABEL_55;
          if ( *(_BYTE *)(v65 + 16) > 0x10u )
          {
LABEL_127:
            v296 = 257;
            v121 = sub_15FB440(27, v65, v50, &v294, 0);
            v122 = a1[7];
            v65 = v121;
            if ( v122 )
            {
              v123 = (__int64 *)a1[8];
              sub_157E9D0(v122 + 40, v121);
              v124 = *(_QWORD *)(v65 + 24);
              v125 = *v123;
              *(_QWORD *)(v65 + 32) = v123;
              v125 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v65 + 24) = v125 | v124 & 7;
              *(_QWORD *)(v125 + 8) = v65 + 24;
              *v123 = *v123 & 7 | (v65 + 24);
            }
            sub_164B780(v65, &v291);
            v126 = a1[6];
            if ( v126 )
            {
              v288 = a1[6];
              sub_1623A60(&v288, v126, 2);
              if ( *(_QWORD *)(v65 + 48) )
                sub_161E7C0(v65 + 48);
              v127 = v288;
              *(_QWORD *)(v65 + 48) = v288;
              if ( v127 )
                sub_1623210(&v288, v127, v65 + 48);
            }
          }
          else
          {
            v65 = sub_15A2D10(v65, v50);
          }
LABEL_55:
          v296 = 257;
          v66 = sub_1648A60(64, 2);
          v67 = (_QWORD *)v66;
          if ( v66 )
            sub_15F9650(v66, v65, v284, v270, 0);
          v68 = a1[7];
          if ( v68 )
          {
            v69 = (unsigned __int64 *)a1[8];
            sub_157E9D0(v68 + 40, v67);
            v70 = v67[3];
            v71 = *v69;
            v67[4] = v69;
            v71 &= 0xFFFFFFFFFFFFFFF8LL;
            v67[3] = v71 | v70 & 7;
            *(_QWORD *)(v71 + 8) = v67 + 3;
            *v69 = *v69 & 7 | (unsigned __int64)(v67 + 3);
          }
          result = sub_164B780(v67, &v294);
          v72 = a1[6];
          if ( v72 )
          {
            v291 = (char *)a1[6];
            result = sub_1623A60(&v291, v72, 2);
            if ( v67[6] )
              result = sub_161E7C0(v67 + 6);
            v73 = v291;
            v67[6] = v291;
            if ( v73 )
              result = sub_1623210(&v291, v73, v67 + 6);
          }
LABEL_64:
          if ( v281 < ++v44 )
            return result;
          goto LABEL_65;
        }
        goto LABEL_79;
      }
LABEL_97:
      if ( *(_BYTE *)(a13 + 136) )
      {
        v99 = a1[5];
        v294 = "oldbyte";
        v296 = 259;
        v264 = sub_1643330(v99);
        v100 = sub_1648A60(64, 1);
        v101 = (_QWORD *)v100;
        if ( v100 )
          sub_15F9210(v100, v264, v284, 0, v270, 0);
        v102 = a1[7];
        if ( v102 )
        {
          v265 = (unsigned __int64 *)a1[8];
          sub_157E9D0(v102 + 40, v101);
          v103 = *v265;
          v104 = v101[3] & 7LL;
          v101[4] = v265;
          v103 &= 0xFFFFFFFFFFFFFFF8LL;
          v101[3] = v103 | v104;
          *(_QWORD *)(v103 + 8) = v101 + 3;
          *v265 = *v265 & 7 | (unsigned __int64)(v101 + 3);
        }
        sub_164B780(v101, &v294);
        v105 = a1[6];
        if ( v105 )
        {
          v291 = (char *)a1[6];
          sub_1623A60(&v291, v105, 2);
          v106 = v101 + 6;
          if ( v101[6] )
          {
            sub_161E7C0(v101 + 6);
            v106 = v101 + 6;
          }
          v107 = v291;
          v101[6] = v291;
          if ( v107 )
            sub_1623210(&v291, v107, v106);
        }
        v295 = 8;
        v294 = 0;
        v108 = *(unsigned __int8 *)(a13 + 136);
        if ( *(_BYTE *)(a13 + 136) )
        {
          if ( (unsigned int)v108 > 0x40 )
            sub_16A5260(&v294, 0, v108);
          else
            v294 = (char *)(0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v108));
        }
        v109 = sub_159C0E0(a1[5], &v294);
        v110 = v109;
        if ( v295 > 0x40 && v294 )
        {
          v266 = v109;
          j_j___libc_free_0_0(v294);
          v110 = v266;
        }
        v294 = "preserve";
        v296 = 259;
        v111 = sub_1281C00(v279, (__int64)v101, v110, (__int64)&v294);
        v294 = "bf.position";
        v296 = 259;
        v112 = sub_1281EE0(v279, v50, *(unsigned __int8 *)(a13 + 136), (__int64)&v294, 0, 0);
        v293 = 259;
        v291 = "bf1.merge";
        if ( *(_BYTE *)(v112 + 16) <= 0x10u )
        {
          if ( (unsigned __int8)sub_1593BB0(v112) )
          {
LABEL_117:
            v296 = 257;
            v113 = sub_1648A60(64, 2);
            v114 = (_QWORD *)v113;
            if ( v113 )
              sub_15F9650(v113, v111, v284, v270, 0);
            v115 = a1[7];
            if ( v115 )
            {
              v116 = (unsigned __int64 *)a1[8];
              sub_157E9D0(v115 + 40, v114);
              v117 = v114[3];
              v118 = *v116;
              v114[4] = v116;
              v118 &= 0xFFFFFFFFFFFFFFF8LL;
              v114[3] = v118 | v117 & 7;
              *(_QWORD *)(v118 + 8) = v114 + 3;
              *v116 = *v116 & 7 | (unsigned __int64)(v114 + 3);
            }
            sub_164B780(v114, &v294);
            v119 = a1[6];
            if ( v119 )
            {
              v291 = (char *)a1[6];
              sub_1623A60(&v291, v119, 2);
              if ( v114[6] )
                sub_161E7C0(v114 + 6);
              v120 = v291;
              v114[6] = v291;
              if ( v120 )
                sub_1623210(&v291, v120, v114 + 6);
            }
            result = 8 - *(unsigned __int8 *)(a13 + 136);
            v277 += result;
            v276 -= result;
            goto LABEL_64;
          }
          if ( *(_BYTE *)(v111 + 16) <= 0x10u )
          {
            v111 = sub_15A2D10(v111, v112);
            goto LABEL_117;
          }
        }
        v296 = 257;
        v111 = sub_15FB440(27, v111, v112, &v294, 0);
        v175 = a1[7];
        if ( v175 )
        {
          v176 = (__int64 *)a1[8];
          sub_157E9D0(v175 + 40, v111);
          v177 = *(_QWORD *)(v111 + 24);
          v178 = *v176;
          *(_QWORD *)(v111 + 32) = v176;
          v178 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v111 + 24) = v178 | v177 & 7;
          *(_QWORD *)(v178 + 8) = v111 + 24;
          *v176 = *v176 & 7 | (v111 + 24);
        }
        sub_164B780(v111, &v291);
        v179 = a1[6];
        if ( v179 )
        {
          v288 = a1[6];
          sub_1623A60(&v288, v179, 2);
          if ( *(_QWORD *)(v111 + 48) )
            sub_161E7C0(v111 + 48);
          v180 = v288;
          *(_QWORD *)(v111 + 48) = v288;
          if ( v180 )
            sub_1623210(&v288, v180, v111 + 48);
        }
        goto LABEL_117;
      }
LABEL_79:
      v296 = 257;
      v84 = sub_1648A60(64, 2);
      v85 = (_QWORD *)v84;
      if ( v84 )
        sub_15F9650(v84, v50, v284, v270, 0);
      v86 = a1[7];
      if ( v86 )
      {
        v87 = (unsigned __int64 *)a1[8];
        sub_157E9D0(v86 + 40, v85);
        v88 = v85[3];
        v89 = *v87;
        v85[4] = v87;
        v89 &= 0xFFFFFFFFFFFFFFF8LL;
        v85[3] = v89 | v88 & 7;
        *(_QWORD *)(v89 + 8) = v85 + 3;
        *v87 = *v87 & 7 | (unsigned __int64)(v85 + 3);
      }
      result = sub_164B780(v85, &v294);
      v90 = a1[6];
      if ( v90 )
      {
        v291 = (char *)a1[6];
        result = sub_1623A60(&v291, v90, 2);
        if ( v85[6] )
          result = sub_161E7C0(v85 + 6);
        v91 = v291;
        v85[6] = v291;
        if ( v91 )
          result = sub_1623210(&v291, v91, v85 + 6);
      }
      v277 += 8;
      ++v44;
      v276 -= 8LL;
      if ( v281 < v44 )
        return result;
LABEL_65:
      v74 = sub_1643350(a1[5]);
      v75 = sub_159C470(v74, v44, 0);
      v76 = a1[5];
      v296 = 257;
      v77 = v75;
      v78 = sub_1643330(v76);
      v79 = sub_12815B0(v279, v78, v269, v77, (__int64)&v294);
      v50 = v272;
      v284 = v79;
      if ( v277 )
      {
        v293 = 257;
        v80 = sub_15A0680(*(_QWORD *)v272, v277, 0);
        if ( *(_BYTE *)(v272 + 16) > 0x10u || *(_BYTE *)(v80 + 16) > 0x10u )
        {
          v296 = 257;
          v128 = sub_15FB440(24, v272, v80, &v294, 0);
          v129 = a1[7];
          v50 = v128;
          if ( v129 )
          {
            v130 = (__int64 *)a1[8];
            sub_157E9D0(v129 + 40, v128);
            v131 = *(_QWORD *)(v50 + 24);
            v132 = *v130;
            *(_QWORD *)(v50 + 32) = v130;
            v132 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v50 + 24) = v132 | v131 & 7;
            *(_QWORD *)(v132 + 8) = v50 + 24;
            *v130 = *v130 & 7 | (v50 + 24);
          }
          sub_164B780(v50, &v291);
          v133 = a1[6];
          if ( v133 )
          {
            v288 = a1[6];
            sub_1623A60(&v288, v133, 2);
            if ( *(_QWORD *)(v50 + 48) )
              sub_161E7C0(v50 + 48);
            v134 = v288;
            *(_QWORD *)(v50 + 48) = v288;
            if ( v134 )
              sub_1623210(&v288, v134, v50 + 48);
          }
        }
        else
        {
          v50 = sub_15A2D80(v272, v80, 0);
        }
      }
    }
    v50 = sub_15A4750(v50, v52, 0);
LABEL_36:
    if ( v267 != v44 )
      goto LABEL_37;
    goto LABEL_97;
  }
  return result;
}
