// Function: sub_18457F0
// Address: 0x18457f0
//
__int64 __fastcall sub_18457F0(
        __int64 a1,
        _BYTE *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12,
        _BYTE *a13,
        int a14)
{
  __int64 v14; // rax
  __int64 v15; // r12
  _QWORD *v16; // rax
  _BYTE *v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rax
  unsigned __int64 v22; // r13
  _BYTE *v23; // rdi
  __int64 *v24; // rbx
  __int64 *v25; // rdx
  __int64 *v26; // r14
  unsigned int v27; // r12d
  __int64 v28; // rax
  _BYTE *v29; // rsi
  int v30; // r8d
  __int64 v31; // r9
  __int64 v32; // rax
  unsigned int v33; // r12d
  __int64 v35; // rdx
  char v36; // al
  __int64 v37; // rbx
  char *v38; // rdi
  char *v39; // rdx
  __int64 v40; // rax
  __int64 *v41; // rax
  __int64 v42; // r12
  __int64 *v43; // rax
  __int64 v44; // rax
  _QWORD *v45; // r14
  unsigned __int64 v46; // rbx
  __int64 v47; // r13
  __int64 *v48; // rax
  __int64 v49; // rbx
  char v50; // r12
  __int64 v51; // rax
  char v52; // r12
  __int64 v53; // rax
  _QWORD *v54; // rsi
  __int64 v55; // rcx
  unsigned __int64 v56; // rcx
  double v57; // xmm4_8
  double v58; // xmm5_8
  __int64 v59; // rdi
  __int64 *v60; // r14
  unsigned __int64 v61; // rax
  unsigned __int8 v62; // dl
  __int64 v63; // rcx
  __int64 v64; // rax
  __int64 *v65; // rax
  unsigned __int64 v66; // r13
  int v67; // eax
  __int64 *v68; // r12
  int v69; // ebx
  __int64 *v70; // r13
  unsigned int v71; // r12d
  __int64 *v72; // r15
  __int64 v73; // rax
  __int64 *v74; // rsi
  int v75; // r8d
  int v76; // r9d
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rdx
  __int64 v80; // r15
  int v81; // r15d
  __int64 v82; // rax
  __int64 v83; // rdx
  __int64 v84; // rax
  __int64 *j; // r13
  __int64 *v86; // rsi
  int v87; // r8d
  __int64 v88; // r9
  __int64 v89; // rax
  __int64 v90; // rax
  __int64 *v91; // rax
  __int64 v92; // rax
  _QWORD *v93; // r13
  unsigned __int64 v94; // rbx
  __int64 v95; // r12
  __int64 *v96; // rax
  __int64 v97; // r12
  __int64 v98; // r9
  __int64 *v99; // r13
  __int64 v100; // rbx
  __int64 v101; // rsi
  __int64 v102; // rdx
  int v103; // ecx
  __int64 v104; // rax
  _QWORD *v105; // rax
  __int64 v106; // r15
  unsigned __int64 v107; // r15
  __int16 v108; // di
  __int64 *v109; // r12
  __int16 v110; // cx
  __int64 *v111; // r13
  __int64 v112; // rsi
  __int64 v113; // rdx
  double v114; // xmm4_8
  double v115; // xmm5_8
  __int64 v116; // rdi
  double v117; // xmm4_8
  double v118; // xmm5_8
  __int64 v119; // rbx
  __int16 *v120; // r12
  __int64 v121; // rdi
  _QWORD *v122; // rbx
  unsigned __int64 *v123; // r12
  __int64 *v124; // r13
  unsigned __int64 v125; // rax
  __int64 ***v126; // r13
  __int64 ***v127; // rdx
  __int64 ***v128; // r12
  int v129; // r14d
  __int64 k; // rbx
  double v131; // xmm4_8
  double v132; // xmm5_8
  __int64 m; // r15
  __int64 v134; // r12
  _QWORD *v135; // rdi
  unsigned __int64 *v136; // rcx
  unsigned __int64 v137; // rdx
  double v138; // xmm4_8
  double v139; // xmm5_8
  __int64 v140; // rbx
  unsigned __int64 v141; // r13
  unsigned int v142; // r14d
  __int64 *v143; // r14
  __int64 v144; // rax
  __int64 v145; // rax
  _QWORD *v146; // rax
  __int64 v147; // r15
  _QWORD *v148; // r12
  __int64 v149; // rax
  __int64 v150; // rdx
  unsigned __int64 v151; // rax
  __int64 v152; // rax
  _QWORD *v153; // rax
  __int64 v154; // r12
  __int64 *v155; // rax
  int v156; // r8d
  int v157; // r9d
  __int64 v158; // r10
  __int64 v159; // rax
  __int64 v160; // rax
  __int64 v161; // rdx
  __int64 v162; // r15
  int v163; // r15d
  __int64 v164; // rax
  __int64 v165; // rdx
  __int64 v166; // rsi
  unsigned __int8 *v167; // rsi
  __int64 v168; // rbx
  __int64 v169; // r15
  __int64 v170; // r9
  __int64 v171; // rcx
  int v172; // esi
  __int64 v173; // rax
  __int64 v174; // rax
  __int64 v175; // r10
  __int64 *v176; // r12
  __int64 v177; // r13
  __int64 v178; // rcx
  __int64 v179; // rdx
  int v180; // r8d
  __int64 v181; // rcx
  __int64 v182; // rax
  __int64 v183; // rcx
  __int64 v184; // rax
  __int64 v185; // rax
  __int64 *v186; // r13
  __int64 v187; // rsi
  __int64 v188; // rax
  _QWORD *v189; // rax
  __int64 v190; // r14
  unsigned int *v191; // rdx
  __int64 *v192; // r15
  __int64 v193; // rax
  __int64 v194; // rcx
  unsigned __int64 v195; // rdx
  __int64 v196; // rdx
  __int64 v197; // rbx
  __int64 v198; // r12
  __int64 v199; // rbx
  __int64 v200; // rdx
  unsigned int v201; // esi
  __int64 *v202; // r12
  __int64 i; // r15
  __int64 v204; // rax
  char v205; // dl
  _BYTE *v206; // rsi
  _BYTE *v207; // rsi
  signed __int64 v208; // rdx
  _QWORD *v209; // rax
  __int64 v210; // [rsp+8h] [rbp-358h]
  __int64 v211; // [rsp+18h] [rbp-348h]
  __int64 v212; // [rsp+18h] [rbp-348h]
  __int64 v213; // [rsp+18h] [rbp-348h]
  unsigned int v214; // [rsp+20h] [rbp-340h]
  __int64 v215; // [rsp+20h] [rbp-340h]
  __int64 v216; // [rsp+20h] [rbp-340h]
  __int64 *v217; // [rsp+28h] [rbp-338h]
  __int64 v218; // [rsp+28h] [rbp-338h]
  __int64 v219; // [rsp+28h] [rbp-338h]
  __int64 v220; // [rsp+28h] [rbp-338h]
  __int64 v221; // [rsp+30h] [rbp-330h]
  __int64 v222; // [rsp+30h] [rbp-330h]
  __int64 *v223; // [rsp+30h] [rbp-330h]
  __int64 v224; // [rsp+40h] [rbp-320h]
  _QWORD *v225; // [rsp+40h] [rbp-320h]
  __int64 *v226; // [rsp+40h] [rbp-320h]
  __int64 v227; // [rsp+40h] [rbp-320h]
  __int64 v228; // [rsp+40h] [rbp-320h]
  __int64 v229; // [rsp+50h] [rbp-310h]
  __int64 v230; // [rsp+50h] [rbp-310h]
  int v231; // [rsp+58h] [rbp-308h]
  __int64 *v232; // [rsp+58h] [rbp-308h]
  __int64 v233; // [rsp+58h] [rbp-308h]
  __int64 v234; // [rsp+58h] [rbp-308h]
  int v235; // [rsp+58h] [rbp-308h]
  __int64 v236; // [rsp+80h] [rbp-2E0h]
  unsigned int v237; // [rsp+88h] [rbp-2D8h]
  __int64 v238; // [rsp+88h] [rbp-2D8h]
  _QWORD *v239; // [rsp+90h] [rbp-2D0h]
  __int64 v240; // [rsp+98h] [rbp-2C8h]
  __int64 **v241; // [rsp+98h] [rbp-2C8h]
  __int64 v242; // [rsp+98h] [rbp-2C8h]
  char v243; // [rsp+A0h] [rbp-2C0h]
  __int64 v244; // [rsp+A0h] [rbp-2C0h]
  __int64 *v245; // [rsp+B0h] [rbp-2B0h]
  unsigned __int64 v246; // [rsp+B0h] [rbp-2B0h]
  __int64 *v247; // [rsp+B0h] [rbp-2B0h]
  unsigned __int64 v248; // [rsp+B8h] [rbp-2A8h]
  int v249; // [rsp+C8h] [rbp-298h] BYREF
  unsigned int v250; // [rsp+CCh] [rbp-294h] BYREF
  __int64 v251; // [rsp+D0h] [rbp-290h] BYREF
  __int64 v252; // [rsp+D8h] [rbp-288h] BYREF
  __int64 v253; // [rsp+E0h] [rbp-280h] BYREF
  _QWORD *v254; // [rsp+E8h] [rbp-278h] BYREF
  _QWORD *v255; // [rsp+F0h] [rbp-270h] BYREF
  _BYTE *v256; // [rsp+F8h] [rbp-268h]
  _BYTE *v257; // [rsp+100h] [rbp-260h]
  _QWORD *v258; // [rsp+110h] [rbp-250h] BYREF
  _BYTE *v259; // [rsp+118h] [rbp-248h]
  _BYTE *v260; // [rsp+120h] [rbp-240h]
  __int64 *v261; // [rsp+130h] [rbp-230h] BYREF
  __int64 *v262; // [rsp+138h] [rbp-228h]
  __int64 *v263; // [rsp+140h] [rbp-220h]
  __int64 v264[2]; // [rsp+150h] [rbp-210h] BYREF
  __int16 v265; // [rsp+160h] [rbp-200h]
  _BYTE *v266; // [rsp+170h] [rbp-1F0h] BYREF
  __int64 v267; // [rsp+178h] [rbp-1E8h]
  _BYTE s[16]; // [rsp+180h] [rbp-1E0h] BYREF
  void *v269; // [rsp+190h] [rbp-1D0h] BYREF
  __int64 v270; // [rsp+198h] [rbp-1C8h]
  _BYTE v271[4]; // [rsp+1A0h] [rbp-1C0h] BYREF
  char v272; // [rsp+1A4h] [rbp-1BCh] BYREF
  _BYTE *v273; // [rsp+1C0h] [rbp-1A0h] BYREF
  __int64 v274; // [rsp+1C8h] [rbp-198h]
  _BYTE v275[64]; // [rsp+1D0h] [rbp-190h] BYREF
  __m128i v276; // [rsp+210h] [rbp-150h] BYREF
  _QWORD *v277; // [rsp+228h] [rbp-138h]
  __m128i v278; // [rsp+270h] [rbp-F0h] BYREF
  _QWORD *v279; // [rsp+288h] [rbp-D8h]
  __m128i v280; // [rsp+2D0h] [rbp-90h] BYREF
  __int16 v281; // [rsp+2E0h] [rbp-80h] BYREF
  _QWORD *v282; // [rsp+2E8h] [rbp-78h]

  v15 = a1;
  v16 = *(_QWORD **)(a1 + 112);
  v248 = (unsigned __int64)a2;
  if ( v16 )
  {
    v17 = (_BYTE *)(a1 + 104);
    a13 = a2;
    a2 = v17;
    do
    {
      while ( 1 )
      {
        v18 = v16[2];
        v19 = v16[3];
        if ( v16[4] >= (unsigned __int64)a13 )
          break;
        v16 = (_QWORD *)v16[3];
        if ( !v19 )
          goto LABEL_6;
      }
      a2 = v16;
      v16 = (_QWORD *)v16[2];
    }
    while ( v18 );
LABEL_6:
    if ( v17 != a2 && *((_QWORD *)a2 + 4) <= v248 )
      return 0;
  }
  v256 = 0;
  v20 = *(_QWORD *)(v248 + 24);
  v21 = *(_QWORD *)(v248 + 112);
  v273 = v275;
  v274 = 0x800000000LL;
  v257 = 0;
  v251 = v21;
  v229 = v20;
  v22 = (unsigned int)(*(_DWORD *)(v20 + 12) - 1);
  v255 = 0;
  v266 = s;
  v267 = 0xA00000000LL;
  if ( (unsigned int)v22 > 0xA )
  {
    a2 = s;
    sub_16CD150((__int64)&v266, s, v22, 1, (int)a13, a14);
    v23 = v266;
  }
  else
  {
    v23 = s;
  }
  LODWORD(v267) = v22;
  if ( v22 )
  {
    a2 = 0;
    memset(v23, 0, v22);
  }
  if ( (*(_BYTE *)(v248 + 18) & 1) != 0 )
  {
    sub_15E08E0(v248, (__int64)a2);
    v24 = *(__int64 **)(v248 + 88);
    if ( (*(_BYTE *)(v248 + 18) & 1) != 0 )
      sub_15E08E0(v248, (__int64)a2);
    v25 = *(__int64 **)(v248 + 88);
  }
  else
  {
    v24 = *(__int64 **)(v248 + 88);
    v25 = v24;
  }
  v245 = &v25[5 * *(_QWORD *)(v248 + 96)];
  if ( v245 == v24 )
  {
    v243 = 0;
  }
  else
  {
    v240 = v15;
    v26 = (__int64 *)(v15 + 48);
    v243 = 0;
    v27 = 0;
    do
    {
      v280.m128i_i32[2] = v27;
      v280.m128i_i8[12] = 1;
      v280.m128i_i64[0] = v248;
      if ( sub_18448E0(v26, v280.m128i_i64) )
      {
        v28 = *v24;
        v29 = v256;
        v278.m128i_i64[0] = *v24;
        if ( v256 == v257 )
        {
          sub_1278040((__int64)&v255, v256, &v278);
        }
        else
        {
          if ( v256 )
          {
            *(_QWORD *)v256 = v28;
            v29 = v256;
          }
          v256 = v29 + 8;
        }
        v266[v27] = 1;
        v31 = sub_1560230(&v251, v27);
        v32 = (unsigned int)v274;
        if ( (unsigned int)v274 >= HIDWORD(v274) )
        {
          v238 = v31;
          sub_16CD150((__int64)&v273, v275, 0, 8, v30, v31);
          v32 = (unsigned int)v274;
          v31 = v238;
        }
        *(_QWORD *)&v273[8 * v32] = v31;
        LODWORD(v274) = v274 + 1;
        v243 |= sub_1560290(&v251, v27, 38);
      }
      v24 += 5;
      ++v27;
    }
    while ( v24 != v245 );
    v15 = v240;
  }
  v241 = **(__int64 ****)(v229 + 16);
  v35 = **(_QWORD **)(*(_QWORD *)(v248 + 24) + 16LL);
  v36 = *(_BYTE *)(v35 + 8);
  if ( !v36 )
  {
    v237 = 0;
    v39 = v271;
LABEL_80:
    HIDWORD(v270) = 5;
    v269 = v271;
    v38 = v271;
    goto LABEL_33;
  }
  if ( v36 == 13 )
  {
    v237 = *(_DWORD *)(v35 + 12);
    goto LABEL_31;
  }
  if ( v36 != 14 )
  {
    v39 = &v272;
    v237 = 1;
    goto LABEL_80;
  }
  v237 = *(_DWORD *)(v35 + 32);
LABEL_31:
  v269 = v271;
  v37 = 4LL * v237;
  v270 = 0x500000000LL;
  if ( v237 <= 5uLL )
  {
    v38 = v271;
    v39 = &v271[v37];
  }
  else
  {
    sub_16CD150((__int64)&v269, v271, v237, 4, (int)a13, a14);
    v38 = (char *)v269;
    v39 = (char *)v269 + v37;
  }
LABEL_33:
  LODWORD(v270) = v237;
  if ( v38 != v39 )
    memset(v38, 255, v39 - v38);
  v259 = 0;
  v260 = 0;
  v258 = 0;
  if ( *((_BYTE *)v241 + 8) && !v243 )
  {
    v202 = (__int64 *)(v15 + 48);
    if ( v237 )
    {
      for ( i = 0; i != v237; ++i )
      {
        v280.m128i_i64[0] = v248;
        v280.m128i_i32[2] = i;
        v280.m128i_i8[12] = 0;
        if ( sub_18448E0(v202, v280.m128i_i64) )
        {
          v204 = **(_QWORD **)(*(_QWORD *)(v248 + 24) + 16LL);
          v205 = *(_BYTE *)(v204 + 8);
          if ( v205 == 13 )
          {
            v204 = *(_QWORD *)(*(_QWORD *)(v204 + 16) + 8 * i);
          }
          else if ( v205 == 14 )
          {
            v204 = *(_QWORD *)(v204 + 24);
          }
          v278.m128i_i64[0] = v204;
          v206 = v259;
          if ( v259 == v260 )
          {
            sub_1278040((__int64)&v258, v259, &v278);
            v207 = v259;
          }
          else
          {
            if ( v259 )
            {
              *(_QWORD *)v259 = v204;
              v206 = v259;
            }
            v207 = v206 + 8;
            v259 = v207;
          }
          *((_DWORD *)v269 + i) = ((v207 - (_BYTE *)v258) >> 3) - 1;
        }
      }
      if ( (unsigned __int64)(v259 - (_BYTE *)v258) > 8 )
      {
        v208 = (v259 - (_BYTE *)v258) >> 3;
        if ( *((_BYTE *)v241 + 8) == 13 )
          v244 = sub_1645600(*v241, v258, v208, ((_DWORD)v241[1] & 0x200) != 0);
        else
          v244 = (__int64)sub_1645D80((__int64 *)*v258, v208);
        goto LABEL_38;
      }
      if ( v259 - (_BYTE *)v258 == 8 )
      {
        v244 = *v258;
        goto LABEL_38;
      }
      if ( v258 != (_QWORD *)v259 )
      {
        v14 = sub_1560240(&v251);
        sub_1563030(&v276, v14);
        BUG();
      }
    }
    v209 = (_QWORD *)sub_15E0530(v248);
    v244 = sub_1643270(v209);
  }
  else
  {
    v244 = (__int64)v241;
  }
LABEL_38:
  v40 = sub_1560240(&v251);
  sub_1563030(&v276, v40);
  if ( !*(_BYTE *)(v244 + 8) )
  {
    sub_1560E30((__int64)&v280, v244);
    sub_1561FA0((__int64)&v276, &v280);
    sub_1842330(v282);
  }
  v41 = (__int64 *)sub_15E0530(v248);
  v42 = sub_1560BF0(v41, &v276);
  v280.m128i_i64[0] = sub_1560250(&v251);
  v43 = (__int64 *)sub_15E0530(v248);
  v44 = sub_1563B90(v280.m128i_i64, v43, 2);
  v45 = v273;
  v46 = (unsigned int)v274;
  v47 = v44;
  v48 = (__int64 *)sub_15E0530(v248);
  v49 = sub_155FDB0(v48, v47, v42, v45, v46);
  v210 = sub_1644EA0((__int64 *)v244, v255, (v256 - (_BYTE *)v255) >> 3, *(_DWORD *)(v229 + 8) >> 8 != 0);
  if ( v210 == v229 )
  {
    v33 = 0;
    goto LABEL_220;
  }
  v50 = *(_BYTE *)(v248 + 32);
  v281 = 257;
  v51 = sub_1648B60(120);
  v52 = v50 & 0xF;
  v239 = (_QWORD *)v51;
  if ( v51 )
    sub_15E2490(v51, v210, v52, (__int64)&v280, 0);
  sub_15E4330((__int64)v239, v248);
  v53 = *(_QWORD *)(v248 + 48);
  v239[14] = v49;
  v239[6] = v53;
  sub_1631B60(*(_QWORD *)(v248 + 40) + 24LL, (__int64)v239);
  v54 = (_QWORD *)v248;
  v55 = *(_QWORD *)(v248 + 56);
  v239[8] = v248 + 56;
  v55 &= 0xFFFFFFFFFFFFFFF8LL;
  v239[7] = v55 | v239[7] & 7LL;
  *(_QWORD *)(v55 + 8) = v239 + 7;
  *(_QWORD *)(v248 + 56) = *(_QWORD *)(v248 + 56) & 7LL | (unsigned __int64)(v239 + 7);
  sub_164B7C0((__int64)v239, v248);
  v59 = *(_QWORD *)(v248 + 8);
  v261 = 0;
  v60 = &v253;
  v262 = 0;
  v263 = 0;
  if ( !v59 )
    goto LABEL_111;
  do
  {
    v61 = (unsigned __int64)sub_1648700(v59);
    v62 = *(_BYTE *)(v61 + 16);
    if ( v62 <= 0x17u )
    {
      v246 = 0;
      v63 = 0;
    }
    else if ( v62 == 78 )
    {
      v246 = v61 & 0xFFFFFFFFFFFFFFF8LL;
      v63 = v61 | 4;
    }
    else
    {
      v246 = 0;
      v63 = 0;
      if ( v62 == 29 )
      {
        v246 = v61 & 0xFFFFFFFFFFFFFFF8LL;
        v63 = v61 & 0xFFFFFFFFFFFFFFFBLL;
      }
    }
    v252 = v63;
    LODWORD(v274) = 0;
    v253 = *(_QWORD *)(v246 + 56);
    v64 = sub_1560240(v60);
    sub_1563030(&v278, v64);
    sub_1560E30((__int64)&v280, v244);
    sub_1561FA0((__int64)&v278, &v280);
    sub_1842330(v282);
    v65 = (__int64 *)sub_15E0530(v248);
    v236 = sub_1560BF0(v65, &v278);
    v66 = v252 & 0xFFFFFFFFFFFFFFF8LL;
    v67 = *(_DWORD *)((v252 & 0xFFFFFFFFFFFFFFF8LL) + 20);
    if ( (v252 & 4) != 0 )
    {
      v68 = (__int64 *)(v66 - 24LL * (v67 & 0xFFFFFFF));
      v231 = *(_DWORD *)(v229 + 12);
      v69 = v231 - 1;
      if ( v231 == 1 )
        goto LABEL_63;
    }
    else
    {
      v68 = (__int64 *)(v66 - 24LL * (v67 & 0xFFFFFFF));
      v235 = *(_DWORD *)(v229 + 12);
      v69 = v235 - 1;
      if ( v235 == 1 )
        goto LABEL_160;
    }
    v70 = v68;
    v232 = v68;
    v71 = 0;
    v72 = v70;
    do
    {
      while ( !v266[v71] )
      {
LABEL_51:
        ++v71;
        v72 += 3;
        if ( v71 == v69 )
          goto LABEL_62;
      }
      v73 = *v72;
      v74 = v262;
      v280.m128i_i64[0] = *v72;
      if ( v262 == v263 )
      {
        sub_12879C0((__int64)&v261, v262, &v280);
      }
      else
      {
        if ( v262 )
        {
          *v262 = v73;
          v74 = v262;
        }
        v262 = v74 + 1;
      }
      v264[0] = sub_1560230(v60, v71);
      if ( (__int64 **)v244 != v241 && (unsigned __int8)sub_155EE10((__int64)v264, 38) )
      {
        sub_1563030(&v280, v264[0]);
        v225 = sub_1560700(&v280, 38);
        v155 = (__int64 *)sub_15E0530(v248);
        v158 = sub_1560BF0(v155, v225);
        v159 = (unsigned int)v274;
        if ( (unsigned int)v274 >= HIDWORD(v274) )
        {
          v228 = v158;
          sub_16CD150((__int64)&v273, v275, 0, 8, v156, v157);
          v159 = (unsigned int)v274;
          v158 = v228;
        }
        *(_QWORD *)&v273[8 * v159] = v158;
        LODWORD(v274) = v274 + 1;
        sub_1842330(v282);
        goto LABEL_51;
      }
      v77 = (unsigned int)v274;
      if ( (unsigned int)v274 >= HIDWORD(v274) )
      {
        sub_16CD150((__int64)&v273, v275, 0, 8, v75, v76);
        v77 = (unsigned int)v274;
      }
      ++v71;
      v72 += 3;
      *(_QWORD *)&v273[8 * v77] = v264[0];
      LODWORD(v274) = v274 + 1;
    }
    while ( v71 != v69 );
LABEL_62:
    v68 = &v232[3 * (unsigned int)(v69 - 1) + 3];
    v66 = v252 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v252 & 4) != 0 )
    {
LABEL_63:
      if ( *(char *)(v66 + 23) < 0 )
      {
        v78 = sub_1648A40(v66);
        v80 = v78 + v79;
        if ( *(char *)(v66 + 23) >= 0 )
        {
          if ( (unsigned int)(v80 >> 4) )
            goto LABEL_282;
        }
        else if ( (unsigned int)((v80 - sub_1648A40(v66)) >> 4) )
        {
          if ( *(char *)(v66 + 23) < 0 )
          {
            v81 = *(_DWORD *)(sub_1648A40(v66) + 8);
            if ( *(char *)(v66 + 23) >= 0 )
              BUG();
            v82 = sub_1648A40(v66);
            v84 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v82 + v83 - 4) - v81);
            goto LABEL_69;
          }
LABEL_282:
          BUG();
        }
      }
      v84 = -24;
      goto LABEL_69;
    }
LABEL_160:
    if ( *(char *)(v66 + 23) >= 0 )
      goto LABEL_184;
    v160 = sub_1648A40(v66);
    v162 = v160 + v161;
    if ( *(char *)(v66 + 23) >= 0 )
    {
      if ( !(unsigned int)(v162 >> 4) )
      {
LABEL_184:
        v84 = -72;
        goto LABEL_69;
      }
LABEL_283:
      BUG();
    }
    if ( !(unsigned int)((v162 - sub_1648A40(v66)) >> 4) )
      goto LABEL_184;
    if ( *(char *)(v66 + 23) >= 0 )
      goto LABEL_283;
    v163 = *(_DWORD *)(sub_1648A40(v66) + 8);
    if ( *(char *)(v66 + 23) >= 0 )
      BUG();
    v164 = sub_1648A40(v66);
    v84 = -72 - 24LL * (unsigned int)(*(_DWORD *)(v164 + v165 - 4) - v163);
LABEL_69:
    for ( j = (__int64 *)(v84 + v66); v68 != j; LODWORD(v274) = v274 + 1 )
    {
      v90 = *v68;
      v86 = v262;
      v280.m128i_i64[0] = *v68;
      if ( v262 == v263 )
      {
        sub_12879C0((__int64)&v261, v262, &v280);
      }
      else
      {
        if ( v262 )
        {
          *v262 = v90;
          v86 = v262;
        }
        v262 = v86 + 1;
      }
      v88 = sub_1560230(v60, v69);
      v89 = (unsigned int)v274;
      if ( (unsigned int)v274 >= HIDWORD(v274) )
      {
        v234 = v88;
        sub_16CD150((__int64)&v273, v275, 0, 8, v87, v88);
        v89 = (unsigned int)v274;
        v88 = v234;
      }
      v68 += 3;
      ++v69;
      *(_QWORD *)&v273[8 * v89] = v88;
    }
    v280.m128i_i64[0] = sub_1560250(v60);
    v91 = (__int64 *)sub_15E0530(v248);
    v92 = sub_1563B90(v280.m128i_i64, v91, 2);
    v93 = v273;
    v94 = (unsigned int)v274;
    v95 = v92;
    v96 = (__int64 *)sub_15E0530(v248);
    v233 = sub_155FDB0(v96, v95, v236, v93, v94);
    v280.m128i_i64[0] = (__int64)&v281;
    v280.m128i_i64[1] = 0x100000000LL;
    sub_1740980(&v252, (__int64)&v280);
    if ( *(_BYTE *)(v246 + 16) != 29 )
    {
      v265 = 257;
      v226 = v261;
      v168 = v280.m128i_i64[0];
      v169 = v262 - v261;
      v170 = v280.m128i_i64[0] + 56LL * v280.m128i_u32[2];
      if ( v280.m128i_i64[0] == v170 )
      {
        v216 = v280.m128i_u32[2];
        v220 = *(_QWORD *)(*v239 + 24LL);
        v213 = v262 - v261;
        v174 = (__int64)sub_1648AB0(72, (int)v169 + 1, 16 * v280.m128i_i32[2]);
        v180 = v169 + 1;
        v175 = v220;
        v181 = v213;
        if ( v174 )
        {
          v176 = (__int64 *)v168;
          v177 = v216;
          goto LABEL_179;
        }
      }
      else
      {
        v171 = v280.m128i_i64[0];
        v172 = 0;
        do
        {
          v173 = *(_QWORD *)(v171 + 40) - *(_QWORD *)(v171 + 32);
          v171 += 56;
          v172 += v173 >> 3;
        }
        while ( v170 != v171 );
        v212 = v280.m128i_i64[0] + 56LL * v280.m128i_u32[2];
        v215 = v280.m128i_u32[2];
        v218 = *(_QWORD *)(*v239 + 24LL);
        v174 = (__int64)sub_1648AB0(72, (int)v169 + 1 + v172, 16 * v280.m128i_i32[2]);
        v175 = v218;
        if ( v174 )
        {
          v176 = (__int64 *)v168;
          v177 = v215;
          LODWORD(v178) = 0;
          do
          {
            v179 = *(_QWORD *)(v168 + 40) - *(_QWORD *)(v168 + 32);
            v168 += 56;
            v178 = (unsigned int)(v179 >> 3) + (unsigned int)v178;
          }
          while ( v212 != v168 );
          v180 = v178 + v169 + 1;
          v181 = v169 + v178;
LABEL_179:
          v219 = v175;
          v222 = v174;
          sub_15F1EA0(v174, **(_QWORD **)(v175 + 16), 54, v174 - 24 * v181 - 24, v180, v246);
          *(_QWORD *)(v222 + 56) = 0;
          sub_15F5B40(v222, v219, (__int64)v239, v226, v169, (__int64)v264, v176, v177);
          v174 = v222;
        }
      }
      v109 = (__int64 *)(v174 & 0xFFFFFFFFFFFFFFF8LL);
      v110 = *(_WORD *)(v246 + 18) & 3 | *(_WORD *)((v174 & 0xFFFFFFFFFFFFFFF8LL) + 18) & 0xFFFC;
      *(_WORD *)((v174 & 0xFFFFFFFFFFFFFFF8LL) + 18) = v110;
      LOBYTE(v108) = v110;
      goto LABEL_88;
    }
    v97 = v280.m128i_u32[2];
    v98 = *(_QWORD *)(v246 + 40);
    v217 = v261;
    v265 = 257;
    v99 = (__int64 *)v280.m128i_i64[0];
    v100 = v262 - v261;
    v221 = *(_QWORD *)(v246 - 24);
    v224 = *(_QWORD *)(v246 - 48);
    v101 = v280.m128i_i64[0] + 56LL * v280.m128i_u32[2];
    if ( v280.m128i_i64[0] == v101 )
    {
      v103 = 0;
    }
    else
    {
      v102 = v280.m128i_i64[0];
      v103 = 0;
      do
      {
        v104 = *(_QWORD *)(v102 + 40) - *(_QWORD *)(v102 + 32);
        v102 += 56;
        v103 += v104 >> 3;
      }
      while ( v101 != v102 );
    }
    v211 = v98;
    v214 = v103 + v100 + 3;
    v105 = sub_1648AB0(72, v214, 16 * v280.m128i_i32[2]);
    v106 = (__int64)v105;
    if ( v105 )
    {
      sub_15F1F50(
        (__int64)v105,
        **(_QWORD **)(*(_QWORD *)(*v239 + 24LL) + 16LL),
        5,
        (__int64)&v105[-3 * v214],
        v214,
        v211);
      *(_QWORD *)(v106 + 56) = 0;
      sub_15F6500(v106, *(_QWORD *)(*v239 + 24LL), (__int64)v239, v224, v221, (__int64)v264, v217, v100, v99, v97);
    }
    v107 = v106 & 0xFFFFFFFFFFFFFFF8LL;
    v108 = *(_WORD *)(v107 + 18);
    v109 = (__int64 *)v107;
    v110 = v108;
LABEL_88:
    v111 = v109 + 6;
    *((_WORD *)v109 + 9) = v110 & 0x8000
                         | v108 & 3
                         | (4 * ((*(_WORD *)((v252 & 0xFFFFFFFFFFFFFFF8LL) + 18) >> 2) & 0xDFFF));
    v109[7] = v233;
    v112 = *(_QWORD *)(v246 + 48);
    v264[0] = v112;
    if ( !v112 )
    {
      if ( v111 == v264 )
        goto LABEL_92;
      v166 = v109[6];
      if ( !v166 )
        goto LABEL_92;
LABEL_169:
      sub_161E7C0((__int64)(v109 + 6), v166);
      goto LABEL_170;
    }
    sub_1623A60((__int64)v264, v112, 2);
    if ( v111 == v264 )
    {
      if ( v264[0] )
        sub_161E7C0((__int64)v264, v264[0]);
      goto LABEL_92;
    }
    v166 = v109[6];
    if ( v166 )
      goto LABEL_169;
LABEL_170:
    v167 = (unsigned __int8 *)v264[0];
    v109[6] = v264[0];
    if ( v167 )
      sub_1623210((__int64)v264, v167, (__int64)(v109 + 6));
LABEL_92:
    v54 = &v254;
    if ( (unsigned __int8)sub_1625980(v246, &v254) )
    {
      v54 = v254;
      sub_15F3B70((__int64)v109, (int)v254);
    }
    if ( v261 != v262 )
      v262 = v261;
    LODWORD(v274) = 0;
    if ( *(_QWORD *)(v246 + 8) )
    {
      v116 = *(_QWORD *)v246;
      if ( *(_QWORD *)v246 == *v109 )
      {
        v187 = (__int64)v109;
        v197 = v246;
      }
      else
      {
        if ( !*(_BYTE *)(*v109 + 8) )
        {
          if ( *(_BYTE *)(v116 + 8) != 9 )
          {
            v54 = (_QWORD *)sub_15A06D0((__int64 **)v116, (__int64)v54, v113, v246);
            sub_164D160(v246, (__int64)v54, a3, a4, a5, a6, v117, v118, a9, a10);
          }
          goto LABEL_101;
        }
        v227 = v246;
        if ( *(_BYTE *)(v246 + 16) == 29 )
        {
          v182 = sub_1AA91E0(v109[5], *(_QWORD *)(v246 - 48), 0, 0);
          v183 = sub_157EE30(v182);
          v184 = v183 - 24;
          if ( !v183 )
            v184 = 0;
          v227 = v184;
        }
        v185 = sub_1599EF0(v241);
        v249 = 0;
        v186 = (__int64 *)v185;
        v187 = v185;
        if ( v237 )
        {
          v223 = v60;
          v188 = 0;
          do
          {
            v191 = (unsigned int *)((char *)v269 + 4 * v188);
            if ( *v191 != -1 )
            {
              v192 = v109;
              if ( (unsigned __int64)(v259 - (_BYTE *)v258) > 8 )
              {
                v264[0] = (__int64)"newret";
                v265 = 259;
                v250 = *v191;
                v192 = sub_1648A60(88, 1u);
                if ( v192 )
                {
                  v193 = sub_15FB2A0(*v109, &v250, 1);
                  sub_15F1EA0((__int64)v192, v193, 62, (__int64)(v192 - 3), 1, v227);
                  if ( *(v192 - 3) )
                  {
                    v194 = *(v192 - 2);
                    v195 = *(v192 - 1) & 0xFFFFFFFFFFFFFFFCLL;
                    *(_QWORD *)v195 = v194;
                    if ( v194 )
                      *(_QWORD *)(v194 + 16) = *(_QWORD *)(v194 + 16) & 3LL | v195;
                  }
                  *(v192 - 3) = (__int64)v109;
                  if ( v109 )
                  {
                    v196 = v109[1];
                    *(v192 - 2) = v196;
                    if ( v196 )
                      *(_QWORD *)(v196 + 16) = (unsigned __int64)(v192 - 2) | *(_QWORD *)(v196 + 16) & 3LL;
                    *(v192 - 1) = (unsigned __int64)(v109 + 1) | *(v192 - 1) & 3;
                    v109[1] = (__int64)(v192 - 3);
                  }
                  v192[7] = (__int64)(v192 + 9);
                  v192[8] = 0x400000000LL;
                  sub_15FB110((__int64)v192, &v250, 1, (__int64)v264);
                }
              }
              v264[0] = (__int64)"oldret";
              v265 = 259;
              v189 = sub_1648A60(88, 2u);
              v190 = (__int64)v189;
              if ( v189 )
              {
                sub_15F1EA0((__int64)v189, *v186, 63, (__int64)(v189 - 6), 2, v227);
                *(_QWORD *)(v190 + 56) = v190 + 72;
                *(_QWORD *)(v190 + 64) = 0x400000000LL;
                sub_15FAD90(v190, (__int64)v186, (__int64)v192, &v249, 1, (__int64)v264);
              }
              v186 = (__int64 *)v190;
            }
            v188 = (unsigned int)(v249 + 1);
            v249 = v188;
          }
          while ( (_DWORD)v188 != v237 );
          v60 = v223;
          v187 = (__int64)v186;
        }
        v197 = v246;
      }
      sub_164D160(v197, v187, a3, a4, a5, a6, v114, v115, a9, a10);
      v54 = (_QWORD *)v197;
      sub_164B7C0((__int64)v109, v197);
    }
LABEL_101:
    sub_15F20C0((_QWORD *)v246);
    v119 = v280.m128i_i64[0];
    v120 = (__int16 *)(v280.m128i_i64[0] + 56LL * v280.m128i_u32[2]);
    if ( (__int16 *)v280.m128i_i64[0] != v120 )
    {
      do
      {
        v121 = *((_QWORD *)v120 - 3);
        v120 -= 28;
        if ( v121 )
        {
          v54 = (_QWORD *)(*((_QWORD *)v120 + 6) - v121);
          j_j___libc_free_0(v121, v54);
        }
        if ( *(__int16 **)v120 != v120 + 8 )
        {
          v54 = (_QWORD *)(*((_QWORD *)v120 + 2) + 1LL);
          j_j___libc_free_0(*(_QWORD *)v120, v54);
        }
      }
      while ( (__int16 *)v119 != v120 );
      v120 = (__int16 *)v280.m128i_i64[0];
    }
    if ( v120 != &v281 )
      _libc_free((unsigned __int64)v120);
    sub_1842330(v279);
    v59 = *(_QWORD *)(v248 + 8);
  }
  while ( v59 );
LABEL_111:
  v242 = (__int64)(v239 + 9);
  v122 = (_QWORD *)(v248 + 72);
  if ( v248 + 72 != (*(_QWORD *)(v248 + 72) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v123 = (unsigned __int64 *)v239[10];
    v124 = *(__int64 **)(v248 + 80);
    if ( v122 != v123 )
    {
      if ( (_QWORD *)v242 != v122 )
      {
        v54 = (_QWORD *)(v248 + 72);
        sub_15809C0(v242, v248 + 72, *(_QWORD *)(v248 + 80), v248 + 72);
      }
      if ( v122 != v123 && v122 != v124 )
      {
        v56 = *(_QWORD *)(v248 + 72) & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)((*v124 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v122;
        *(_QWORD *)(v248 + 72) = *(_QWORD *)(v248 + 72) & 7LL | *v124 & 0xFFFFFFFFFFFFFFF8LL;
        v125 = *v123;
        *(_QWORD *)(v56 + 8) = v123;
        v125 &= 0xFFFFFFFFFFFFFFF8LL;
        *v124 = v125 | *v124 & 7;
        *(_QWORD *)(v125 + 8) = v124;
        *v123 = v56 | *v123 & 7;
      }
    }
  }
  if ( (*(_BYTE *)(v248 + 18) & 1) != 0 )
  {
    sub_15E08E0(v248, (__int64)v54);
    v126 = *(__int64 ****)(v248 + 88);
    if ( (*(_BYTE *)(v248 + 18) & 1) != 0 )
      sub_15E08E0(v248, (__int64)v54);
    v127 = *(__int64 ****)(v248 + 88);
  }
  else
  {
    v126 = *(__int64 ****)(v248 + 88);
    v127 = v126;
  }
  v128 = &v127[5 * *(_QWORD *)(v248 + 96)];
  if ( (*((_BYTE *)v239 + 18) & 1) != 0 )
    sub_15E08E0((__int64)v239, (__int64)v54);
  v129 = 0;
  for ( k = v239[11]; v126 != v128; ++v129 )
  {
    while ( v266[v129] )
    {
      ++v129;
      sub_164D160((__int64)v126, k, a3, a4, a5, a6, v57, v58, a9, a10);
      v54 = v126;
      v126 += 5;
      sub_164B7C0(k, (__int64)v54);
      k += 40;
      if ( v126 == v128 )
        goto LABEL_129;
    }
    if ( *((_BYTE *)*v126 + 8) != 9 )
    {
      v54 = (_QWORD *)sub_15A06D0(*v126, (__int64)v54, (__int64)v266, v56);
      sub_164D160((__int64)v126, (__int64)v54, a3, a4, a5, a6, v131, v132, a9, a10);
    }
    v126 += 5;
  }
LABEL_129:
  if ( **(_QWORD **)(v239[3] + 16LL) != **(_QWORD **)(*(_QWORD *)(v248 + 24) + 16LL) )
  {
    for ( m = v239[10]; v242 != m; m = *(_QWORD *)(m + 8) )
    {
      v140 = m - 24;
      if ( !m )
        v140 = 0;
      v141 = sub_157EBA0(v140);
      if ( *(_BYTE *)(v141 + 16) == 25 )
      {
        v142 = 0;
        v247 = 0;
        if ( *(_BYTE *)(**(_QWORD **)(v210 + 16) + 8LL) )
        {
          v143 = *(__int64 **)(v141 - 24LL * (*(_DWORD *)(v141 + 20) & 0xFFFFFFF));
          v144 = sub_1599EF0((__int64 **)v244);
          LODWORD(v264[0]) = 0;
          v247 = (__int64 *)v144;
          if ( v237 )
          {
            v145 = 0;
            v230 = m;
            do
            {
              if ( *((_DWORD *)v269 + v145) != -1 )
              {
                v280.m128i_i64[0] = (__int64)"oldret";
                v281 = 259;
                v146 = sub_1648A60(88, 1u);
                v147 = (__int64)v146;
                if ( v146 )
                {
                  v148 = v146 - 3;
                  v149 = sub_15FB2A0(*v143, (unsigned int *)v264, 1);
                  sub_15F1EA0(v147, v149, 62, v147 - 24, 1, v141);
                  if ( *(_QWORD *)(v147 - 24) )
                  {
                    v150 = *(_QWORD *)(v147 - 16);
                    v151 = *(_QWORD *)(v147 - 8) & 0xFFFFFFFFFFFFFFFCLL;
                    *(_QWORD *)v151 = v150;
                    if ( v150 )
                      *(_QWORD *)(v150 + 16) = *(_QWORD *)(v150 + 16) & 3LL | v151;
                  }
                  *(_QWORD *)(v147 - 24) = v143;
                  v152 = v143[1];
                  *(_QWORD *)(v147 - 16) = v152;
                  if ( v152 )
                    *(_QWORD *)(v152 + 16) = (v147 - 16) | *(_QWORD *)(v152 + 16) & 3LL;
                  *(_QWORD *)(v147 - 8) = (unsigned __int64)(v143 + 1) | *(_QWORD *)(v147 - 8) & 3LL;
                  v143[1] = (__int64)v148;
                  *(_QWORD *)(v147 + 56) = v147 + 72;
                  *(_QWORD *)(v147 + 64) = 0x400000000LL;
                  sub_15FB110(v147, v264, 1, (__int64)&v280);
                }
                if ( (unsigned __int64)(v259 - (_BYTE *)v258) <= 8 )
                {
                  v247 = (__int64 *)v147;
                }
                else
                {
                  v280.m128i_i64[0] = (__int64)"newret";
                  v281 = 259;
                  v278.m128i_i32[0] = *((_DWORD *)v269 + LODWORD(v264[0]));
                  v153 = sub_1648A60(88, 2u);
                  v154 = (__int64)v153;
                  if ( v153 )
                  {
                    sub_15F1EA0((__int64)v153, *v247, 63, (__int64)(v153 - 6), 2, v141);
                    *(_QWORD *)(v154 + 56) = v154 + 72;
                    *(_QWORD *)(v154 + 64) = 0x400000000LL;
                    sub_15FAD90(v154, (__int64)v247, v147, &v278, 1, (__int64)&v280);
                  }
                  v247 = (__int64 *)v154;
                }
              }
              v145 = (unsigned int)(LODWORD(v264[0]) + 1);
              LODWORD(v264[0]) = v145;
            }
            while ( (_DWORD)v145 != v237 );
            m = v230;
          }
          v142 = v247 != 0;
        }
        v134 = sub_15E0530(v248);
        v135 = sub_1648A60(56, v142);
        if ( v135 )
          sub_15F6F90((__int64)v135, v134, (__int64)v247, v141);
        sub_157EA20(v140 + 40, v141);
        v136 = *(unsigned __int64 **)(v141 + 32);
        v137 = *(_QWORD *)(v141 + 24) & 0xFFFFFFFFFFFFFFF8LL;
        *v136 = v137 | *v136 & 7;
        *(_QWORD *)(v137 + 8) = v136;
        *(_QWORD *)(v141 + 24) &= 7uLL;
        *(_QWORD *)(v141 + 32) = 0;
        sub_164BEC0(v141, v141, v137, (__int64)v136, a3, a4, a5, a6, v138, v139, a9, a10);
      }
    }
  }
  v280.m128i_i64[0] = (__int64)&v281;
  v280.m128i_i64[1] = 0x100000000LL;
  sub_1626D60(v248, (__int64)&v280);
  v198 = v280.m128i_i64[0];
  v199 = v280.m128i_i64[0] + 16LL * v280.m128i_u32[2];
  if ( v280.m128i_i64[0] != v199 )
  {
    do
    {
      v200 = *(_QWORD *)(v198 + 8);
      v201 = *(_DWORD *)v198;
      v198 += 16;
      sub_16267C0((__int64)v239, v201, v200);
    }
    while ( v199 != v198 );
  }
  sub_15E3D00(v248);
  if ( (__int16 *)v280.m128i_i64[0] != &v281 )
    _libc_free(v280.m128i_u64[0]);
  if ( v261 )
    j_j___libc_free_0(v261, (char *)v263 - (char *)v261);
  v33 = 1;
LABEL_220:
  sub_1842330(v277);
  if ( v258 )
    j_j___libc_free_0(v258, v260 - (_BYTE *)v258);
  if ( v269 != v271 )
    _libc_free((unsigned __int64)v269);
  if ( v266 != s )
    _libc_free((unsigned __int64)v266);
  if ( v273 != v275 )
    _libc_free((unsigned __int64)v273);
  if ( v255 )
    j_j___libc_free_0(v255, v257 - (_BYTE *)v255);
  return v33;
}
