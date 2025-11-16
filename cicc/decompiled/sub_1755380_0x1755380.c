// Function: sub_1755380
// Address: 0x1755380
//
__int64 __fastcall sub_1755380(
        __int64 a1,
        __int64 a2,
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
        __int64 a13,
        int a14)
{
  __int64 v14; // r12
  __int64 **v15; // r13
  char v16; // al
  __int64 *v17; // rcx
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // rdi
  unsigned __int64 v21; // rax
  __int64 v22; // rdx
  int v23; // ebx
  __int64 v24; // rdi
  __int64 *v25; // r14
  int v26; // edx
  __int64 v27; // rdx
  char v28; // al
  __int64 v29; // r14
  __int64 v31; // rax
  __int64 v32; // rbx
  __int64 v33; // rax
  __int64 **v34; // r13
  char v35; // dl
  __int64 v36; // rdi
  char v37; // cl
  bool v38; // al
  __int64 v39; // r8
  __int16 *v40; // rdi
  __int64 v41; // r15
  unsigned __int64 v42; // rbx
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // rax
  __int16 *v46; // rdi
  __int64 v47; // r15
  __int64 v48; // rbx
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // r13
  __int64 v52; // rax
  __int64 v53; // r10
  __int64 v54; // r14
  _QWORD *v55; // rax
  _QWORD *v56; // r12
  __int64 v57; // rdi
  unsigned __int64 *v58; // r15
  __int64 v59; // rax
  unsigned __int64 v60; // rsi
  __int64 *v61; // rsi
  _QWORD *v62; // rdi
  __int64 v63; // rdx
  bool v64; // zf
  __int64 v65; // rsi
  __int64 v66; // rax
  __int64 v67; // rbx
  __int64 v68; // r13
  _QWORD *v69; // rax
  double v70; // xmm4_8
  double v71; // xmm5_8
  _QWORD *v72; // rax
  unsigned __int8 *v73; // rax
  __int64 v74; // r15
  _QWORD *v75; // rax
  __int64 **v76; // rax
  __int64 v77; // rdx
  __int64 v78; // rcx
  __int64 *v79; // r12
  _QWORD *v80; // rax
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 ****v83; // rdx
  __int64 v84; // rax
  unsigned __int8 *v85; // rax
  __int64 v86; // rdi
  unsigned __int8 *v87; // r12
  unsigned __int8 *v88; // rax
  _QWORD *v89; // r15
  __int64 v90; // r13
  _QWORD *v91; // rax
  __int64 v92; // r14
  __int64 *v93; // rbx
  __int64 v94; // rax
  __int64 v95; // r15
  __int64 v96; // r9
  __int64 v97; // rdx
  char v98; // al
  __int64 v99; // rax
  unsigned __int8 v100; // al
  __int64 v101; // rax
  __int64 **v102; // rax
  unsigned __int8 *v103; // rax
  __int64 **v104; // rax
  unsigned __int8 *v105; // r14
  unsigned int v106; // ebx
  __int64 v107; // rax
  __int64 v108; // r15
  _QWORD *v109; // rax
  __int64 v110; // rax
  __int64 v111; // rbx
  int v112; // r9d
  __int64 v113; // rbx
  unsigned int v114; // r8d
  _QWORD *v115; // rax
  __int64 **v116; // r13
  __int64 **v117; // rax
  __int64 v118; // rdi
  unsigned __int8 *v119; // rax
  unsigned __int8 *v120; // r13
  __int64 v121; // rbx
  _QWORD *v122; // rax
  unsigned __int64 v123; // rax
  __int16 *v124; // rcx
  __int64 v125; // rax
  unsigned __int8 *v126; // r15
  __int64 **v127; // r14
  __int64 v128; // rdi
  __int64 *v129; // rbx
  __int64 v130; // rbx
  int v131; // r9d
  int v132; // r8d
  __int64 v133; // rax
  int v134; // r14d
  unsigned int v135; // ebx
  __int64 v136; // r14
  __int64 v137; // rax
  _QWORD *v138; // rax
  unsigned __int64 v139; // rdi
  __int64 v140; // rbx
  _QWORD *v141; // rax
  __int64 **v142; // rax
  __int64 v143; // rdx
  __int64 v144; // rcx
  __int64 v145; // rax
  __int64 v146; // r12
  _QWORD *v147; // r15
  __int64 v148; // rax
  __int64 v149; // rbx
  __int64 v150; // r13
  _QWORD *v151; // rax
  double v152; // xmm4_8
  double v153; // xmm5_8
  __int64 v154; // rcx
  __int64 v155; // rdx
  __int64 v156; // r14
  __int64 v157; // rdx
  unsigned __int8 v158; // cl
  __int64 v159; // rax
  unsigned __int64 v160; // r12
  __int64 *v161; // rbx
  int v162; // r13d
  __int64 v163; // r14
  __int64 v164; // rax
  int v165; // r8d
  int v166; // r9d
  __int64 v167; // r15
  __int64 *v168; // rax
  unsigned int v169; // r15d
  __int64 v170; // rax
  __int64 v171; // r13
  _QWORD *v172; // rax
  __int64 v173; // rax
  __int64 *v174; // rax
  __int64 *v175; // r10
  __int64 *v176; // rcx
  __int64 *v177; // rax
  __int64 v178; // rdx
  __int64 v179; // r13
  __int64 v180; // rdx
  __int64 v181; // r9
  __int64 **v182; // rcx
  char v183; // si
  __int64 v184; // rax
  __int64 v185; // rax
  int v186; // eax
  int v187; // eax
  __int64 *v188; // r9
  __int64 v189; // r14
  unsigned __int8 *v190; // rax
  __int64 v191; // rdx
  __int64 v192; // r12
  __int64 v193; // rax
  __int64 v194; // r13
  _QWORD *v195; // rax
  _QWORD *v196; // rax
  __int64 v197; // rdx
  unsigned __int64 *v198; // r12
  __int64 v199; // rax
  unsigned __int64 v200; // rcx
  __int64 v201; // rsi
  __int64 v202; // rsi
  unsigned __int8 *v203; // rsi
  __int64 v204; // rax
  int v205; // edx
  int v206; // edx
  __int64 *v207; // rcx
  __int64 v208; // r14
  __int64 **v209; // rcx
  unsigned __int8 *v210; // rax
  __int64 *v211; // rsi
  __int64 v212; // r15
  __int64 **v213; // rcx
  __int64 *v214; // rax
  int v215; // eax
  int v216; // eax
  __int64 *v217; // rsi
  __int64 v218; // r14
  __int64 v219; // rdx
  unsigned __int8 *v220; // rax
  __int64 v221; // rax
  __int64 *v222; // r8
  int v223; // r14d
  int v224; // r9d
  __int64 v225; // rax
  __int64 v226; // r12
  int v227; // ebx
  __int64 v228; // rax
  int v229; // r9d
  int v230; // r8d
  int v231; // ebx
  int v232; // r12d
  __int64 **v233; // rax
  __int64 v234; // rax
  bool v235; // al
  bool v236; // al
  __int64 v237; // rax
  _QWORD *v238; // rax
  __int64 v239; // rax
  __int64 v240; // [rsp+8h] [rbp-168h]
  __int64 **v241; // [rsp+10h] [rbp-160h]
  int v242; // [rsp+10h] [rbp-160h]
  unsigned __int8 *v243; // [rsp+18h] [rbp-158h]
  __int64 v244; // [rsp+18h] [rbp-158h]
  __int64 *v245; // [rsp+18h] [rbp-158h]
  __int64 v246; // [rsp+28h] [rbp-148h]
  __int64 v247; // [rsp+28h] [rbp-148h]
  __int64 v248; // [rsp+28h] [rbp-148h]
  __int64 *v249; // [rsp+28h] [rbp-148h]
  __int64 v250; // [rsp+28h] [rbp-148h]
  __int64 v251; // [rsp+28h] [rbp-148h]
  __int64 v252; // [rsp+28h] [rbp-148h]
  __int64 *v253; // [rsp+30h] [rbp-140h]
  __int64 v254; // [rsp+30h] [rbp-140h]
  __int64 v255; // [rsp+30h] [rbp-140h]
  __int64 v256; // [rsp+30h] [rbp-140h]
  __int64 v257; // [rsp+30h] [rbp-140h]
  __int64 v258; // [rsp+30h] [rbp-140h]
  __int64 *v259; // [rsp+30h] [rbp-140h]
  int v260; // [rsp+30h] [rbp-140h]
  __int64 v261; // [rsp+30h] [rbp-140h]
  char *v262; // [rsp+30h] [rbp-140h]
  _QWORD *v263; // [rsp+30h] [rbp-140h]
  __int64 v264; // [rsp+30h] [rbp-140h]
  __int64 v265; // [rsp+30h] [rbp-140h]
  __int64 v266; // [rsp+30h] [rbp-140h]
  __int64 v267; // [rsp+30h] [rbp-140h]
  __int64 v268; // [rsp+30h] [rbp-140h]
  int v270; // [rsp+38h] [rbp-138h]
  __int64 v271; // [rsp+40h] [rbp-130h]
  __int64 v272; // [rsp+40h] [rbp-130h]
  _BYTE *v273; // [rsp+48h] [rbp-128h]
  __int64 v274; // [rsp+48h] [rbp-128h]
  __int64 *v275; // [rsp+48h] [rbp-128h]
  _QWORD *v276; // [rsp+50h] [rbp-120h] BYREF
  _QWORD *v277; // [rsp+58h] [rbp-118h] BYREF
  __int64 v278[2]; // [rsp+60h] [rbp-110h] BYREF
  __int16 v279; // [rsp+70h] [rbp-100h]
  __int64 v280[2]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v281; // [rsp+90h] [rbp-E0h]
  __int64 v282[2]; // [rsp+A0h] [rbp-D0h] BYREF
  __int16 v283; // [rsp+B0h] [rbp-C0h]
  __int64 *v284; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v285; // [rsp+C8h] [rbp-A8h]
  _QWORD v286[4]; // [rsp+D0h] [rbp-A0h] BYREF
  __int16 *v287; // [rsp+F0h] [rbp-80h] BYREF
  __int64 v288; // [rsp+F8h] [rbp-78h]
  __int16 v289; // [rsp+100h] [rbp-70h] BYREF
  char v290; // [rsp+108h] [rbp-68h] BYREF

  v14 = a2;
  v15 = *(__int64 ***)a2;
  v273 = *(_BYTE **)(a2 - 24);
  v271 = *(_QWORD *)v273;
  if ( *(_QWORD *)v273 == *(_QWORD *)a2 )
  {
    v67 = *(_QWORD *)(a2 + 8);
    v29 = a2;
    if ( v67 )
    {
      v68 = *(_QWORD *)a1;
      do
      {
        v69 = sub_1648700(v67);
        sub_170B990(v68, (__int64)v69);
        v67 = *(_QWORD *)(v67 + 8);
      }
      while ( v67 );
      if ( (_BYTE *)a2 == v273 )
        v273 = (_BYTE *)sub_1599EF0(*(__int64 ***)a2);
      sub_164D160(a2, (__int64)v273, a3, a4, a5, a6, v70, v71, a9, a10);
      return v29;
    }
    return 0;
  }
  v16 = *((_BYTE *)v15 + 8);
  if ( v16 == 15 )
  {
    v17 = v15[3];
    v18 = *(_QWORD *)(v271 + 24);
    if ( v17 == (__int64 *)v18 && *((_DWORD *)v15 + 2) >> 8 != *(_DWORD *)(v271 + 8) >> 8 )
    {
      v289 = 257;
      v72 = sub_1648A60(56, 1u);
      v29 = (__int64)v72;
      if ( v72 )
        sub_15FDB10((__int64)v72, (__int64)v273, (__int64)v15, (__int64)&v287, 0);
      return v29;
    }
    if ( v273[16] == 53 )
    {
      v259 = v15[3];
      v29 = sub_174CCF0((__int64 *)a1, a2, (__int64)v273, *(double *)a3.m128_u64, a4, a5, a6, a7, a8, a9, a10);
      if ( v29 )
        return v29;
      v19 = *(_QWORD *)v273;
      v17 = v259;
    }
    else
    {
      v19 = *(_QWORD *)v273;
    }
    if ( *(_BYTE *)(v19 + 8) == 16 )
      v19 = **(_QWORD **)(v19 + 16);
    v20 = *(_QWORD *)(v19 + 24);
    v21 = *(unsigned __int8 *)(v20 + 8);
    if ( (unsigned __int8)v21 > 0xFu || (v22 = 35454, !_bittest64(&v22, v21)) )
    {
      if ( (unsigned int)(v21 - 13) > 1 && (_DWORD)v21 != 16 )
        return 0;
      a2 = 0;
      v253 = v17;
      v38 = sub_16435F0(v20, 0);
      v17 = v253;
      if ( !v38 )
        return 0;
    }
    v23 = 0;
    v24 = v18;
    v25 = v17;
    if ( v17 == (__int64 *)v18 )
    {
      v204 = sub_1643350(*(_QWORD **)(*(_QWORD *)(a1 + 8) + 24LL));
      v167 = sub_159C470(v204, 0, 0);
      v288 = 0x800000001LL;
      v287 = &v289;
      v161 = (__int64 *)&v289;
      v168 = (__int64 *)&v290;
    }
    else
    {
      while ( 1 )
      {
        v26 = *(unsigned __int8 *)(v24 + 8);
        if ( (_BYTE)v26 != 14 && (v26 != 16 && v26 != 13 || *(_BYTE *)(v24 + 8) == 15) || !*(_DWORD *)(v24 + 12) )
        {
          v16 = *((_BYTE *)v15 + 8);
          goto LABEL_15;
        }
        a2 = 0;
        v24 = sub_1643D80(v24, 0);
        if ( (__int64 *)v24 == v25 )
          break;
        ++v23;
      }
      v160 = (unsigned int)(v23 + 2);
      v161 = (__int64 *)&v289;
      v162 = v160;
      v163 = v160;
      v164 = sub_1643350(*(_QWORD **)(*(_QWORD *)(a1 + 8) + 24LL));
      v167 = sub_159C470(v164, 0, 0);
      v287 = &v289;
      v288 = 0x800000000LL;
      if ( v160 > 8 )
      {
        sub_16CD150((__int64)&v287, &v289, v160, 8, v165, v166);
        v161 = (__int64 *)v287;
      }
      v168 = &v161[v163];
      LODWORD(v288) = v160;
      if ( &v161[v163] == v161 )
      {
LABEL_192:
        LOWORD(v286[0]) = 257;
        v169 = v162 + 1;
        v170 = *(_QWORD *)v273;
        if ( *(_BYTE *)(*(_QWORD *)v273 + 8LL) == 16 )
          v170 = **(_QWORD **)(v170 + 16);
        v171 = *(_QWORD *)(v170 + 24);
        v172 = sub_1648A60(72, v169);
        v29 = (__int64)v172;
        if ( v172 )
        {
          v272 = (__int64)&v172[-3 * v169];
          v173 = *(_QWORD *)v273;
          if ( *(_BYTE *)(*(_QWORD *)v273 + 8LL) == 16 )
            v173 = **(_QWORD **)(v173 + 16);
          v270 = *(_DWORD *)(v173 + 8) >> 8;
          v174 = (__int64 *)sub_15F9F50(v171, (__int64)v161, v160);
          v175 = (__int64 *)sub_1646BA0(v174, v270);
          if ( *(_BYTE *)(*(_QWORD *)v273 + 8LL) == 16 )
          {
            v175 = sub_16463B0(v175, *(_QWORD *)(*(_QWORD *)v273 + 32LL));
          }
          else
          {
            v176 = &v161[v160];
            if ( v176 != v161 )
            {
              v177 = v161;
              while ( 1 )
              {
                v178 = *(_QWORD *)*v177;
                if ( *(_BYTE *)(v178 + 8) == 16 )
                  break;
                if ( v176 == ++v177 )
                  goto LABEL_203;
              }
              v175 = sub_16463B0(v175, *(_QWORD *)(v178 + 32));
            }
          }
LABEL_203:
          sub_15F1EA0(v29, (__int64)v175, 32, v272, v169, 0);
          *(_QWORD *)(v29 + 56) = v171;
          *(_QWORD *)(v29 + 64) = sub_15F9F50(v171, (__int64)v161, v160);
          sub_15F9CE0(v29, (__int64)v273, v161, v160, (__int64)&v284);
        }
        sub_15FA2E0(v29, 1);
        v139 = (unsigned __int64)v287;
        if ( v287 == &v289 )
          return v29;
        goto LABEL_158;
      }
    }
    do
      *v161++ = v167;
    while ( v161 != v168 );
    v160 = (unsigned int)v288;
    v161 = (__int64 *)v287;
    v162 = v288;
    goto LABEL_192;
  }
LABEL_15:
  v27 = *(unsigned __int8 *)(v271 + 8);
  if ( v16 != 16 )
  {
LABEL_18:
    if ( (_BYTE)v27 != 16 )
      goto LABEL_20;
    goto LABEL_19;
  }
  if ( v15[4] != (__int64 *)1 )
  {
    if ( (_BYTE)v27 != 11 )
      goto LABEL_18;
    if ( (unsigned __int8)(v273[16] - 60) > 1u )
      goto LABEL_41;
    v124 = &v289;
    v125 = *((_QWORD *)v273 - 3);
    if ( *(_BYTE *)(v125 + 16) != 71 )
      goto LABEL_41;
    v126 = *(unsigned __int8 **)(v125 - 24);
    v127 = *(__int64 ***)v126;
    if ( *(_BYTE *)(*(_QWORD *)v126 + 8LL) != 16 )
      goto LABEL_41;
    v128 = (__int64)v127[3];
    v129 = v15[3];
    if ( (__int64 *)v128 != v129 )
    {
      v260 = sub_1643030(v128);
      if ( v260 != (unsigned int)sub_1643030((__int64)v129) )
      {
LABEL_41:
        v39 = *(_QWORD *)v14;
        v40 = &v289;
        v41 = *(_QWORD *)(v14 - 24);
        v42 = *(_QWORD *)(*(_QWORD *)v14 + 32LL);
        v287 = &v289;
        v288 = 0x800000000LL;
        if ( v42 > 8 )
        {
          v264 = v39;
          sub_16CD150((__int64)&v287, &v289, v42, 8, v39, a14);
          v40 = v287;
          v39 = v264;
        }
        LODWORD(v288) = v42;
        if ( 8LL * (unsigned int)v42 )
        {
          v254 = v39;
          memset(v40, 0, 8LL * (unsigned int)v42);
          v39 = v254;
        }
        a2 = 0;
        if ( (unsigned __int8)sub_1749E50(
                                v41,
                                0,
                                (__int64)&v287,
                                *(_QWORD *)(v39 + 24),
                                **(_BYTE **)(a1 + 2664),
                                *(double *)a3.m128_u64,
                                a4,
                                a5) )
        {
          v45 = sub_15A06D0(*(__int64 ***)v14, 0, v43, v44);
          v46 = v287;
          v47 = v45;
          if ( (_DWORD)v288 )
          {
            v255 = (unsigned int)v288;
            v48 = 0;
            v241 = v15;
            v240 = v14;
            do
            {
              if ( *(_QWORD *)&v46[4 * v48] )
              {
                v281 = 257;
                v51 = *(_QWORD *)(a1 + 8);
                v52 = sub_1643350(*(_QWORD **)(v51 + 24));
                v53 = sub_159C470(v52, v48, 0);
                v54 = *(_QWORD *)&v287[4 * v48];
                if ( *(_BYTE *)(v47 + 16) > 0x10u || *(_BYTE *)(v54 + 16) > 0x10u || *(_BYTE *)(v53 + 16) > 0x10u )
                {
                  v246 = v53;
                  v283 = 257;
                  v55 = sub_1648A60(56, 3u);
                  v56 = v55;
                  if ( v55 )
                    sub_15FA480((__int64)v55, (__int64 *)v47, v54, v246, (__int64)v282, 0);
                  v57 = *(_QWORD *)(v51 + 8);
                  if ( v57 )
                  {
                    v58 = *(unsigned __int64 **)(v51 + 16);
                    sub_157E9D0(v57 + 40, (__int64)v56);
                    v59 = v56[3];
                    v60 = *v58;
                    v56[4] = v58;
                    v60 &= 0xFFFFFFFFFFFFFFF8LL;
                    v56[3] = v60 | v59 & 7;
                    *(_QWORD *)(v60 + 8) = v56 + 3;
                    *v58 = *v58 & 7 | (unsigned __int64)(v56 + 3);
                  }
                  v61 = v280;
                  v62 = v56;
                  sub_164B780((__int64)v56, v280);
                  v64 = *(_QWORD *)(v51 + 80) == 0;
                  v276 = v56;
                  if ( v64 )
                    goto LABEL_317;
                  (*(void (__fastcall **)(__int64, _QWORD **))(v51 + 88))(v51 + 64, &v276);
                  a2 = *(_QWORD *)v51;
                  if ( *(_QWORD *)v51 )
                  {
                    v284 = *(__int64 **)v51;
                    sub_1623A60((__int64)&v284, a2, 2);
                    v65 = v56[6];
                    if ( v65 )
                      sub_161E7C0((__int64)(v56 + 6), v65);
                    a2 = (__int64)v284;
                    v56[6] = v284;
                    if ( a2 )
                      sub_1623210((__int64)&v284, (unsigned __int8 *)a2, (__int64)(v56 + 6));
                  }
                  v47 = (__int64)v56;
                }
                else
                {
                  v49 = sub_15A3890((__int64 *)v47, *(_QWORD *)&v287[4 * v48], v53, 0);
                  a2 = *(_QWORD *)(v51 + 96);
                  v47 = v49;
                  v50 = sub_14DBA30(v49, a2, 0);
                  if ( v50 )
                    v47 = v50;
                }
                v46 = v287;
              }
              ++v48;
            }
            while ( v48 != v255 );
            v15 = v241;
            v14 = v240;
          }
          if ( v46 != &v289 )
            _libc_free((unsigned __int64)v46);
          if ( v47 )
          {
            v149 = *(_QWORD *)(v14 + 8);
            v29 = v14;
            if ( v149 )
            {
              v150 = *(_QWORD *)a1;
              do
              {
                v151 = sub_1648700(v149);
                sub_170B990(v150, (__int64)v151);
                v149 = *(_QWORD *)(v149 + 8);
              }
              while ( v149 );
              if ( v14 == v47 )
                v47 = sub_1599EF0(*(__int64 ***)v14);
              sub_164D160(v14, v47, a3, a4, a5, a6, v152, v153, a9, a10);
              return v29;
            }
            return 0;
          }
        }
        else if ( v287 != &v289 )
        {
          _libc_free((unsigned __int64)v287);
        }
        LOBYTE(v27) = *(_BYTE *)(v271 + 8);
        goto LABEL_18;
      }
      a2 = 47;
      v127 = (__int64 **)sub_16463B0(v129, (unsigned int)v127[4]);
      v289 = 257;
      v126 = sub_1708970(*(_QWORD *)(a1 + 8), 47, (__int64)v126, v127, (__int64 *)&v287);
    }
    v287 = &v289;
    v288 = 0x1000000000LL;
    if ( v127[4] <= v15[4] )
    {
      v221 = sub_15A06D0(v127, a2, v27, (__int64)v124);
      v222 = v127[4];
      v130 = v221;
      v223 = (int)v222;
      if ( (_DWORD)v222 )
      {
        v224 = 0;
        v225 = (unsigned int)v288;
        v265 = v14;
        v226 = v130;
        v227 = 0;
        do
        {
          if ( HIDWORD(v288) <= (unsigned int)v225 )
          {
            v245 = v222;
            sub_16CD150((__int64)&v287, &v289, 0, 4, (int)v222, v224);
            v225 = (unsigned int)v288;
            v222 = v245;
          }
          *(_DWORD *)&v287[2 * v225] = v227++;
          v225 = (unsigned int)(v288 + 1);
          LODWORD(v288) = v288 + 1;
        }
        while ( v223 != v227 );
        v130 = v226;
        v14 = v265;
      }
      v228 = (unsigned int)v288;
      v229 = *((_DWORD *)v15 + 8) - (_DWORD)v222;
      if ( !v229 )
        goto LABEL_156;
      v230 = 0;
      v261 = v130;
      v250 = v14;
      v231 = 0;
      v232 = v229;
      do
      {
        if ( HIDWORD(v288) <= (unsigned int)v228 )
        {
          sub_16CD150((__int64)&v287, &v289, 0, 4, v230, v229);
          v228 = (unsigned int)v288;
        }
        ++v231;
        *(_DWORD *)&v287[2 * v228] = v223;
        v228 = (unsigned int)(v288 + 1);
        LODWORD(v288) = v288 + 1;
      }
      while ( v232 != v231 );
    }
    else
    {
      v130 = sub_1599EF0(v127);
      v132 = (unsigned int)v15[4];
      if ( !v132 )
      {
LABEL_156:
        v136 = (unsigned int)v288;
        v262 = (char *)v287;
        v137 = sub_16498A0(v130);
        LOWORD(v286[0]) = 257;
        v263 = (_QWORD *)sub_1599580(v137, v262, v136);
        v138 = sub_1648A60(56, 3u);
        v29 = (__int64)v138;
        if ( v138 )
        {
          sub_15FA660((__int64)v138, v126, v130, v263, (__int64)&v284, 0);
          v139 = (unsigned __int64)v287;
          if ( v287 == &v289 )
            return v29;
LABEL_158:
          _libc_free(v139);
          return v29;
        }
        if ( v287 != &v289 )
          _libc_free((unsigned __int64)v287);
        goto LABEL_41;
      }
      v133 = (unsigned int)v288;
      v134 = 0;
      v261 = v130;
      v250 = v14;
      v135 = (unsigned int)v15[4];
      do
      {
        if ( HIDWORD(v288) <= (unsigned int)v133 )
        {
          sub_16CD150((__int64)&v287, &v289, 0, 4, v132, v131);
          v133 = (unsigned int)v288;
        }
        *(_DWORD *)&v287[2 * v133] = v134++;
        v133 = (unsigned int)(v288 + 1);
        LODWORD(v288) = v288 + 1;
      }
      while ( v135 != v134 );
    }
    v130 = v261;
    v14 = v250;
    goto LABEL_156;
  }
  if ( (_BYTE)v27 != 16 )
  {
    v289 = 257;
    v73 = sub_1708970(*(_QWORD *)(a1 + 8), 47, (__int64)v273, (__int64 **)v15[3], (__int64 *)&v287);
    v289 = 257;
    v74 = (__int64)v73;
    v75 = (_QWORD *)sub_16498A0(v14);
    v76 = (__int64 **)sub_1643350(v75);
    v274 = sub_15A06D0(v76, 47, v77, v78);
    v79 = (__int64 *)sub_1599EF0(v15);
    v80 = sub_1648A60(56, 3u);
    v29 = (__int64)v80;
    if ( v80 )
      sub_15FA480((__int64)v80, v79, v74, v274, (__int64)&v287, 0);
    return v29;
  }
LABEL_19:
  if ( *(_QWORD *)(v271 + 32) == 1 )
  {
    if ( *((_BYTE *)v15 + 8) != 16 )
    {
      v279 = 257;
      v140 = *(_QWORD *)(a1 + 8);
      v141 = (_QWORD *)sub_16498A0(v14);
      v142 = (__int64 **)sub_1643350(v141);
      v145 = sub_15A06D0(v142, a2, v143, v144);
      v146 = v145;
      if ( v273[16] > 0x10u || *(_BYTE *)(v145 + 16) > 0x10u )
      {
        LOWORD(v286[0]) = 257;
        v196 = sub_1648A60(56, 2u);
        v147 = v196;
        if ( v196 )
          sub_15FA320((__int64)v196, v273, v146, (__int64)&v284, 0);
        v197 = *(_QWORD *)(v140 + 8);
        if ( v197 )
        {
          v198 = *(unsigned __int64 **)(v140 + 16);
          sub_157E9D0(v197 + 40, (__int64)v147);
          v199 = v147[3];
          v200 = *v198;
          v147[4] = v198;
          v200 &= 0xFFFFFFFFFFFFFFF8LL;
          v147[3] = v200 | v199 & 7;
          *(_QWORD *)(v200 + 8) = v147 + 3;
          *v198 = *v198 & 7 | (unsigned __int64)(v147 + 3);
        }
        v61 = v278;
        v62 = v147;
        sub_164B780((__int64)v147, v278);
        v64 = *(_QWORD *)(v140 + 80) == 0;
        v277 = v147;
        if ( v64 )
LABEL_317:
          sub_4263D6(v62, v61, v63);
        (*(void (__fastcall **)(__int64, _QWORD **))(v140 + 88))(v140 + 64, &v277);
        v201 = *(_QWORD *)v140;
        if ( *(_QWORD *)v140 )
        {
          v287 = *(__int16 **)v140;
          sub_1623A60((__int64)&v287, v201, 2);
          v202 = v147[6];
          if ( v202 )
            sub_161E7C0((__int64)(v147 + 6), v202);
          v203 = (unsigned __int8 *)v287;
          v147[6] = v287;
          if ( v203 )
            sub_1623210((__int64)&v287, v203, (__int64)(v147 + 6));
        }
      }
      else
      {
        v147 = (_QWORD *)sub_15A37D0(v273, v145, 0);
        v148 = sub_14DBA30((__int64)v147, *(_QWORD *)(v140 + 96), 0);
        if ( v148 )
          v147 = (_QWORD *)v148;
      }
      v289 = 257;
      return sub_15FDBD0(47, (__int64)v147, (__int64)v15, (__int64)&v287, 0);
    }
    v66 = *(_QWORD *)(v14 - 24);
    if ( *(_BYTE *)(v66 + 16) == 84 )
    {
      v289 = 257;
      return sub_15FDBD0(47, *(_QWORD *)(v66 - 48), (__int64)v15, (__int64)&v287, 0);
    }
  }
LABEL_20:
  v28 = v273[16];
  if ( v28 != 85 )
    goto LABEL_21;
  v81 = *((_QWORD *)v273 + 1);
  if ( v81 )
  {
    if ( !*(_QWORD *)(v81 + 8) && *((_BYTE *)v15 + 8) == 16 )
    {
      v82 = *(_QWORD *)(*(_QWORD *)v273 + 32LL);
      if ( v82 == *((_DWORD *)v15 + 8) )
      {
        v83 = (__int64 ****)*((_QWORD *)v273 - 9);
        if ( v82 == *((_DWORD *)*v83 + 8) )
        {
          if ( *((_BYTE *)v83 + 16) == 71 && v15 == **(v83 - 3)
            || (v84 = *((_QWORD *)v273 - 6), *(_BYTE *)(v84 + 16) == 71) && v15 == **(__int64 ****)(v84 - 24) )
          {
            v289 = 257;
            v85 = sub_1708970(*(_QWORD *)(a1 + 8), 47, *((_QWORD *)v273 - 9), v15, (__int64 *)&v287);
            v86 = *(_QWORD *)(a1 + 8);
            v289 = 257;
            v87 = v85;
            v88 = sub_1708970(v86, 47, *((_QWORD *)v273 - 6), v15, (__int64 *)&v287);
            v89 = (_QWORD *)*((_QWORD *)v273 - 3);
            v90 = (__int64)v88;
            v289 = 257;
            v91 = sub_1648A60(56, 3u);
            v29 = (__int64)v91;
            if ( v91 )
              sub_15FA660((__int64)v91, v87, v90, v89, (__int64)&v287, 0);
            return v29;
          }
        }
      }
    }
  }
  v92 = *(_QWORD *)(a1 + 8);
  if ( !sub_1642F90(*(_QWORD *)v14, 32) )
    goto LABEL_130;
  v93 = *(__int64 **)(v14 - 24);
  if ( *((_BYTE *)v93 + 16) != 85 )
    goto LABEL_130;
  v287 = &v289;
  v288 = 0x1000000000LL;
  sub_15FAA20((unsigned __int8 *)*(v93 - 3), (__int64)&v287);
  if ( (_DWORD)v288 != 4 )
    goto LABEL_108;
  v94 = v93[1];
  if ( !v94 )
    goto LABEL_108;
  if ( *(_QWORD *)(v94 + 8) )
    goto LABEL_108;
  if ( *(_DWORD *)(*v93 + 32) != 4 )
    goto LABEL_108;
  v256 = *v93;
  if ( 4 * (unsigned int)sub_1643030(*(_QWORD *)(*v93 + 24)) != 32 )
    goto LABEL_108;
  v95 = *(v93 - 6);
  v96 = *(v93 - 9);
  v97 = v256;
  v98 = *(_BYTE *)(v96 + 16);
  if ( v95 )
  {
    if ( v98 == 9 || v96 == v95 )
      goto LABEL_108;
  }
  else if ( v98 == 9 )
  {
    goto LABEL_108;
  }
  if ( v98 != 54
    || (v251 = v256, v267 = *(v93 - 9), v235 = sub_15F32D0(v267), v96 = v267, v97 = v251, v235)
    || (*(_BYTE *)(v267 + 18) & 1) != 0 )
  {
    if ( *(_BYTE *)(v96 + 16) != 71 )
      goto LABEL_108;
    v99 = *(_QWORD *)(v96 + 8);
    if ( !v99 )
      goto LABEL_108;
    goto LABEL_111;
  }
  v99 = *(_QWORD *)(v267 + 8);
  if ( !v99 )
    goto LABEL_108;
  if ( *(_QWORD *)(v99 + 8) )
  {
    if ( *(_BYTE *)(v267 + 16) != 71 )
      goto LABEL_108;
LABEL_111:
    if ( *(_QWORD *)(v99 + 8) )
      goto LABEL_108;
    goto LABEL_112;
  }
  if ( v251 != *(_QWORD *)v267 )
  {
    if ( *(_BYTE *)(v267 + 16) != 71 )
      goto LABEL_108;
LABEL_112:
    v247 = v97;
    v257 = v96;
    if ( !sub_1642F90(**(_QWORD **)(v96 - 24), 32) )
      goto LABEL_108;
    v97 = v247;
    v96 = *(_QWORD *)(v257 - 24);
    if ( !v96 )
      goto LABEL_108;
  }
  if ( *(_BYTE *)(v95 + 16) == 54 )
  {
    v252 = v96;
    v268 = v97;
    v236 = sub_15F32D0(v95);
    v96 = v252;
    if ( !v236 && (*(_BYTE *)(v95 + 18) & 1) == 0 )
    {
      v237 = *(_QWORD *)(v95 + 8);
      if ( v237 )
      {
        if ( !*(_QWORD *)(v237 + 8) && v268 == *(_QWORD *)v95 )
          goto LABEL_121;
      }
    }
  }
  v100 = *(_BYTE *)(v95 + 16);
  if ( v100 <= 0x17u )
  {
    if ( v100 <= 0x10u )
    {
      v266 = v96;
      if ( v100 == 9 )
      {
        v238 = (_QWORD *)sub_16498A0(v14);
        v239 = sub_1643350(v238);
        v234 = sub_159C470(v239, 0, 0);
      }
      else
      {
        LOWORD(v286[0]) = 257;
        v233 = (__int64 **)sub_1643350(*(_QWORD **)(v92 + 24));
        v234 = (__int64)sub_1708970(v92, 47, v95, v233, (__int64 *)&v284);
      }
      v96 = v266;
      v95 = v234;
      if ( v234 )
      {
LABEL_121:
        LOWORD(v286[0]) = 257;
        v248 = v96;
        v102 = (__int64 **)sub_1643350(*(_QWORD **)(v92 + 24));
        v103 = sub_1708970(v92, 47, v248, v102, (__int64 *)&v284);
        LOWORD(v286[0]) = 257;
        v243 = v103;
        v104 = (__int64 **)sub_1643350(*(_QWORD **)(v92 + 24));
        v105 = sub_1708970(v92, 47, v95, v104, (__int64 *)&v284);
        v106 = ((unsigned __int16)*((_DWORD *)v287 + 3) << 12) & 0x7000
             | *(_DWORD *)v287 & 7
             | ((unsigned __int16)*((_DWORD *)v287 + 2) << 8) & 0x700
             | (16 * (unsigned __int8)*((_DWORD *)v287 + 1)) & 0x70;
        v107 = sub_15E26F0(*(__int64 **)(*(_QWORD *)(*(_QWORD *)(v14 + 40) + 56LL) + 40LL), 4227, 0, 0);
        v286[1] = v105;
        v108 = v107;
        v284 = v286;
        v286[0] = v243;
        v285 = 0x300000002LL;
        v109 = (_QWORD *)sub_16498A0(v14);
        v110 = sub_1643350(v109);
        v111 = sub_159C470(v110, v106, 0);
        if ( (unsigned int)v285 >= HIDWORD(v285) )
          sub_16CD150((__int64)&v284, v286, 0, 8, (int)&v284, v112);
        v284[(unsigned int)v285] = v111;
        v282[0] = (__int64)"prmtCall";
        v113 = (unsigned int)(v285 + 1);
        v114 = v285 + 2;
        LODWORD(v285) = v285 + 1;
        v283 = 259;
        v249 = v284;
        v242 = v114;
        v244 = *(_QWORD *)(*(_QWORD *)v108 + 24LL);
        v115 = sub_1648AB0(72, v114, 0);
        v29 = (__int64)v115;
        if ( v115 )
        {
          sub_15F1EA0((__int64)v115, **(_QWORD **)(v244 + 16), 54, (__int64)&v115[-3 * v113 - 3], v242, 0);
          *(_QWORD *)(v29 + 56) = 0;
          sub_15F5B40(v29, v244, v108, v249, v113, (__int64)v282, 0, 0);
        }
        if ( v284 != v286 )
          _libc_free((unsigned __int64)v284);
        if ( v287 != &v289 )
          _libc_free((unsigned __int64)v287);
        if ( v29 )
          return v29;
        goto LABEL_130;
      }
    }
  }
  else if ( v100 == 71 )
  {
    v101 = *(_QWORD *)(v95 + 8);
    v258 = v96;
    if ( v101 )
    {
      if ( !*(_QWORD *)(v101 + 8) && sub_1642F90(**(_QWORD **)(v95 - 24), 32) )
      {
        v95 = *(_QWORD *)(v95 - 24);
        v96 = v258;
        if ( v95 )
          goto LABEL_121;
      }
    }
  }
LABEL_108:
  if ( v287 != &v289 )
  {
    _libc_free((unsigned __int64)v287);
    v28 = v273[16];
    goto LABEL_21;
  }
LABEL_130:
  v28 = v273[16];
LABEL_21:
  if ( v28 != 78 )
  {
LABEL_22:
    if ( v28 == 77 )
    {
      v29 = sub_1753C70(a1, (__int64 *)v14, (__int64)v273, a3, a4, a5, a6, a7, a8, a9, a10);
      if ( v29 )
        return v29;
    }
    goto LABEL_26;
  }
  v31 = *((_QWORD *)v273 - 3);
  if ( !*(_BYTE *)(v31 + 16) && (*(_BYTE *)(v31 + 33) & 0x20) != 0 && *(_DWORD *)(v31 + 36) == 4086 )
  {
    v123 = *((unsigned __int8 *)v15 + 8);
    if ( (_BYTE)v123 == 16 )
    {
      if ( *(_BYTE *)(*v15[2] + 8) != 15 && *((_DWORD *)v15 + 8) <= 4u )
      {
        if ( sub_12BE0A0(*(_QWORD *)(a1 + 2664), (__int64)v15) <= 0x10 )
          goto LABEL_236;
        v123 = *((unsigned __int8 *)v15 + 8);
LABEL_142:
        if ( (unsigned __int8)v123 > 0x10u
          || (v191 = 100990, !_bittest64(&v191, v123))
          || sub_12BE0A0(*(_QWORD *)(a1 + 2664), (__int64)v15) > 8 )
        {
          v28 = v273[16];
          goto LABEL_22;
        }
LABEL_236:
        v282[0] = (__int64)v15;
        v282[1] = **(_QWORD **)&v273[24 * (1LL - (*((_DWORD *)v273 + 5) & 0xFFFFFFF))];
        v192 = sub_15E26F0(*(__int64 **)(*(_QWORD *)(*(_QWORD *)(v14 + 40) + 56LL) + 40LL), 4086, v282, 2);
        v193 = *((_DWORD *)v273 + 5) & 0xFFFFFFF;
        v284 = *(__int64 **)&v273[-24 * v193];
        v285 = *(_QWORD *)&v273[24 * (1 - v193)];
        v286[0] = *(_QWORD *)&v273[24 * (2 - v193)];
        v289 = 257;
        v194 = *(_QWORD *)(*(_QWORD *)v192 + 24LL);
        v195 = sub_1648AB0(72, 4u, 0);
        v29 = (__int64)v195;
        if ( v195 )
        {
          sub_15F1EA0((__int64)v195, **(_QWORD **)(v194 + 16), 54, (__int64)(v195 - 12), 4, 0);
          *(_QWORD *)(v29 + 56) = 0;
          sub_15F5B40(v29, v194, v192, (__int64 *)&v284, 3, (__int64)&v287, 0, 0);
        }
        return v29;
      }
    }
    else if ( (_BYTE)v123 != 15 )
    {
      goto LABEL_142;
    }
  }
LABEL_26:
  v32 = *(_QWORD *)(v14 - 24);
  v33 = *(_QWORD *)(v32 + 8);
  if ( *(_BYTE *)(v32 + 16) == 83 && v33 && !*(_QWORD *)(v33 + 8) )
  {
    v116 = *(__int64 ***)v14;
    if ( (unsigned __int8)sub_1643F10(*(_QWORD *)v14) )
    {
      v117 = (__int64 **)sub_16463B0((__int64 *)v116, *(_QWORD *)(**(_QWORD **)(v32 - 48) + 32LL));
      v289 = 259;
      v118 = *(_QWORD *)(a1 + 8);
      v287 = (__int16 *)"bc";
      v119 = sub_1708970(v118, 47, *(_QWORD *)(v32 - 48), v117, (__int64 *)&v287);
      v289 = 257;
      v120 = v119;
      v121 = *(_QWORD *)(v32 - 24);
      v122 = sub_1648A60(56, 2u);
      v29 = (__int64)v122;
      if ( v122 )
      {
        sub_15FA320((__int64)v122, v120, v121, (__int64)&v287, 0);
        return v29;
      }
    }
    v32 = *(_QWORD *)(v14 - 24);
    v33 = *(_QWORD *)(v32 + 8);
  }
  v34 = *(__int64 ***)v14;
  v35 = *(_BYTE *)(*(_QWORD *)v14 + 8LL);
  v36 = *(_QWORD *)(a1 + 8);
  v37 = v35;
  if ( v35 == 16 )
    v37 = *(_BYTE *)(*v34[2] + 8);
  if ( v37 == 11 )
  {
    if ( !v33 )
      goto LABEL_34;
    if ( !*(_QWORD *)(v33 + 8) && (unsigned __int8)(*(_BYTE *)(v32 + 16) - 50) <= 2u )
    {
      if ( v35 != 16 || *(_BYTE *)(*(_QWORD *)v32 + 8LL) != 16 )
        goto LABEL_211;
      v154 = *(_QWORD *)(v32 - 48);
      v155 = *(_QWORD *)(v154 + 8);
      if ( v155 && !*(_QWORD *)(v155 + 8) )
      {
        v205 = *(unsigned __int8 *)(v154 + 16);
        if ( (unsigned __int8)v205 > 0x17u )
        {
          v206 = v205 - 24;
        }
        else
        {
          if ( (_BYTE)v205 != 5 )
            goto LABEL_178;
          v206 = *(unsigned __int16 *)(v154 + 18);
        }
        if ( v206 == 47 )
        {
          v207 = (*(_BYTE *)(v154 + 23) & 0x40) != 0
               ? *(__int64 **)(v154 - 8)
               : (__int64 *)(v154 - 24LL * (*(_DWORD *)(v154 + 20) & 0xFFFFFFF));
          v208 = *v207;
          if ( *v207 )
          {
            if ( v34 == *(__int64 ***)v208 && *(_BYTE *)(v208 + 16) > 0x10u )
            {
              v209 = *(__int64 ***)v14;
              v289 = 257;
              v210 = sub_1708970(v36, 47, *(_QWORD *)(v32 - 24), v209, (__int64 *)&v287);
              v289 = 257;
              v29 = sub_15FB440(
                      (unsigned int)*(unsigned __int8 *)(v32 + 16) - 24,
                      (__int64 *)v208,
                      (__int64)v210,
                      (__int64)&v287,
                      0);
              goto LABEL_182;
            }
          }
        }
      }
LABEL_178:
      v156 = *(_QWORD *)(v32 - 24);
      v157 = *(_QWORD *)(v156 + 8);
      v158 = *(_BYTE *)(v156 + 16);
      if ( !v157 || *(_QWORD *)(v157 + 8) )
      {
LABEL_180:
        if ( v158 > 0x10u )
          goto LABEL_211;
        goto LABEL_181;
      }
      if ( v158 > 0x17u )
      {
        if ( v158 != 71 )
          goto LABEL_211;
      }
      else
      {
        if ( v158 != 5 )
          goto LABEL_180;
        if ( *(_WORD *)(v156 + 18) != 47 )
        {
LABEL_181:
          v289 = 257;
          v275 = (__int64 *)sub_1708970(v36, 47, *(_QWORD *)(v32 - 48), v34, (__int64 *)&v287);
          v159 = sub_15A4510((__int64 ***)v156, v34, 0);
          v289 = 257;
          v29 = sub_15FB440((unsigned int)*(unsigned __int8 *)(v32 + 16) - 24, v275, v159, (__int64)&v287, 0);
LABEL_182:
          if ( v29 )
            return v29;
          v32 = *(_QWORD *)(v14 - 24);
          v36 = *(_QWORD *)(a1 + 8);
          v33 = *(_QWORD *)(v32 + 8);
          goto LABEL_33;
        }
      }
      if ( (*(_BYTE *)(v156 + 23) & 0x40) != 0 )
        v211 = *(__int64 **)(v156 - 8);
      else
        v211 = (__int64 *)(v156 - 24LL * (*(_DWORD *)(v156 + 20) & 0xFFFFFFF));
      v212 = *v211;
      if ( *v211 && v34 == *(__int64 ***)v212 && *(_BYTE *)(v212 + 16) > 0x10u )
      {
        v213 = *(__int64 ***)v14;
        v289 = 257;
        v214 = (__int64 *)sub_1708970(v36, 47, *(_QWORD *)(v32 - 48), v213, (__int64 *)&v287);
        v289 = 257;
        v29 = sub_15FB440((unsigned int)*(unsigned __int8 *)(v32 + 16) - 24, v214, v212, (__int64)&v287, 0);
        goto LABEL_182;
      }
      goto LABEL_180;
    }
  }
LABEL_33:
  if ( !v33 )
    goto LABEL_34;
LABEL_211:
  if ( *(_QWORD *)(v33 + 8) )
    goto LABEL_34;
  if ( *(_BYTE *)(v32 + 16) != 79 )
    goto LABEL_34;
  v179 = *(_QWORD *)(v32 - 72);
  if ( !v179 )
    goto LABEL_34;
  v180 = *(_QWORD *)(v32 - 48);
  if ( !v180 )
    goto LABEL_34;
  v181 = *(_QWORD *)(v32 - 24);
  if ( !v181 )
    goto LABEL_34;
  v182 = *(__int64 ***)v14;
  v183 = *(_BYTE *)(*(_QWORD *)v14 + 8LL);
  if ( *(_BYTE *)(*(_QWORD *)v179 + 8LL) == 16
    && (v183 != 16 || *((_DWORD *)v182 + 8) != *(_DWORD *)(*(_QWORD *)v179 + 32LL)) )
  {
    goto LABEL_34;
  }
  if ( (v183 == 16) != (*(_BYTE *)(*(_QWORD *)v180 + 8LL) == 16) )
    goto LABEL_34;
  v184 = *(_QWORD *)(v180 + 8);
  if ( !v184 || *(_QWORD *)(v184 + 8) )
    goto LABEL_220;
  v215 = *(unsigned __int8 *)(v180 + 16);
  if ( (unsigned __int8)v215 > 0x17u )
  {
    v216 = v215 - 24;
  }
  else
  {
    if ( (_BYTE)v215 != 5 )
      goto LABEL_220;
    v216 = *(unsigned __int16 *)(v180 + 18);
  }
  if ( v216 == 47 )
  {
    v217 = (*(_BYTE *)(v180 + 23) & 0x40) != 0
         ? *(__int64 **)(v180 - 8)
         : (__int64 *)(v180 - 24LL * (*(_DWORD *)(v180 + 20) & 0xFFFFFFF));
    v218 = *v217;
    if ( *v217 )
    {
      if ( v182 == *(__int64 ***)v218 && *(_BYTE *)(v218 + 16) > 0x10u )
      {
        v219 = *(_QWORD *)(v32 - 24);
        v289 = 257;
        v220 = sub_1708970(v36, 47, v219, v182, (__int64 *)&v287);
        v289 = 257;
        v29 = sub_14EDD70(v179, (_QWORD *)v218, (__int64)v220, (__int64)&v287, 0, v32);
        goto LABEL_232;
      }
    }
  }
LABEL_220:
  v185 = *(_QWORD *)(v181 + 8);
  if ( !v185 || *(_QWORD *)(v185 + 8) )
    goto LABEL_34;
  v186 = *(unsigned __int8 *)(v181 + 16);
  if ( (unsigned __int8)v186 > 0x17u )
  {
    v187 = v186 - 24;
  }
  else
  {
    if ( (_BYTE)v186 != 5 )
      goto LABEL_34;
    v187 = *(unsigned __int16 *)(v181 + 18);
  }
  if ( v187 != 47 )
    goto LABEL_34;
  v188 = (*(_BYTE *)(v181 + 23) & 0x40) != 0
       ? *(__int64 **)(v181 - 8)
       : (__int64 *)(v181 - 24LL * (*(_DWORD *)(v181 + 20) & 0xFFFFFFF));
  v189 = *v188;
  if ( !*v188 || v182 != *(__int64 ***)v189 || *(_BYTE *)(v189 + 16) <= 0x10u )
    goto LABEL_34;
  v289 = 257;
  v190 = sub_1708970(v36, 47, v180, v182, (__int64 *)&v287);
  v289 = 257;
  v29 = sub_14EDD70(v179, v190, v189, (__int64)&v287, 0, v32);
LABEL_232:
  if ( v29 )
    return v29;
LABEL_34:
  if ( *(_BYTE *)(v271 + 8) == 15 )
    return sub_174C560((__int64 *)a1, v14, a3, a4, a5, a6, a7, a8, a9, a10);
  else
    return sub_174B490((__int64 *)a1, v14, a3, a4, a5, a6, a7, a8, a9, a10);
}
