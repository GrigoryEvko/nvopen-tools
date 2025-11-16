// Function: sub_177D4A0
// Address: 0x177d4a0
//
__int64 __fastcall sub_177D4A0(
        __int64 *a1,
        __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r12
  bool v11; // zf
  __int64 v12; // rcx
  __int64 v13; // rax
  double v14; // xmm4_8
  double v15; // xmm5_8
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rcx
  unsigned __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v22; // r15
  unsigned __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rbx
  __int64 v26; // r14
  unsigned __int64 v27; // r13
  __int64 v28; // rdx
  _QWORD *v29; // rsi
  __int64 *v30; // r14
  __int64 v31; // rbx
  __int64 v32; // rdx
  unsigned int v33; // r15d
  _QWORD *v34; // rax
  _QWORD *v35; // r13
  __int64 v36; // rdi
  unsigned __int64 *v37; // r14
  __int64 v38; // rax
  unsigned __int64 v39; // rcx
  _QWORD *v40; // rdi
  __int64 *v41; // rsi
  __int64 v42; // rdx
  __int64 v43; // rsi
  __int64 v44; // rsi
  unsigned __int8 *v45; // rsi
  unsigned __int64 *v46; // rbx
  char v47; // al
  __int64 v48; // rsi
  __int64 **v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // rdx
  __int64 v53; // rax
  _QWORD *v54; // rax
  _QWORD *v55; // r14
  _QWORD *v56; // rcx
  __int64 v57; // rax
  __int64 *v58; // rax
  __int64 v59; // rax
  __int64 v60; // rcx
  __int64 *v61; // r11
  __int64 v62; // rax
  unsigned __int64 v63; // rcx
  __int64 v64; // rax
  __int64 v65; // rbx
  __int64 v66; // r13
  _QWORD *v67; // rax
  double v68; // xmm4_8
  double v69; // xmm5_8
  __int64 v70; // rsi
  __int64 **v71; // rax
  __int64 v72; // r13
  __int64 v73; // rbx
  __int64 v74; // rax
  __int64 v75; // rsi
  __int64 v76; // rax
  const char **v77; // rdi
  __int64 v78; // rdx
  __int64 v79; // r13
  unsigned int v80; // eax
  __int64 v81; // r14
  unsigned __int64 v82; // r13
  int v83; // r8d
  int v84; // r9d
  __int64 v85; // r15
  unsigned __int8 v86; // al
  __int64 v87; // rcx
  __int64 v88; // rax
  __int64 v89; // rbx
  __int64 v90; // r13
  __int64 v91; // r14
  _QWORD *v92; // rax
  double v93; // xmm4_8
  double v94; // xmm5_8
  __int64 v95; // rax
  unsigned int v96; // ebx
  bool v97; // al
  unsigned __int64 v98; // rdi
  __int64 v99; // rax
  char v100; // bl
  unsigned int v101; // edx
  const char **v102; // rax
  __int64 *v103; // rax
  unsigned __int64 v104; // rbx
  unsigned __int64 v105; // r12
  __int64 v106; // rcx
  unsigned __int64 v107; // rax
  unsigned __int64 v108; // rbx
  int v109; // ecx
  char v110; // al
  unsigned __int64 v111; // rdx
  unsigned __int64 v112; // rsi
  _QWORD *v113; // r12
  __int64 v114; // rax
  __int64 v115; // rdx
  __int64 v116; // rax
  __int64 v117; // rdx
  __int64 v118; // rax
  __int64 v119; // rdx
  __int64 i; // rax
  char v121; // dl
  __int64 v122; // rax
  __int64 v123; // rdi
  __int64 v124; // rax
  __int64 v125; // rdx
  unsigned __int64 v126; // rax
  __int64 v127; // rax
  __int64 *v128; // r14
  __int64 v129; // rax
  __int64 v130; // rcx
  __int64 v131; // rsi
  __int64 v132; // rsi
  unsigned __int8 *v133; // rsi
  __int64 v134; // rdx
  unsigned __int64 v135; // rax
  __int64 v136; // rax
  unsigned __int64 v137; // r15
  const char **v138; // rax
  unsigned int v139; // eax
  __int64 v140; // rax
  __int64 v141; // rcx
  unsigned __int64 v142; // rdx
  __int64 v143; // rcx
  __int64 v144; // rax
  __int64 v145; // rdx
  __int64 v146; // rax
  __int64 v147; // rdx
  __int64 v148; // rax
  __int64 v149; // rax
  __int64 v150; // rdx
  __int64 v151; // r12
  __int64 v152; // rax
  __int64 v153; // rax
  int v154; // edx
  __int64 v155; // rcx
  unsigned int v156; // esi
  __int64 v157; // r14
  __int64 v158; // rax
  unsigned __int64 v159; // rbx
  _QWORD *v160; // rdi
  __int64 v161; // rax
  __int64 v162; // rax
  __int64 v163; // rdx
  __int64 v164; // rbx
  int v165; // ebx
  __int64 v166; // rax
  __int64 v167; // rdx
  __int64 v168; // rdx
  __int64 v169; // rax
  unsigned __int64 v170; // rdx
  _QWORD *v171; // rdi
  __int64 v172; // rax
  __int64 v173; // rax
  __int64 v174; // rdx
  __int64 v175; // r12
  __int64 v176; // rax
  __int64 v177; // rax
  __int64 v178; // rdx
  __int64 v179; // rbx
  int v180; // ebx
  __int64 v181; // rax
  __int64 v182; // rdx
  __int64 v183; // r15
  unsigned int v184; // r14d
  __int64 v185; // rax
  __int64 v186; // rbx
  unsigned __int64 v187; // r13
  __int64 v188; // rsi
  __int64 v189; // r14
  __int64 v190; // rax
  unsigned __int64 v191; // rax
  __int16 v192; // cx
  char v193; // bl
  __int64 v194; // rbx
  __int64 v195; // r12
  __int64 v196; // rsi
  __int64 v197; // rax
  __int64 v198; // rbx
  __int64 ***v199; // r13
  __int64 v200; // rax
  __int64 v201; // rax
  __int64 ***v202; // r13
  __int64 v203; // rax
  int v204; // eax
  double v205; // xmm4_8
  double v206; // xmm5_8
  __int64 v207; // rax
  __int64 v208; // rsi
  __int64 v209; // r14
  __int64 v210; // rbx
  _QWORD *v211; // rax
  double v212; // xmm4_8
  double v213; // xmm5_8
  __int64 v214; // rbx
  __int64 v215; // r14
  _QWORD *v216; // rax
  unsigned int v217; // eax
  __int64 v218; // rax
  __int64 v219; // rax
  __int64 v220; // [rsp+0h] [rbp-350h]
  unsigned __int64 v221; // [rsp+8h] [rbp-348h]
  __int64 v222; // [rsp+10h] [rbp-340h]
  __int64 v223; // [rsp+10h] [rbp-340h]
  __int64 v224; // [rsp+10h] [rbp-340h]
  unsigned __int64 v225; // [rsp+18h] [rbp-338h]
  __int64 v226; // [rsp+18h] [rbp-338h]
  unsigned __int64 v227; // [rsp+18h] [rbp-338h]
  char v228; // [rsp+20h] [rbp-330h]
  __int64 v229; // [rsp+20h] [rbp-330h]
  __int64 v230; // [rsp+20h] [rbp-330h]
  __int64 v231; // [rsp+28h] [rbp-328h]
  __int64 v232; // [rsp+28h] [rbp-328h]
  __int64 v233; // [rsp+28h] [rbp-328h]
  __int64 v234; // [rsp+28h] [rbp-328h]
  __int64 v235; // [rsp+28h] [rbp-328h]
  int v236; // [rsp+30h] [rbp-320h]
  __int64 v237; // [rsp+38h] [rbp-318h]
  __int64 v238; // [rsp+38h] [rbp-318h]
  _QWORD *v239; // [rsp+40h] [rbp-310h]
  char v240; // [rsp+40h] [rbp-310h]
  __int64 v241; // [rsp+40h] [rbp-310h]
  _QWORD *v243; // [rsp+50h] [rbp-300h] BYREF
  __int64 v244; // [rsp+58h] [rbp-2F8h] BYREF
  _QWORD v245[2]; // [rsp+60h] [rbp-2F0h] BYREF
  __int64 v246[2]; // [rsp+70h] [rbp-2E0h] BYREF
  __int16 v247; // [rsp+80h] [rbp-2D0h]
  _QWORD *v248; // [rsp+90h] [rbp-2C0h] BYREF
  __int64 v249; // [rsp+98h] [rbp-2B8h]
  __int16 v250; // [rsp+A0h] [rbp-2B0h]
  const char *v251; // [rsp+B0h] [rbp-2A0h] BYREF
  __int64 v252; // [rsp+B8h] [rbp-298h]
  _WORD v253[16]; // [rsp+C0h] [rbp-290h] BYREF
  const char **v254; // [rsp+E0h] [rbp-270h] BYREF
  __int64 v255; // [rsp+E8h] [rbp-268h]
  __int64 v256; // [rsp+F0h] [rbp-260h] BYREF
  char v257; // [rsp+F8h] [rbp-258h]
  __int64 v258; // [rsp+110h] [rbp-240h] BYREF
  __int64 v259; // [rsp+118h] [rbp-238h]
  __int64 v260; // [rsp+120h] [rbp-230h]
  int v261; // [rsp+128h] [rbp-228h]
  __int64 v262; // [rsp+130h] [rbp-220h]
  __int64 v263; // [rsp+138h] [rbp-218h]
  __int64 v264; // [rsp+140h] [rbp-210h]
  __int64 *v265; // [rsp+148h] [rbp-208h]

  v10 = (__int64)a2;
  v11 = (unsigned __int8)sub_15F8BF0((__int64)a2) == 0;
  v13 = *(a2 - 3);
  if ( v11 )
  {
    if ( !sub_1642F90(*(_QWORD *)v13, 32) )
    {
      v16 = sub_1643350(*(_QWORD **)(a1[1] + 24));
      v17 = sub_159C470(v16, 1, 0);
      if ( *(a2 - 3) )
      {
        v18 = *(a2 - 2);
        v19 = *(a2 - 1) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v19 = v18;
        if ( v18 )
          *(_QWORD *)(v18 + 16) = *(_QWORD *)(v18 + 16) & 3LL | v19;
      }
      *(a2 - 3) = v17;
      if ( v17 )
      {
        v20 = *(_QWORD *)(v17 + 8);
        *(a2 - 2) = v20;
        if ( v20 )
          *(_QWORD *)(v20 + 16) = (unsigned __int64)(a2 - 2) | *(_QWORD *)(v20 + 16) & 3LL;
        *(a2 - 1) = (v17 + 8) | *(a2 - 1) & 3;
        *(_QWORD *)(v17 + 8) = a2 - 3;
      }
      return v10;
    }
    goto LABEL_12;
  }
  v28 = *(unsigned __int8 *)(v13 + 16);
  if ( (_BYTE)v28 == 13 )
  {
    v29 = *(_QWORD **)(v13 + 24);
    if ( *(_DWORD *)(v13 + 32) > 0x40u )
      v29 = (_QWORD *)*v29;
    v239 = (_QWORD *)v10;
    v30 = sub_1645D80(*(__int64 **)(v10 + 56), (__int64)v29);
    v31 = a1[1];
    v245[0] = sub_1649960(v10);
    v250 = 261;
    v248 = v245;
    v245[1] = v32;
    v33 = *(_DWORD *)(sub_1632FA0(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v31 + 8) + 56LL) + 40LL)) + 4);
    v253[0] = 257;
    v34 = sub_1648A60(64, 1u);
    v35 = v34;
    if ( v34 )
      sub_15F8BC0((__int64)v34, v30, v33, 0, (__int64)&v251, 0);
    v36 = *(_QWORD *)(v31 + 8);
    if ( v36 )
    {
      v37 = *(unsigned __int64 **)(v31 + 16);
      sub_157E9D0(v36 + 40, (__int64)v35);
      v38 = v35[3];
      v39 = *v37;
      v35[4] = v37;
      v39 &= 0xFFFFFFFFFFFFFFF8LL;
      v35[3] = v39 | v38 & 7;
      *(_QWORD *)(v39 + 8) = v35 + 3;
      *v37 = *v37 & 7 | (unsigned __int64)(v35 + 3);
    }
    v40 = v35;
    v41 = (__int64 *)&v248;
    sub_164B780((__int64)v35, (__int64 *)&v248);
    v11 = *(_QWORD *)(v31 + 80) == 0;
    v243 = v35;
    if ( !v11 )
    {
      (*(void (__fastcall **)(__int64, _QWORD **))(v31 + 88))(v31 + 64, &v243);
      v43 = *(_QWORD *)v31;
      if ( *(_QWORD *)v31 )
      {
        v254 = *(const char ***)v31;
        sub_1623A60((__int64)&v254, v43, 2);
        v44 = v35[6];
        if ( v44 )
          sub_161E7C0((__int64)(v35 + 6), v44);
        v45 = (unsigned __int8 *)v254;
        v35[6] = v254;
        if ( v45 )
          sub_1623210((__int64)&v254, v45, (__int64)(v35 + 6));
        sub_15F8A20((__int64)v35, (unsigned int)(1 << *(_WORD *)(v10 + 18)) >> 1);
      }
      else
      {
        sub_15F8A20((__int64)v35, (unsigned int)(1 << *(_WORD *)(v10 + 18)) >> 1);
        if ( !v35 )
LABEL_351:
          BUG();
      }
      v46 = v35 + 3;
      while ( 1 )
      {
        v47 = *((_BYTE *)v46 - 8);
        if ( v47 != 53 )
        {
          if ( v47 != 78 )
            break;
          v99 = *(v46 - 6);
          if ( *(_BYTE *)(v99 + 16)
            || (*(_BYTE *)(v99 + 33) & 0x20) == 0
            || (unsigned int)(*(_DWORD *)(v99 + 36) - 35) > 3 )
          {
            break;
          }
        }
        v46 = (unsigned __int64 *)v46[1];
        if ( !v46 )
          goto LABEL_351;
      }
      v48 = *(_QWORD *)v10;
      v49 = (__int64 **)sub_15A9650(a1[333], *(_QWORD *)v10);
      v248 = (_QWORD *)sub_15A06D0(v49, v48, v50, v51);
      v249 = (__int64)v248;
      v251 = sub_1649960((__int64)v35);
      v254 = &v251;
      v252 = v52;
      LOWORD(v256) = 773;
      v255 = (__int64)".sub";
      v53 = *v35;
      if ( *(_BYTE *)(*v35 + 8LL) == 16 )
        v53 = **(_QWORD **)(v53 + 16);
      v237 = *(_QWORD *)(v53 + 24);
      v54 = sub_1648A60(72, 3u);
      v55 = v54;
      if ( v54 )
      {
        v56 = v54 - 9;
        v57 = *v35;
        if ( *(_BYTE *)(*v35 + 8LL) == 16 )
          v57 = **(_QWORD **)(v57 + 16);
        v231 = (__int64)v56;
        v236 = *(_DWORD *)(v57 + 8) >> 8;
        v58 = (__int64 *)sub_15F9F50(v237, (__int64)&v248, 2);
        v59 = sub_1646BA0(v58, v236);
        v60 = v231;
        v61 = (__int64 *)v59;
        v62 = *v35;
        if ( *(_BYTE *)(*v35 + 8LL) == 16
          || (v62 = *v248, *(_BYTE *)(*v248 + 8LL) == 16)
          || (v62 = *(_QWORD *)v249, *(_BYTE *)(*(_QWORD *)v249 + 8LL) == 16) )
        {
          v103 = sub_16463B0(v61, *(_QWORD *)(v62 + 32));
          v60 = v231;
          v61 = v103;
        }
        sub_15F1EA0((__int64)v55, (__int64)v61, 32, v60, 3, 0);
        v55[7] = v237;
        v55[8] = sub_15F9F50(v237, (__int64)&v248, 2);
        sub_15F9CE0((__int64)v55, (__int64)v35, (__int64 *)&v248, 2, (__int64)&v254);
      }
      sub_15FA2E0((__int64)v55, 1);
      sub_157E9D0(v46[2] + 40, (__int64)v55);
      v63 = *v46;
      v64 = v55[3];
      v55[4] = v46;
      v63 &= 0xFFFFFFFFFFFFFFF8LL;
      v55[3] = v63 | v64 & 7;
      *(_QWORD *)(v63 + 8) = v55 + 3;
      *v46 = *v46 & 7 | (unsigned __int64)(v55 + 3);
      sub_170B990(*a1, (__int64)v55);
      v65 = *(_QWORD *)(v10 + 8);
      if ( v65 )
      {
        v66 = *a1;
        do
        {
          v67 = sub_1648700(v65);
          sub_170B990(v66, (__int64)v67);
          v65 = *(_QWORD *)(v65 + 8);
        }
        while ( v65 );
        if ( (_QWORD *)v10 == v55 )
          v55 = (_QWORD *)sub_1599EF0(*(__int64 ***)v10);
        sub_164D160(v10, (__int64)v55, a3, a4, a5, a6, v68, v69, a9, a10);
        return (__int64)v239;
      }
      goto LABEL_12;
    }
LABEL_331:
    sub_4263D6(v40, v41, v42);
  }
  v70 = *a2;
  if ( (_BYTE)v28 == 9 )
  {
    v88 = sub_15A06D0(*(__int64 ***)v10, v70, v28, v12);
    v89 = *(_QWORD *)(v10 + 8);
    v239 = (_QWORD *)v10;
    v90 = v88;
    if ( v89 )
    {
      v91 = *a1;
      do
      {
        v92 = sub_1648700(v89);
        sub_170B990(v91, (__int64)v92);
        v89 = *(_QWORD *)(v89 + 8);
      }
      while ( v89 );
      if ( v90 == v10 )
        v90 = sub_1599EF0(*(__int64 ***)v90);
      goto LABEL_81;
    }
    goto LABEL_12;
  }
  v71 = (__int64 **)sub_15A9650(a1[333], v70);
  v72 = *(_QWORD *)(v10 - 24);
  if ( v71 != *(__int64 ***)v72 )
  {
    v73 = a1[1];
    v247 = 257;
    if ( v71 != *(__int64 ***)v72 )
    {
      if ( *(_BYTE *)(v72 + 16) > 0x10u )
      {
        LOWORD(v256) = 257;
        v72 = sub_15FE0A0((_QWORD *)v72, (__int64)v71, 0, (__int64)&v254, 0);
        v127 = *(_QWORD *)(v73 + 8);
        if ( v127 )
        {
          v128 = *(__int64 **)(v73 + 16);
          sub_157E9D0(v127 + 40, v72);
          v129 = *(_QWORD *)(v72 + 24);
          v130 = *v128;
          *(_QWORD *)(v72 + 32) = v128;
          v130 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v72 + 24) = v130 | v129 & 7;
          *(_QWORD *)(v130 + 8) = v72 + 24;
          *v128 = *v128 & 7 | (v72 + 24);
        }
        v41 = v246;
        v40 = (_QWORD *)v72;
        sub_164B780(v72, v246);
        v11 = *(_QWORD *)(v73 + 80) == 0;
        v244 = v72;
        if ( v11 )
          goto LABEL_331;
        (*(void (__fastcall **)(__int64, __int64 *))(v73 + 88))(v73 + 64, &v244);
        v131 = *(_QWORD *)v73;
        if ( *(_QWORD *)v73 )
        {
          v251 = *(const char **)v73;
          sub_1623A60((__int64)&v251, v131, 2);
          v132 = *(_QWORD *)(v72 + 48);
          if ( v132 )
            sub_161E7C0(v72 + 48, v132);
          v133 = (unsigned __int8 *)v251;
          *(_QWORD *)(v72 + 48) = v251;
          if ( v133 )
            sub_1623210((__int64)&v251, v133, v72 + 48);
        }
      }
      else
      {
        v72 = sub_15A4750((__int64 ***)v72, v71, 0);
        v74 = sub_14DBA30(v72, *(_QWORD *)(v73 + 96), 0);
        if ( v74 )
        {
          v75 = v10 - 24;
          if ( !*(_QWORD *)(v10 - 24) )
          {
            *(_QWORD *)(v10 - 24) = v74;
            v72 = v74;
            goto LABEL_57;
          }
          v72 = v74;
          goto LABEL_163;
        }
      }
      v75 = v10 - 24;
      if ( !*(_QWORD *)(v10 - 24) )
        goto LABEL_165;
      goto LABEL_163;
    }
    v75 = v10 - 24;
LABEL_163:
    v134 = *(_QWORD *)(v10 - 16);
    v135 = *(_QWORD *)(v10 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v135 = v134;
    if ( v134 )
      *(_QWORD *)(v134 + 16) = *(_QWORD *)(v134 + 16) & 3LL | v135;
LABEL_165:
    *(_QWORD *)(v10 - 24) = v72;
    if ( !v72 )
      return v10;
LABEL_57:
    v76 = *(_QWORD *)(v72 + 8);
    *(_QWORD *)(v10 - 16) = v76;
    if ( v76 )
      *(_QWORD *)(v76 + 16) = (v10 - 16) | *(_QWORD *)(v76 + 16) & 3LL;
    *(_QWORD *)(v10 - 8) = (v72 + 8) | *(_QWORD *)(v10 - 8) & 3LL;
    *(_QWORD *)(v72 + 8) = v75;
    return v10;
  }
LABEL_12:
  v22 = *(_QWORD *)(v10 + 56);
  v23 = *(unsigned __int8 *)(v22 + 8);
  if ( (unsigned __int8)v23 <= 0xFu )
  {
    v24 = 35454;
    if ( _bittest64(&v24, v23) )
    {
LABEL_14:
      if ( !((unsigned int)(1 << *(_WORD *)(v10 + 18)) >> 1) )
      {
        v139 = sub_15AAE50(a1[333], v22);
        sub_15F8A20(v10, v139);
        v22 = *(_QWORD *)(v10 + 56);
      }
      v25 = 1;
      v26 = a1[333];
      v27 = (unsigned int)sub_15A9FE0(v26, v22);
      while ( 2 )
      {
        LODWORD(v23) = *(unsigned __int8 *)(v22 + 8);
        switch ( (char)v23 )
        {
          case 0:
          case 8:
          case 10:
          case 12:
            v136 = *(_QWORD *)(v22 + 32);
            v22 = *(_QWORD *)(v22 + 24);
            v25 *= v136;
            continue;
          case 1:
            v122 = 16;
            goto LABEL_142;
          case 2:
            v122 = 32;
            goto LABEL_142;
          case 3:
          case 9:
            v122 = 64;
            goto LABEL_142;
          case 4:
            v122 = 80;
            goto LABEL_142;
          case 5:
          case 6:
            v122 = 128;
            goto LABEL_142;
          case 7:
            v122 = 8 * (unsigned int)sub_15A9520(v26, 0);
            goto LABEL_142;
          case 11:
            v122 = *(_DWORD *)(v22 + 8) >> 8;
            goto LABEL_142;
          case 13:
            v122 = 8LL * *(_QWORD *)sub_15A9930(v26, v22);
            goto LABEL_142;
          case 14:
            v238 = *(_QWORD *)(v22 + 24);
            v241 = *(_QWORD *)(v22 + 32);
            v137 = (unsigned int)sub_15A9FE0(v26, v238);
            v122 = 8 * v137 * v241 * ((v137 + ((unsigned __int64)(sub_127FA20(v26, v238) + 7) >> 3) - 1) / v137);
            goto LABEL_142;
          case 15:
            v122 = 8 * (unsigned int)sub_15A9520(v26, *(_DWORD *)(v22 + 8) >> 8);
LABEL_142:
            if ( v27 * ((v27 + ((unsigned __int64)(v25 * v122 + 7) >> 3) - 1) / v27) )
              goto LABEL_63;
            if ( (unsigned __int8)sub_15F8BF0(v10) )
            {
              v140 = sub_15A0680(**(_QWORD **)(v10 - 24), 1, 0);
              if ( *(_QWORD *)(v10 - 24) )
              {
                v141 = *(_QWORD *)(v10 - 16);
                v142 = *(_QWORD *)(v10 - 8) & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v142 = v141;
                if ( v141 )
                  *(_QWORD *)(v141 + 16) = *(_QWORD *)(v141 + 16) & 3LL | v142;
              }
              *(_QWORD *)(v10 - 24) = v140;
              if ( v140 )
              {
                v143 = *(_QWORD *)(v140 + 8);
                *(_QWORD *)(v10 - 16) = v143;
                if ( v143 )
                  *(_QWORD *)(v143 + 16) = (v10 - 16) | *(_QWORD *)(v143 + 16) & 3LL;
                *(_QWORD *)(v10 - 8) = *(_QWORD *)(v10 - 8) & 3LL | (v140 + 8);
                *(_QWORD *)(v140 + 8) = v10 - 24;
              }
              return v10;
            }
            v123 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v10 + 40) + 56LL) + 80LL);
            if ( v123 )
              v123 -= 24;
            v124 = sub_157ED60(v123);
            v239 = (_QWORD *)v10;
            v90 = v124;
            if ( v10 == v124 )
              goto LABEL_63;
            if ( *(_BYTE *)(v124 + 16) != 53 )
              goto LABEL_152;
            v125 = *(_QWORD *)(v124 + 56);
            v126 = *(unsigned __int8 *)(v125 + 8);
            if ( (unsigned __int8)v126 <= 0xFu )
            {
              v155 = 35454;
              if ( _bittest64(&v155, v126) )
                goto LABEL_216;
            }
            if ( (unsigned int)(v126 - 13) > 1 && (_DWORD)v126 != 16 || !sub_16435F0(v125, 0) )
              goto LABEL_152;
            v125 = *(_QWORD *)(v90 + 56);
LABEL_216:
            if ( sub_12BE0A0(a1[333], v125) )
            {
LABEL_152:
              sub_15F22F0((_QWORD *)v10, v90);
              return (__int64)v239;
            }
            v156 = (unsigned int)(1 << *(_WORD *)(v90 + 18)) >> 1;
            if ( !v156 )
            {
              v217 = sub_15AAE50(a1[333], *(_QWORD *)(v90 + 56));
              sub_15F8A20(v90, v217);
              v156 = (unsigned int)(1 << *(_WORD *)(v90 + 18)) >> 1;
            }
            if ( (unsigned int)(1 << *(_WORD *)(v10 + 18)) >> 1 >= v156 )
              v156 = (unsigned int)(1 << *(_WORD *)(v10 + 18)) >> 1;
            sub_15F8A20(v90, v156);
            v157 = *(_QWORD *)v10;
            if ( *(_QWORD *)v90 != *(_QWORD *)v10 )
            {
              LOWORD(v256) = 257;
              v239 = sub_1648A60(56, 1u);
              if ( v239 )
                sub_15FD590((__int64)v239, v90, v157, (__int64)&v254, 0);
              return (__int64)v239;
            }
            v214 = *(_QWORD *)(v10 + 8);
            if ( !v214 )
              return 0;
            v215 = *a1;
            do
            {
              v216 = sub_1648700(v214);
              sub_170B990(v215, (__int64)v216);
              v214 = *(_QWORD *)(v214 + 8);
            }
            while ( v214 );
            break;
          default:
            goto LABEL_60;
        }
        break;
      }
LABEL_81:
      sub_164D160(v10, v90, a3, a4, a5, a6, v93, v94, a9, a10);
      return (__int64)v239;
    }
  }
LABEL_60:
  if ( ((unsigned int)(v23 - 13) <= 1 || (_DWORD)v23 == 16) && sub_16435F0(v22, 0) )
  {
    v22 = *(_QWORD *)(v10 + 56);
    goto LABEL_14;
  }
LABEL_63:
  if ( !((unsigned int)(1 << *(_WORD *)(v10 + 18)) >> 1) )
    return sub_170E170(a1, v10, a3, a4, a5, a6, v14, v15, a9, a10);
  v77 = (const char **)&v256;
  v78 = v10;
  v79 = v10;
  v254 = (const char **)&v256;
  HIDWORD(v255) = 35;
  v256 = v10;
  v257 = 0;
  v240 = 0;
  v232 = 0;
  v251 = (const char *)v253;
  v252 = 0x400000000LL;
  v80 = 1;
  while ( 1 )
  {
    LODWORD(v255) = --v80;
    if ( *(_QWORD *)(v78 + 8) )
      break;
LABEL_72:
    if ( !v80 )
    {
      v183 = v232;
      v10 = v79;
      if ( v77 != (const char **)&v256 )
        _libc_free((unsigned __int64)v77);
      v239 = (_QWORD *)v79;
      if ( v232 )
      {
        v184 = (unsigned int)(1 << *(_WORD *)(v79 + 18)) >> 1;
        v226 = a1[332];
        v229 = a1[330];
        v233 = a1[333];
        v185 = sub_1649C60(*(_QWORD *)(v183 + 24 * (1LL - (*(_DWORD *)(v183 + 20) & 0xFFFFFFF))));
        if ( (unsigned int)sub_1AE99B0(v185, v184, v233, v79, v229, v226) >= (unsigned int)(1 << *(_WORD *)(v79 + 18)) >> 1 )
        {
          v186 = a1[333];
          v187 = sub_1649C60(*(_QWORD *)(v183 + 24 * (1LL - (*(_DWORD *)(v183 + 20) & 0xFFFFFFF))));
          if ( !(unsigned __int8)sub_15F8BF0(v10) )
          {
            v188 = *(_QWORD *)(v10 + 56);
            v189 = 1;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v188 + 8) )
              {
                case 1:
                  v190 = 16;
                  goto LABEL_275;
                case 2:
                  v190 = 32;
                  goto LABEL_275;
                case 3:
                case 9:
                  v190 = 64;
                  goto LABEL_275;
                case 4:
                  v190 = 80;
                  goto LABEL_275;
                case 5:
                case 6:
                  v190 = 128;
                  goto LABEL_275;
                case 7:
                  v190 = 8 * (unsigned int)sub_15A9520(v186, 0);
                  goto LABEL_275;
                case 0xB:
                  v190 = *(_DWORD *)(v188 + 8) >> 8;
                  goto LABEL_275;
                case 0xD:
                  v190 = 8LL * *(_QWORD *)sub_15A9930(v186, v188);
                  goto LABEL_275;
                case 0xE:
                  v230 = *(_QWORD *)(v188 + 32);
                  v208 = *(_QWORD *)(v188 + 24);
                  v235 = 1;
                  v227 = (unsigned int)sub_15A9FE0(v186, v208);
                  while ( 2 )
                  {
                    switch ( *(_BYTE *)(v208 + 8) )
                    {
                      case 1:
                        v218 = 16;
                        goto LABEL_333;
                      case 2:
                        v218 = 32;
                        goto LABEL_333;
                      case 3:
                      case 9:
                        v218 = 64;
                        goto LABEL_333;
                      case 4:
                        v218 = 80;
                        goto LABEL_333;
                      case 5:
                      case 6:
                        v218 = 128;
                        goto LABEL_333;
                      case 7:
                        v218 = 8 * (unsigned int)sub_15A9520(v186, 0);
                        goto LABEL_333;
                      case 0xB:
                        v218 = *(_DWORD *)(v208 + 8) >> 8;
                        goto LABEL_333;
                      case 0xD:
                        v218 = 8LL * *(_QWORD *)sub_15A9930(v186, v208);
                        goto LABEL_333;
                      case 0xE:
                        v220 = *(_QWORD *)(v208 + 24);
                        v224 = *(_QWORD *)(v208 + 32);
                        v221 = (unsigned int)sub_15A9FE0(v186, v220);
                        v218 = 8
                             * v224
                             * v221
                             * ((v221 + ((unsigned __int64)(sub_127FA20(v186, v220) + 7) >> 3) - 1)
                              / v221);
                        goto LABEL_333;
                      case 0xF:
                        v218 = 8 * (unsigned int)sub_15A9520(v186, *(_DWORD *)(v208 + 8) >> 8);
LABEL_333:
                        v190 = 8 * v227 * v230 * ((v227 + ((unsigned __int64)(v235 * v218 + 7) >> 3) - 1) / v227);
                        goto LABEL_275;
                      case 0x10:
                        v219 = v235 * *(_QWORD *)(v208 + 32);
                        v208 = *(_QWORD *)(v208 + 24);
                        v235 = v219;
                        continue;
                      default:
                        goto LABEL_352;
                    }
                  }
                case 0xF:
                  v190 = 8 * (unsigned int)sub_15A9520(v186, *(_DWORD *)(v188 + 8) >> 8);
LABEL_275:
                  v191 = (unsigned __int64)(v189 * v190 + 7) >> 3;
                  if ( !v191 )
                    goto LABEL_87;
                  v192 = *(_WORD *)(v10 + 18);
                  v254 = (const char **)v191;
                  LODWORD(v255) = 64;
                  v193 = sub_13F8110(v187, (unsigned int)(1 << v192) >> 1, (unsigned __int64)&v254, v186, 0, 0);
                  if ( (unsigned int)v255 > 0x40 && v254 )
                    j_j___libc_free_0_0(v254);
                  if ( !v193 )
                    goto LABEL_87;
                  v194 = 8LL * (unsigned int)v252;
                  if ( (_DWORD)v252 )
                  {
                    v234 = v10;
                    v195 = 0;
                    do
                    {
                      v196 = *(_QWORD *)&v251[v195];
                      v195 += 8;
                      sub_170BC50((__int64)a1, v196);
                    }
                    while ( v195 != v194 );
                    v10 = v234;
                  }
                  v197 = sub_1649C60(*(_QWORD *)(v183 + 24 * (1LL - (*(_DWORD *)(v183 + 20) & 0xFFFFFFF))));
                  v198 = *(_QWORD *)v197;
                  v199 = (__int64 ***)v197;
                  v11 = *(_BYTE *)(*(_QWORD *)v197 + 8LL) == 16;
                  v200 = *(_QWORD *)v197;
                  if ( v11 )
                    v200 = **(_QWORD **)(v198 + 16);
                  v201 = sub_1646BA0(**(__int64 ***)(*(_QWORD *)v10 + 16LL), *(_DWORD *)(v200 + 8) >> 8);
                  v202 = (__int64 ***)sub_15A4AD0(v199, v201);
                  v203 = *(_QWORD *)v10;
                  if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) == 16 )
                    v203 = **(_QWORD **)(v203 + 16);
                  v204 = *(_DWORD *)(v203 + 8) >> 8;
                  if ( *(_BYTE *)(v198 + 8) == 16 )
                    v198 = **(_QWORD **)(v198 + 16);
                  if ( *(_DWORD *)(v198 + 8) >> 8 == v204 )
                  {
                    v209 = *(_QWORD *)(v10 + 8);
                    if ( v209 )
                    {
                      v210 = *a1;
                      do
                      {
                        v211 = sub_1648700(v209);
                        sub_170B990(v210, (__int64)v211);
                        v209 = *(_QWORD *)(v209 + 8);
                      }
                      while ( v209 );
                      if ( v202 == (__int64 ***)v10 )
                        v202 = (__int64 ***)sub_1599EF0(*v202);
                      sub_164D160(v10, (__int64)v202, a3, a4, a5, a6, v212, v213, a9, a10);
                    }
                    else
                    {
                      v239 = 0;
                    }
                    sub_170BC50((__int64)a1, v183);
                    if ( v251 != (const char *)v253 )
                      _libc_free((unsigned __int64)v251);
                    return (__int64)v239;
                  }
                  v258 = 0;
                  v259 = 0;
                  v254 = (const char **)&v256;
                  v255 = 0x400000000LL;
                  v260 = 0;
                  v261 = 0;
                  v262 = 0;
                  v263 = 0;
                  v264 = 0;
                  v265 = a1;
                  v248 = (_QWORD *)v10;
                  *(_QWORD *)sub_177C990((__int64)&v258, (unsigned __int64 *)&v248) = v202;
                  sub_177D390((__int64)&v254, v10, a3, a4, a5, a6, v205, v206, a9, a10);
                  if ( v262 )
                    j_j___libc_free_0(v262, v264 - v262);
                  j___libc_free_0(v259);
                  v98 = (unsigned __int64)v254;
                  if ( v254 != (const char **)&v256 )
                    goto LABEL_86;
                  break;
                case 0x10:
                  v207 = *(_QWORD *)(v188 + 32);
                  v188 = *(_QWORD *)(v188 + 24);
                  v189 *= v207;
                  continue;
                default:
                  goto LABEL_352;
              }
              break;
            }
          }
        }
      }
      goto LABEL_87;
    }
    v87 = (__int64)&v77[2 * v80 - 2];
    v78 = *(_QWORD *)v87;
    v240 = *(_BYTE *)(v87 + 8);
  }
  v81 = v79;
  v82 = *(_QWORD *)(v78 + 8);
  while ( 1 )
  {
    v85 = (__int64)sub_1648700(v82);
    v86 = *(_BYTE *)(v85 + 16);
    if ( v86 == 54 )
    {
      if ( sub_15F32D0(v85) || (*(_BYTE *)(v85 + 18) & 1) != 0 )
        goto LABEL_85;
      goto LABEL_70;
    }
    if ( (unsigned __int8)(v86 - 71) <= 1u )
    {
      v101 = v255;
      if ( (unsigned int)v255 >= HIDWORD(v255) )
      {
        sub_16CD150((__int64)&v254, &v256, 0, 16, v83, v84);
        v101 = v255;
      }
      v138 = &v254[2 * v101];
      if ( v138 )
      {
        *v138 = (const char *)v85;
        *((_BYTE *)v138 + 8) = v240;
        v101 = v255;
      }
      goto LABEL_103;
    }
    if ( v86 != 56 )
      break;
    v100 = v240;
    if ( !v240 )
      v100 = sub_15FA1F0(v85) ^ 1;
    v101 = v255;
    if ( (unsigned int)v255 >= HIDWORD(v255) )
    {
      sub_16CD150((__int64)&v254, &v256, 0, 16, v83, v84);
      v101 = v255;
    }
    v102 = &v254[2 * v101];
    if ( v102 )
    {
      *v102 = (const char *)v85;
      *((_BYTE *)v102 + 8) = v100;
      v101 = v255;
    }
LABEL_103:
    LODWORD(v255) = v101 + 1;
LABEL_70:
    v82 = *(_QWORD *)(v82 + 8);
    if ( !v82 )
    {
      v80 = v255;
      v77 = v254;
      v79 = v81;
      goto LABEL_72;
    }
  }
  if ( v86 <= 0x17u )
    goto LABEL_85;
  if ( v86 == 78 )
  {
    v104 = v85 | 4;
    v246[0] = v85 | 4;
    v105 = v85 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v85 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      goto LABEL_109;
  }
  else
  {
    if ( v86 != 29 )
      goto LABEL_85;
    v104 = v85 & 0xFFFFFFFFFFFFFFFBLL;
    v246[0] = v85 & 0xFFFFFFFFFFFFFFFBLL;
    v105 = v85 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v85 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_85;
LABEL_109:
    v106 = v104;
    v107 = v105 - 72;
    v108 = v105;
    v109 = (v106 >> 2) & 1;
    if ( v109 )
      v107 = v105 - 24;
    v228 = v109;
    if ( v107 == v82 )
      goto LABEL_70;
    v110 = *(_BYTE *)(v105 + 23);
    if ( (v110 & 0x40) != 0 )
      v111 = *(_QWORD *)(v105 - 8);
    else
      v111 = v105 - 24LL * (*(_DWORD *)(v105 + 20) & 0xFFFFFFF);
    v112 = v105 - 24LL * (*(_DWORD *)(v105 + 20) & 0xFFFFFFF);
    v225 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v82 - v111) >> 3);
    if ( (_BYTE)v109 )
    {
      if ( v82 < v112 )
      {
        v228 = 0;
        v113 = (_QWORD *)(v105 + 56);
        goto LABEL_117;
      }
      if ( v110 < 0 )
      {
        v177 = sub_1648A40(v105);
        v179 = v177 + v178;
        if ( *(char *)(v105 + 23) >= 0 )
        {
          if ( (unsigned int)(v179 >> 4) )
LABEL_352:
            BUG();
        }
        else if ( (unsigned int)((v179 - sub_1648A40(v105)) >> 4) )
        {
          if ( *(char *)(v105 + 23) >= 0 )
            goto LABEL_352;
          v180 = *(_DWORD *)(sub_1648A40(v105) + 8);
          if ( *(char *)(v105 + 23) >= 0 )
LABEL_350:
            BUG();
          v181 = sub_1648A40(v105);
          v168 = -24LL * (unsigned int)(*(_DWORD *)(v181 + v182 - 4) - v180) - 24;
LABEL_239:
          v108 = v246[0] & 0xFFFFFFFFFFFFFFF8LL;
          v169 = (v246[0] >> 2) & 1;
          goto LABEL_240;
        }
        v168 = -24;
        v108 = v246[0] & 0xFFFFFFFFFFFFFFF8LL;
        v169 = (v246[0] >> 2) & 1;
        goto LABEL_240;
      }
      v171 = (_QWORD *)(v105 + 56);
      if ( v82 >= v105 - 24 )
      {
        v228 = 0;
        v113 = (_QWORD *)(v105 + 56);
LABEL_117:
        if ( !(unsigned __int8)sub_1560260(v113, -1, 36) )
        {
          if ( *(char *)(v108 + 23) < 0 )
          {
            v114 = sub_1648A40(v108);
            v116 = v115 + v114;
            v117 = 0;
            v222 = v116;
            if ( *(char *)(v108 + 23) < 0 )
              v117 = sub_1648A40(v108);
            if ( (unsigned int)((v222 - v117) >> 4) )
              goto LABEL_355;
          }
          v118 = *(_QWORD *)(v108 - 24);
          if ( *(_BYTE *)(v118 + 16) || (v248 = *(_QWORD **)(v118 + 112), !(unsigned __int8)sub_1560260(&v248, -1, 36)) )
          {
LABEL_355:
            if ( !(unsigned __int8)sub_1560260(v113, -1, 37) )
            {
              if ( *(char *)(v108 + 23) < 0 )
              {
                v173 = sub_1648A40(v108);
                v175 = v173 + v174;
                v176 = *(char *)(v108 + 23) >= 0 ? 0LL : sub_1648A40(v108);
                if ( v176 != v175 )
                {
                  while ( *(_DWORD *)(*(_QWORD *)v176 + 8LL) <= 1u )
                  {
                    v176 += 16;
                    if ( v175 == v176 )
                      goto LABEL_259;
                  }
                  goto LABEL_126;
                }
              }
LABEL_259:
              v153 = *(_QWORD *)(v108 - 24);
              if ( *(_BYTE *)(v153 + 16) )
                goto LABEL_126;
LABEL_207:
              v248 = *(_QWORD **)(v153 + 112);
              if ( !(unsigned __int8)sub_1560260(&v248, -1, 37) )
                goto LABEL_126;
            }
          }
        }
LABEL_124:
        if ( !*(_QWORD *)((v246[0] & 0xFFFFFFFFFFFFFFF8LL) + 8) || (unsigned __int8)sub_1779030(v246, (int)v225 + 1, 22) )
          goto LABEL_70;
        goto LABEL_126;
      }
LABEL_245:
      if ( (unsigned __int8)sub_1560290(v171, v225, 11) )
        goto LABEL_85;
      v172 = *(_QWORD *)(v108 - 24);
      if ( !*(_BYTE *)(v172 + 16) )
      {
LABEL_247:
        v248 = *(_QWORD **)(v172 + 112);
        if ( (unsigned __int8)sub_1560290(&v248, v225, 11) )
          goto LABEL_85;
      }
LABEL_248:
      v228 = 1;
      v108 = v246[0] & 0xFFFFFFFFFFFFFFF8LL;
      v169 = (v246[0] >> 2) & 1;
      v113 = (_QWORD *)((v246[0] & 0xFFFFFFFFFFFFFFF8LL) + 56);
LABEL_242:
      if ( (_BYTE)v169 )
        goto LABEL_117;
    }
    else
    {
      if ( v82 >= v112 )
      {
        if ( v110 >= 0 )
        {
          v171 = (_QWORD *)(v105 + 56);
          if ( v82 >= v105 - 72 )
          {
            v113 = (_QWORD *)(v105 + 56);
            goto LABEL_192;
          }
        }
        else
        {
          v162 = sub_1648A40(v105);
          v164 = v162 + v163;
          if ( *(char *)(v105 + 23) >= 0 )
          {
            if ( (unsigned int)(v164 >> 4) )
              goto LABEL_352;
          }
          else if ( (unsigned int)((v164 - sub_1648A40(v105)) >> 4) )
          {
            if ( *(char *)(v105 + 23) >= 0 )
              goto LABEL_352;
            v165 = *(_DWORD *)(sub_1648A40(v105) + 8);
            if ( *(char *)(v105 + 23) >= 0 )
              goto LABEL_350;
            v166 = sub_1648A40(v105);
            v168 = -24LL * (unsigned int)(*(_DWORD *)(v166 + v167 - 4) - v165) - 72;
            goto LABEL_239;
          }
          v168 = -72;
          v108 = v246[0] & 0xFFFFFFFFFFFFFFF8LL;
          v169 = (v246[0] >> 2) & 1;
LABEL_240:
          v170 = v105 + v168;
          v113 = (_QWORD *)(v108 + 56);
          v171 = (_QWORD *)(v108 + 56);
          if ( v170 <= v82 )
          {
            v228 = 0;
            goto LABEL_242;
          }
          if ( (_BYTE)v169 )
            goto LABEL_245;
        }
        if ( (unsigned __int8)sub_1560290(v171, v225, 11) )
          goto LABEL_85;
        v172 = *(_QWORD *)(v108 - 72);
        if ( !*(_BYTE *)(v172 + 16) )
          goto LABEL_247;
        goto LABEL_248;
      }
      v113 = (_QWORD *)(v105 + 56);
    }
LABEL_192:
    if ( (unsigned __int8)sub_1560260(v113, -1, 36) )
      goto LABEL_124;
    if ( *(char *)(v108 + 23) >= 0 )
      goto LABEL_356;
    v144 = sub_1648A40(v108);
    v146 = v145 + v144;
    v147 = 0;
    v223 = v146;
    if ( *(char *)(v108 + 23) < 0 )
      v147 = sub_1648A40(v108);
    if ( !(unsigned int)((v223 - v147) >> 4) )
    {
LABEL_356:
      v148 = *(_QWORD *)(v108 - 72);
      if ( !*(_BYTE *)(v148 + 16) )
      {
        v248 = *(_QWORD **)(v148 + 112);
        if ( (unsigned __int8)sub_1560260(&v248, -1, 36) )
          goto LABEL_124;
      }
    }
    if ( (unsigned __int8)sub_1560260(v113, -1, 37) )
      goto LABEL_124;
    if ( *(char *)(v108 + 23) >= 0
      || ((v149 = sub_1648A40(v108), v151 = v149 + v150, *(char *)(v108 + 23) >= 0)
        ? (v152 = 0)
        : (v152 = sub_1648A40(v108)),
          v152 == v151) )
    {
LABEL_206:
      v153 = *(_QWORD *)(v108 - 72);
      if ( *(_BYTE *)(v153 + 16) )
        goto LABEL_126;
      goto LABEL_207;
    }
    while ( *(_DWORD *)(*(_QWORD *)v152 + 8LL) <= 1u )
    {
      v152 += 16;
      if ( v151 == v152 )
        goto LABEL_206;
    }
LABEL_126:
    if ( v228 )
    {
      v159 = v246[0] & 0xFFFFFFFFFFFFFFF8LL;
      v160 = (_QWORD *)((v246[0] & 0xFFFFFFFFFFFFFFF8LL) + 56);
      if ( (v246[0] & 4) != 0 )
      {
        if ( (unsigned __int8)sub_1560290(v160, v225, 6) )
          goto LABEL_70;
        v161 = *(_QWORD *)(v159 - 24);
        if ( *(_BYTE *)(v161 + 16) )
          goto LABEL_127;
      }
      else
      {
        if ( (unsigned __int8)sub_1560290(v160, v225, 6) )
          goto LABEL_70;
        v161 = *(_QWORD *)(v159 - 72);
        if ( *(_BYTE *)(v161 + 16) )
          goto LABEL_127;
      }
      v248 = *(_QWORD **)(v161 + 112);
      if ( (unsigned __int8)sub_1560290(&v248, v225, 6) )
        goto LABEL_70;
    }
LABEL_127:
    if ( *(_BYTE *)(v85 + 16) != 78 )
      goto LABEL_85;
  }
  v119 = *(_QWORD *)(v85 - 24);
  if ( *(_BYTE *)(v119 + 16) || (*(_BYTE *)(v119 + 33) & 0x20) == 0 )
    goto LABEL_85;
  if ( (unsigned int)(*(_DWORD *)(v119 + 36) - 116) <= 1 )
  {
    v158 = (unsigned int)v252;
    if ( (unsigned int)v252 >= HIDWORD(v252) )
    {
      sub_16CD150((__int64)&v251, v253, 0, 8, v83, v84);
      v158 = (unsigned int)v252;
    }
    *(_QWORD *)&v251[8 * v158] = v85;
    LODWORD(v252) = v252 + 1;
    goto LABEL_70;
  }
  if ( (*(_BYTE *)(v119 + 33) & 0x20) == 0 || (*(_DWORD *)(v119 + 36) & 0xFFFFFFFD) != 0x85 )
    goto LABEL_85;
  if ( (unsigned int)sub_1648720(v82) == 1 )
  {
    v95 = *(_QWORD *)(v85 + 24 * (3LL - (*(_DWORD *)(v85 + 20) & 0xFFFFFFF)));
    v96 = *(_DWORD *)(v95 + 32);
    if ( v96 <= 0x40 )
      v97 = *(_QWORD *)(v95 + 24) == 0;
    else
      v97 = v96 == (unsigned int)sub_16A57B0(v95 + 24);
    if ( !v97 )
      goto LABEL_85;
    goto LABEL_70;
  }
  if ( !v232 && !v240 && !(unsigned int)sub_1648720(v82) )
  {
    for ( i = sub_1649C60(*(_QWORD *)(v85 + 24 * (1LL - (*(_DWORD *)(v85 + 20) & 0xFFFFFFF))));
          ;
          i = *(_QWORD *)(i - 24LL * (*(_DWORD *)(i + 20) & 0xFFFFFFF)) )
    {
      v121 = *(_BYTE *)(i + 16);
      if ( v121 == 3 )
        break;
      if ( v121 != 5 )
        goto LABEL_85;
      v154 = *(unsigned __int16 *)(i + 18);
      if ( (unsigned int)(v154 - 47) > 1 && v154 != 32 )
        goto LABEL_85;
    }
    if ( (*(_BYTE *)(i + 80) & 1) != 0 )
    {
      v232 = v85;
      goto LABEL_70;
    }
  }
LABEL_85:
  v98 = (unsigned __int64)v254;
  v10 = v81;
  if ( v254 != (const char **)&v256 )
LABEL_86:
    _libc_free(v98);
LABEL_87:
  if ( v251 != (const char *)v253 )
    _libc_free((unsigned __int64)v251);
  return sub_170E170(a1, v10, a3, a4, a5, a6, v14, v15, a9, a10);
}
