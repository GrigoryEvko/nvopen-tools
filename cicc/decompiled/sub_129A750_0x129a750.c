// Function: sub_129A750
// Address: 0x129a750
//
__int64 __fastcall sub_129A750(__int64 a1, __int64 a2, __int64 a3, char **a4)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  unsigned int v9; // r14d
  __int64 v10; // rbx
  __int64 v11; // rax
  int v12; // eax
  __int64 v13; // rax
  char *v14; // rsi
  _DWORD *v15; // rbx
  __int64 v16; // r8
  int v17; // edi
  int v18; // eax
  int v19; // r14d
  unsigned int v20; // edi
  _BOOL8 v21; // r12
  unsigned int v22; // r14d
  __int64 v23; // r13
  unsigned __int64 v24; // rdx
  __int64 v25; // rcx
  unsigned __int64 v26; // rsi
  int v27; // r15d
  unsigned int v28; // r15d
  __int64 v29; // r8
  char *v30; // r12
  __int64 i; // rsi
  unsigned __int64 v32; // rsi
  int v33; // edx
  __int64 v34; // rcx
  __int64 v35; // rdx
  __int64 v36; // rdi
  __int64 v37; // rsi
  _BOOL8 v38; // r12
  _BOOL4 v39; // edi
  unsigned int v40; // r15d
  __int64 v41; // r14
  unsigned __int64 v42; // rdx
  __int64 v43; // r8
  unsigned __int64 v44; // rsi
  __int64 v45; // rcx
  __int64 v46; // rcx
  int v47; // ecx
  char *v48; // r12
  __int64 j; // rdi
  unsigned __int64 v50; // rdi
  int v51; // edx
  unsigned int v52; // esi
  __int64 v53; // rdx
  __int64 v54; // r8
  __int64 v55; // r8
  __int64 v56; // rdi
  _BOOL8 v57; // r12
  _BOOL4 v58; // esi
  __int64 v59; // rcx
  __int64 v60; // r15
  unsigned __int64 v61; // rdx
  unsigned int v62; // r9d
  unsigned __int64 v63; // rdi
  __int64 v64; // r8
  __int64 v65; // r8
  unsigned int v66; // ecx
  unsigned int v67; // r8d
  char *v68; // r12
  __int64 k; // r9
  unsigned __int64 v70; // r9
  int v71; // edx
  unsigned int v72; // esi
  __int64 v73; // rdx
  __int64 v74; // r10
  __int64 v75; // r9
  __int64 v76; // rcx
  __m128i *v77; // rax
  __int64 v78; // rcx
  __m128i *v79; // r9
  unsigned __int64 v80; // rax
  unsigned __int64 v81; // rdi
  unsigned __int64 v82; // rcx
  __m128i *v83; // rax
  __int64 v84; // rcx
  __m128i *v85; // rax
  unsigned __int64 v86; // rax
  unsigned __int64 v87; // rdi
  unsigned __int64 v88; // rcx
  __m128i *v89; // rax
  __int64 v90; // rcx
  __int64 v91; // r8
  _BOOL8 v92; // r12
  _BOOL4 v93; // edi
  unsigned int v94; // r14d
  __int64 v95; // r13
  unsigned __int64 v96; // rdx
  __int64 v97; // r8
  unsigned __int64 v98; // rsi
  __int64 v99; // rcx
  __int64 v100; // rcx
  __int64 v101; // rcx
  char *v102; // r12
  __int64 m; // rdi
  unsigned __int64 v104; // rdi
  int v105; // edx
  unsigned int v106; // esi
  __int64 v107; // rdx
  __int64 v108; // r8
  __int64 v109; // rdi
  int v110; // edi
  int v111; // eax
  int v112; // r8d
  unsigned int v113; // edi
  _BOOL8 v114; // r12
  __int64 v115; // r8
  __int64 v116; // rcx
  unsigned __int64 v117; // rdx
  int v118; // r10d
  unsigned __int64 v119; // rsi
  int v120; // r9d
  int v121; // r9d
  int v122; // r9d
  unsigned int v123; // r8d
  __int64 v124; // rcx
  char *v125; // r12
  unsigned __int64 v126; // rcx
  int v127; // edx
  unsigned int v128; // esi
  __int64 v129; // rdx
  __int64 v130; // rdi
  __int64 v131; // rcx
  __int64 v132; // r8
  int v133; // ebx
  int v134; // eax
  int v135; // ecx
  unsigned int v136; // esi
  _BOOL8 v137; // rbx
  __int64 v138; // rcx
  __int64 v139; // r12
  unsigned __int64 v140; // rdx
  unsigned int v141; // r9d
  unsigned __int64 v142; // rdi
  __int64 v143; // r8
  __int64 v144; // r8
  unsigned int v145; // r8d
  unsigned int v146; // ecx
  char *v147; // rbx
  __int64 n; // r9
  unsigned __int64 v149; // r9
  int v150; // edx
  unsigned int v151; // esi
  __int64 v152; // rdx
  __int64 v153; // r10
  __int64 v154; // r9
  __int64 v155; // rcx
  __m128i *v156; // rax
  __int64 v157; // rcx
  __m128i *v158; // r9
  unsigned __int64 v159; // rax
  unsigned __int64 v160; // rdi
  unsigned __int64 v161; // rcx
  __m128i *v162; // rax
  __int64 v163; // rcx
  __m128i *v164; // rax
  unsigned __int64 v165; // rax
  unsigned __int64 v166; // rdi
  unsigned __int64 v167; // rcx
  __m128i *v168; // rax
  __int64 v170; // rax
  char *v171; // rsi
  __int64 v172; // rax
  char *v173; // rsi
  __int64 v174; // r12
  char v175; // al
  char v176; // al
  __int64 v177; // r14
  char v178; // al
  __int64 v179; // r14
  char v180; // al
  char v181; // al
  __int64 v182; // r15
  char v183; // al
  __m128i *v184; // rax
  unsigned int v185; // [rsp+Ch] [rbp-2E4h]
  unsigned int v186; // [rsp+Ch] [rbp-2E4h]
  __int64 v187; // [rsp+10h] [rbp-2E0h]
  unsigned int v188; // [rsp+10h] [rbp-2E0h]
  unsigned int v189; // [rsp+10h] [rbp-2E0h]
  unsigned int v190; // [rsp+10h] [rbp-2E0h]
  unsigned int v191; // [rsp+10h] [rbp-2E0h]
  unsigned int v192; // [rsp+10h] [rbp-2E0h]
  unsigned int v193; // [rsp+18h] [rbp-2D8h]
  unsigned int v194; // [rsp+18h] [rbp-2D8h]
  unsigned int v195; // [rsp+18h] [rbp-2D8h]
  int v196; // [rsp+18h] [rbp-2D8h]
  unsigned int v197; // [rsp+28h] [rbp-2C8h]
  unsigned int v198; // [rsp+28h] [rbp-2C8h]
  unsigned int v199; // [rsp+28h] [rbp-2C8h]
  unsigned int v200; // [rsp+28h] [rbp-2C8h]
  int v201; // [rsp+30h] [rbp-2C0h]
  int v202; // [rsp+30h] [rbp-2C0h]
  unsigned int v203; // [rsp+30h] [rbp-2C0h]
  int v204; // [rsp+30h] [rbp-2C0h]
  __int64 v205; // [rsp+30h] [rbp-2C0h]
  unsigned int v206; // [rsp+30h] [rbp-2C0h]
  __int64 v209; // [rsp+50h] [rbp-2A0h]
  int v210; // [rsp+50h] [rbp-2A0h]
  unsigned int v211; // [rsp+50h] [rbp-2A0h]
  int v212; // [rsp+50h] [rbp-2A0h]
  __int64 v213; // [rsp+58h] [rbp-298h]
  _QWORD v214[2]; // [rsp+60h] [rbp-290h] BYREF
  _QWORD v215[2]; // [rsp+70h] [rbp-280h] BYREF
  __m128i *v216; // [rsp+80h] [rbp-270h] BYREF
  __int64 v217; // [rsp+88h] [rbp-268h]
  __m128i v218; // [rsp+90h] [rbp-260h] BYREF
  _QWORD *v219; // [rsp+A0h] [rbp-250h] BYREF
  __int64 v220; // [rsp+A8h] [rbp-248h]
  _QWORD v221[2]; // [rsp+B0h] [rbp-240h] BYREF
  __m128i *v222; // [rsp+C0h] [rbp-230h] BYREF
  __int64 v223; // [rsp+C8h] [rbp-228h]
  __m128i v224; // [rsp+D0h] [rbp-220h] BYREF
  _QWORD v225[2]; // [rsp+E0h] [rbp-210h] BYREF
  _QWORD v226[2]; // [rsp+F0h] [rbp-200h] BYREF
  __m128i *v227; // [rsp+100h] [rbp-1F0h] BYREF
  __int64 v228; // [rsp+108h] [rbp-1E8h]
  __m128i v229; // [rsp+110h] [rbp-1E0h] BYREF
  _QWORD *v230; // [rsp+120h] [rbp-1D0h] BYREF
  __int64 v231; // [rsp+128h] [rbp-1C8h]
  _QWORD v232[2]; // [rsp+130h] [rbp-1C0h] BYREF
  __m128i *v233; // [rsp+140h] [rbp-1B0h] BYREF
  __int64 v234; // [rsp+148h] [rbp-1A8h]
  __m128i v235; // [rsp+150h] [rbp-1A0h] BYREF
  __m128i *v236; // [rsp+160h] [rbp-190h] BYREF
  __int64 v237; // [rsp+168h] [rbp-188h]
  __m128i v238; // [rsp+170h] [rbp-180h] BYREF
  __m128i *v239; // [rsp+180h] [rbp-170h] BYREF
  __int64 v240; // [rsp+188h] [rbp-168h]
  __m128i v241; // [rsp+190h] [rbp-160h] BYREF
  _QWORD v242[2]; // [rsp+1A0h] [rbp-150h] BYREF
  int v243; // [rsp+1B0h] [rbp-140h] BYREF
  _QWORD *v244; // [rsp+1B8h] [rbp-138h]
  int *v245; // [rsp+1C0h] [rbp-130h]
  int *v246; // [rsp+1C8h] [rbp-128h]
  __int64 v247; // [rsp+1D0h] [rbp-120h]
  __int64 v248; // [rsp+1D8h] [rbp-118h]
  __int64 v249; // [rsp+1E0h] [rbp-110h]
  __int64 v250; // [rsp+1E8h] [rbp-108h]
  __int64 v251; // [rsp+1F0h] [rbp-100h]
  __int64 v252; // [rsp+1F8h] [rbp-F8h]
  _QWORD v253[2]; // [rsp+200h] [rbp-F0h] BYREF
  int v254; // [rsp+210h] [rbp-E0h] BYREF
  _QWORD *v255; // [rsp+218h] [rbp-D8h]
  int *v256; // [rsp+220h] [rbp-D0h]
  int *v257; // [rsp+228h] [rbp-C8h]
  __int64 v258; // [rsp+230h] [rbp-C0h]
  __int64 v259; // [rsp+238h] [rbp-B8h]
  __int64 v260; // [rsp+240h] [rbp-B0h]
  __int64 v261; // [rsp+248h] [rbp-A8h]
  __int64 v262; // [rsp+250h] [rbp-A0h]
  __int64 v263; // [rsp+258h] [rbp-98h]
  __m128i *v264; // [rsp+260h] [rbp-90h] BYREF
  __int64 v265; // [rsp+268h] [rbp-88h]
  __m128i v266; // [rsp+270h] [rbp-80h] BYREF
  __m128i *v267; // [rsp+280h] [rbp-70h]
  __m128i *v268; // [rsp+288h] [rbp-68h]
  __int64 v269; // [rsp+290h] [rbp-60h]
  __int64 v270; // [rsp+298h] [rbp-58h]
  __int64 v271; // [rsp+2A0h] [rbp-50h]
  __int64 v272; // [rsp+2A8h] [rbp-48h]
  __int64 v273; // [rsp+2B0h] [rbp-40h]
  __int64 v274; // [rsp+2B8h] [rbp-38h]

  v245 = &v243;
  v246 = &v243;
  v242[0] = 0;
  v243 = 0;
  v244 = 0;
  v247 = 0;
  v248 = 0;
  v249 = 0;
  v250 = 0;
  v251 = 0;
  v252 = 0;
  v253[0] = 0;
  v254 = 0;
  v255 = 0;
  v256 = &v254;
  v257 = &v254;
  v258 = 0;
  v259 = 0;
  v260 = 0;
  v261 = 0;
  v262 = 0;
  v263 = 0;
  if ( a3 )
    sub_12972A0((__int64)v242, a3);
  v6 = *(_QWORD *)(a2 + 16);
  if ( *(_DWORD *)(v6 + 12) == 1 )
  {
    v174 = *(_QWORD *)(v6 + 24);
    if ( (unsigned __int8)sub_127B3A0(v174) )
    {
      sub_15606E0(v253, 40);
    }
    else if ( (unsigned __int8)sub_127B3E0(v174) )
    {
      sub_15606E0(v253, 58);
    }
  }
  if ( (unsigned __int8)sub_1560CB0(v253) )
  {
    v170 = sub_1560CD0(*(_QWORD *)(a1 + 360), 0, v253);
    v264 = (__m128i *)v170;
    v171 = a4[1];
    if ( v171 == a4[2] )
    {
      sub_129A5D0(a4, v171, &v264);
    }
    else
    {
      if ( v171 )
      {
        *(_QWORD *)v171 = v170;
        v171 = a4[1];
      }
      a4[1] = v171 + 8;
    }
  }
  v7 = *(unsigned int *)(a2 + 8);
  v8 = *(_QWORD *)(a2 + 16);
  v9 = 1;
  v10 = v8 + 40;
  v209 = v8 + 8 * (5 * v7 + 5);
  if ( v209 != v8 + 40 )
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)(v10 + 24);
      v266.m128i_i32[0] = 0;
      v264 = 0;
      v213 = v11;
      v266.m128i_i64[1] = 0;
      v267 = &v266;
      v268 = &v266;
      v269 = 0;
      v270 = 0;
      v271 = 0;
      v272 = 0;
      v273 = 0;
      v274 = 0;
      if ( !a3 )
        break;
      if ( (*(_BYTE *)(a3 + 198) & 0x20) == 0 && *(_BYTE *)(v10 + 34)
        || *(_BYTE *)(v10 + 32)
        || dword_4D04628 && (*(_BYTE *)(a3 + 198) & 0x20) != 0 && (unsigned int)sub_8D2E30(v11) )
      {
        goto LABEL_208;
      }
LABEL_19:
      v12 = *(_DWORD *)(v10 + 12);
      if ( v12 != 1 )
      {
        if ( v12 == 2 )
        {
          if ( *(_BYTE *)(v10 + 16) )
            sub_15606E0(&v264, 6);
          sub_1560C00(&v264, *(unsigned int *)(v10 + 8));
          sub_1560700(v242, 37);
          sub_1560700(v242, 36);
        }
LABEL_11:
        if ( (unsigned __int8)sub_1560CB0(&v264) )
          goto LABEL_22;
        goto LABEL_12;
      }
      if ( !(unsigned __int8)sub_127B3A0(v213) )
      {
        if ( (unsigned __int8)sub_127B3E0(v213) )
          sub_15606E0(&v264, 58);
        goto LABEL_11;
      }
      sub_15606E0(&v264, 40);
      if ( (unsigned __int8)sub_1560CB0(&v264) )
      {
LABEL_22:
        v13 = sub_1560CD0(*(_QWORD *)(a1 + 360), v9, &v264);
        v239 = (__m128i *)v13;
        v14 = a4[1];
        if ( v14 == a4[2] )
        {
          sub_129A5D0(a4, v14, &v239);
        }
        else
        {
          if ( v14 )
          {
            *(_QWORD *)v14 = v13;
            v14 = a4[1];
          }
          a4[1] = v14 + 8;
        }
      }
LABEL_12:
      ++v9;
      v10 += 40;
      sub_12973F0((_QWORD *)v266.m128i_i64[1]);
      if ( v10 == v209 )
        goto LABEL_30;
    }
    if ( !*(_BYTE *)(v10 + 32) )
      goto LABEL_19;
LABEL_208:
    sub_15606E0(&v264, 20);
    goto LABEL_19;
  }
LABEL_30:
  if ( !a3 )
    goto LABEL_204;
  v15 = *(_DWORD **)(a3 + 328);
  if ( !v15 || (int)v15[10] <= 0 )
    goto LABEL_204;
  sub_1562A10(v242, "nvvm.blocksareclusters", 22, 0, 0);
  v17 = v15[12];
  v18 = v17 >> 31;
  v19 = v17 ^ (v17 >> 31);
  v20 = (unsigned int)v17 >> 31;
  v21 = (int)v15[12] < 0;
  v22 = v19 - v18;
  if ( v22 <= 9 )
  {
    v236 = &v238;
    sub_2240A50(&v236, v20 + 1, 45, &v238, v16);
    v30 = &v236->m128i_i8[v21];
LABEL_48:
    *v30 = v22 + 48;
    goto LABEL_49;
  }
  if ( v22 <= 0x63 )
  {
    v236 = &v238;
    sub_2240A50(&v236, v20 + 2, 45, &v238, v16);
    v30 = &v236->m128i_i8[v21];
  }
  else
  {
    if ( v22 <= 0x3E7 )
    {
      v28 = 2;
      v25 = 3;
      v23 = v22;
    }
    else
    {
      v23 = v22;
      v24 = v22;
      if ( v22 <= 0x270F )
      {
        v28 = 3;
        v25 = 4;
      }
      else
      {
        v16 = 0x346DC5D63886594BLL;
        LODWORD(v25) = 1;
        do
        {
          v26 = v24;
          v27 = v25;
          v25 = (unsigned int)(v25 + 4);
          v24 /= 0x2710u;
          if ( v26 <= 0x1869F )
          {
            v28 = v27 + 3;
            goto LABEL_43;
          }
          if ( (unsigned int)v24 <= 0x63 )
          {
            v211 = v25;
            v236 = &v238;
            sub_2240A50(&v236, v27 + v20 + 5, 45, &v238, 0x346DC5D63886594BLL);
            v30 = &v236->m128i_i8[v21];
            v28 = v211;
            goto LABEL_44;
          }
          if ( (unsigned int)v24 <= 0x3E7 )
          {
            v25 = (unsigned int)(v27 + 6);
            v28 = v27 + 5;
            goto LABEL_43;
          }
        }
        while ( (unsigned int)v24 > 0x270F );
        v25 = (unsigned int)(v27 + 7);
        v28 = v27 + 6;
      }
    }
LABEL_43:
    v236 = &v238;
    sub_2240A50(&v236, (unsigned int)v25 + v20, 45, v25, v16);
    v30 = &v236->m128i_i8[v21];
LABEL_44:
    for ( i = v23; ; i = v22 )
    {
      v32 = (unsigned __int64)(1374389535 * i) >> 37;
      v33 = v22 - 100 * v32;
      v34 = v22;
      v22 = v32;
      v35 = (unsigned int)(2 * v33);
      v36 = (unsigned int)(v35 + 1);
      LOBYTE(v35) = a00010203040506[v35];
      v30[v28] = a00010203040506[v36];
      v37 = v28 - 1;
      v28 -= 2;
      v30[v37] = v35;
      if ( (unsigned int)v34 <= 0x270F )
        break;
    }
    if ( (unsigned int)v34 <= 0x3E7 )
      goto LABEL_48;
  }
  v179 = 2 * v22;
  v180 = a00010203040506[v179];
  v30[1] = a00010203040506[(unsigned int)(v179 + 1)];
  *v30 = v180;
LABEL_49:
  v38 = (int)v15[11] < 0;
  v39 = (int)v15[11] < 0;
  v40 = abs32(v15[11]);
  if ( v40 <= 9 )
  {
    v219 = v221;
    sub_2240A50(&v219, (unsigned int)(v39 + 1), 45, v34, v29);
    v48 = (char *)v219 + v38;
LABEL_64:
    *v48 = v40 + 48;
    goto LABEL_65;
  }
  if ( v40 <= 0x63 )
  {
    v219 = v221;
    sub_2240A50(&v219, (unsigned int)(v39 + 2), 45, v34, v29);
    v48 = (char *)v219 + v38;
  }
  else
  {
    if ( v40 <= 0x3E7 )
    {
      v46 = 2;
      v43 = 3;
      v41 = v40;
    }
    else
    {
      v41 = v40;
      v42 = v40;
      if ( v40 <= 0x270F )
      {
        v46 = 3;
        v43 = 4;
      }
      else
      {
        LODWORD(v43) = 1;
        do
        {
          v44 = v42;
          v45 = (unsigned int)v43;
          v43 = (unsigned int)(v43 + 4);
          v42 /= 0x2710u;
          if ( v44 <= 0x1869F )
          {
            v46 = (unsigned int)(v45 + 3);
            goto LABEL_59;
          }
          if ( (unsigned int)v42 <= 0x63 )
          {
            v212 = v43;
            v219 = v221;
            sub_2240A50(&v219, (unsigned int)(v45 + v39 + 5), 45, v45, v43);
            v48 = (char *)v219 + v38;
            v47 = v212;
            goto LABEL_60;
          }
          if ( (unsigned int)v42 <= 0x3E7 )
          {
            v43 = (unsigned int)(v45 + 6);
            v46 = (unsigned int)(v45 + 5);
            goto LABEL_59;
          }
        }
        while ( (unsigned int)v42 > 0x270F );
        v43 = (unsigned int)(v45 + 7);
        v46 = (unsigned int)(v45 + 6);
      }
    }
LABEL_59:
    v210 = v46;
    v219 = v221;
    sub_2240A50(&v219, (unsigned int)(v43 + v39), 45, v46, v43);
    v47 = v210;
    v48 = (char *)v219 + v38;
LABEL_60:
    for ( j = v41; ; j = v40 )
    {
      v50 = (unsigned __int64)(1374389535 * j) >> 37;
      v51 = v40 - 100 * v50;
      v52 = v40;
      v40 = v50;
      v53 = (unsigned int)(2 * v51);
      v54 = (unsigned int)(v53 + 1);
      LOBYTE(v53) = a00010203040506[v53];
      v55 = (unsigned __int8)a00010203040506[v54];
      v48[v47] = v55;
      v56 = (unsigned int)(v47 - 1);
      v47 -= 2;
      v48[v56] = v53;
      if ( v52 <= 0x270F )
        break;
    }
    if ( v52 <= 0x3E7 )
      goto LABEL_64;
  }
  v182 = 2 * v40;
  v183 = a00010203040506[v182];
  v48[1] = a00010203040506[(unsigned int)(v182 + 1)];
  *v48 = v183;
LABEL_65:
  v57 = (int)v15[10] < 0;
  v58 = (int)v15[10] < 0;
  v59 = abs32(v15[10]);
  if ( (unsigned int)v59 > 9 )
  {
    if ( (unsigned int)v59 <= 0x63 )
    {
      v198 = v59;
      v214[0] = v215;
      sub_2240A50(v214, (unsigned int)(v58 + 2), 45, v59, v55);
      v66 = v198;
      v68 = (char *)(v214[0] + v57);
    }
    else
    {
      if ( (unsigned int)v59 <= 0x3E7 )
      {
        v65 = 2;
        v62 = 3;
        v60 = (unsigned int)v59;
      }
      else
      {
        v60 = (unsigned int)v59;
        v61 = (unsigned int)v59;
        if ( (unsigned int)v59 <= 0x270F )
        {
          v65 = 3;
          v62 = 4;
        }
        else
        {
          v62 = 1;
          do
          {
            v63 = v61;
            v64 = v62;
            v62 += 4;
            v61 /= 0x2710u;
            if ( v63 <= 0x1869F )
            {
              v65 = (unsigned int)(v64 + 3);
              goto LABEL_75;
            }
            if ( (unsigned int)v61 <= 0x63 )
            {
              v195 = v62;
              v199 = v59;
              v214[0] = v215;
              sub_2240A50(v214, (unsigned int)(v64 + v58 + 5), 45, v59, v64);
              v66 = v199;
              v68 = (char *)(v214[0] + v57);
              v67 = v195;
              goto LABEL_76;
            }
            if ( (unsigned int)v61 <= 0x3E7 )
            {
              v62 = v64 + 6;
              v65 = (unsigned int)(v64 + 5);
              goto LABEL_75;
            }
          }
          while ( (unsigned int)v61 > 0x270F );
          v62 = v64 + 7;
          v65 = (unsigned int)(v64 + 6);
        }
      }
LABEL_75:
      v193 = v65;
      v197 = v59;
      v214[0] = v215;
      sub_2240A50(v214, v62 + v58, 45, v59, v65);
      v66 = v197;
      v67 = v193;
      v68 = (char *)(v214[0] + v57);
LABEL_76:
      for ( k = v60; ; k = v66 )
      {
        v70 = (unsigned __int64)(1374389535 * k) >> 37;
        v71 = v66 - 100 * v70;
        v72 = v66;
        v66 = v70;
        v73 = (unsigned int)(2 * v71);
        v74 = (unsigned int)(v73 + 1);
        LOBYTE(v73) = a00010203040506[v73];
        v68[v67] = a00010203040506[v74];
        v75 = v67 - 1;
        v67 -= 2;
        v68[v75] = v73;
        if ( v72 <= 0x270F )
          break;
      }
      if ( v72 <= 0x3E7 )
        goto LABEL_80;
    }
    v76 = 2 * v66;
    v181 = a00010203040506[v76];
    v68[1] = a00010203040506[(unsigned int)(v76 + 1)];
    *v68 = v181;
    goto LABEL_81;
  }
  v200 = v59;
  v214[0] = v215;
  sub_2240A50(v214, (unsigned int)(v58 + 1), 45, v59, v55);
  v66 = v200;
  v68 = (char *)(v214[0] + v57);
LABEL_80:
  v76 = v66 + 48;
  *v68 = v76;
LABEL_81:
  if ( v214[1] == 0x3FFFFFFFFFFFFFFFLL )
    goto LABEL_286;
  v77 = (__m128i *)sub_2241490(v214, ",", 1, v76);
  v216 = &v218;
  if ( (__m128i *)v77->m128i_i64[0] == &v77[1] )
  {
    v218 = _mm_loadu_si128(v77 + 1);
  }
  else
  {
    v216 = (__m128i *)v77->m128i_i64[0];
    v218.m128i_i64[0] = v77[1].m128i_i64[0];
  }
  v78 = v77->m128i_i64[1];
  v77[1].m128i_i8[0] = 0;
  v217 = v78;
  v77->m128i_i64[0] = (__int64)v77[1].m128i_i64;
  v79 = v216;
  v77->m128i_i64[1] = 0;
  v80 = 15;
  v81 = 15;
  if ( v79 != &v218 )
    v81 = v218.m128i_i64[0];
  v82 = v217 + v220;
  if ( v217 + v220 <= v81 )
    goto LABEL_90;
  if ( v219 != v221 )
    v80 = v221[0];
  if ( v82 <= v80 )
    v83 = (__m128i *)sub_2241130(&v219, 0, 0, v79, v217);
  else
LABEL_90:
    v83 = (__m128i *)sub_2241490(&v216, v219, v220, v82);
  v222 = &v224;
  if ( (__m128i *)v83->m128i_i64[0] == &v83[1] )
  {
    v224 = _mm_loadu_si128(v83 + 1);
  }
  else
  {
    v222 = (__m128i *)v83->m128i_i64[0];
    v224.m128i_i64[0] = v83[1].m128i_i64[0];
  }
  v84 = v83->m128i_i64[1];
  v223 = v84;
  v83->m128i_i64[0] = (__int64)v83[1].m128i_i64;
  v83->m128i_i64[1] = 0;
  v83[1].m128i_i8[0] = 0;
  if ( v223 == 0x3FFFFFFFFFFFFFFFLL )
LABEL_286:
    sub_4262D8((__int64)"basic_string::append");
  v85 = (__m128i *)sub_2241490(&v222, ",", 1, v84);
  v239 = &v241;
  if ( (__m128i *)v85->m128i_i64[0] == &v85[1] )
  {
    v241 = _mm_loadu_si128(v85 + 1);
  }
  else
  {
    v239 = (__m128i *)v85->m128i_i64[0];
    v241.m128i_i64[0] = v85[1].m128i_i64[0];
  }
  v240 = v85->m128i_i64[1];
  v85->m128i_i64[0] = (__int64)v85[1].m128i_i64;
  v85->m128i_i64[1] = 0;
  v85[1].m128i_i8[0] = 0;
  v86 = 15;
  v87 = 15;
  if ( v239 != &v241 )
    v87 = v241.m128i_i64[0];
  v88 = v240 + v237;
  if ( v240 + v237 <= v87 )
    goto LABEL_102;
  if ( v236 != &v238 )
    v86 = v238.m128i_i64[0];
  if ( v88 <= v86 )
  {
    v184 = (__m128i *)sub_2241130(&v236, 0, 0, v239, v240);
    v264 = &v266;
    if ( (__m128i *)v184->m128i_i64[0] == &v184[1] )
    {
      v266 = _mm_loadu_si128(v184 + 1);
    }
    else
    {
      v264 = (__m128i *)v184->m128i_i64[0];
      v266.m128i_i64[0] = v184[1].m128i_i64[0];
    }
    v265 = v184->m128i_i64[1];
    v184->m128i_i64[0] = (__int64)v184[1].m128i_i64;
    v184->m128i_i64[1] = 0;
    v184[1].m128i_i8[0] = 0;
  }
  else
  {
LABEL_102:
    v89 = (__m128i *)sub_2241490(&v239, v236, v237, v88);
    v264 = &v266;
    if ( (__m128i *)v89->m128i_i64[0] == &v89[1] )
    {
      v266 = _mm_loadu_si128(v89 + 1);
    }
    else
    {
      v264 = (__m128i *)v89->m128i_i64[0];
      v266.m128i_i64[0] = v89[1].m128i_i64[0];
    }
    v265 = v89->m128i_i64[1];
    v89->m128i_i64[0] = (__int64)v89[1].m128i_i64;
    v89->m128i_i64[1] = 0;
    v89[1].m128i_i8[0] = 0;
  }
  sub_1562A10(v242, "nvvm.reqntid", 12, v264, v265);
  if ( v264 != &v266 )
    j_j___libc_free_0(v264, v266.m128i_i64[0] + 1);
  if ( v239 != &v241 )
    j_j___libc_free_0(v239, v241.m128i_i64[0] + 1);
  if ( v222 != &v224 )
    j_j___libc_free_0(v222, v224.m128i_i64[0] + 1);
  if ( v216 != &v218 )
    j_j___libc_free_0(v216, v218.m128i_i64[0] + 1);
  if ( (_QWORD *)v214[0] != v215 )
    j_j___libc_free_0(v214[0], v215[0] + 1LL);
  if ( v219 != v221 )
    j_j___libc_free_0(v219, v221[0] + 1LL);
  if ( v236 != &v238 )
    j_j___libc_free_0(v236, v238.m128i_i64[0] + 1);
  v92 = (int)v15[7] < 0;
  v93 = (int)v15[7] < 0;
  v94 = abs32(v15[7]);
  if ( v94 <= 9 )
  {
    v239 = &v241;
    sub_2240A50(&v239, (unsigned int)(v93 + 1), 45, v90, v91);
    v102 = &v239->m128i_i8[v92];
LABEL_134:
    *v102 = v94 + 48;
    goto LABEL_135;
  }
  if ( v94 <= 0x63 )
  {
    v239 = &v241;
    sub_2240A50(&v239, (unsigned int)(v93 + 2), 45, v90, v91);
    v102 = &v239->m128i_i8[v92];
  }
  else
  {
    if ( v94 <= 0x3E7 )
    {
      v100 = 2;
      v97 = 3;
      v95 = v94;
    }
    else
    {
      v95 = v94;
      v96 = v94;
      if ( v94 <= 0x270F )
      {
        v100 = 3;
        v97 = 4;
      }
      else
      {
        LODWORD(v97) = 1;
        do
        {
          v98 = v96;
          v99 = (unsigned int)v97;
          v97 = (unsigned int)(v97 + 4);
          v96 /= 0x2710u;
          if ( v98 <= 0x1869F )
          {
            v100 = (unsigned int)(v99 + 3);
            goto LABEL_129;
          }
          if ( (unsigned int)v96 <= 0x63 )
          {
            v204 = v97;
            v239 = &v241;
            sub_2240A50(&v239, (unsigned int)(v99 + v93 + 5), 45, v99, v97);
            v102 = &v239->m128i_i8[v92];
            LODWORD(v101) = v204;
            goto LABEL_130;
          }
          if ( (unsigned int)v96 <= 0x3E7 )
          {
            v97 = (unsigned int)(v99 + 6);
            v100 = (unsigned int)(v99 + 5);
            goto LABEL_129;
          }
        }
        while ( (unsigned int)v96 > 0x270F );
        v97 = (unsigned int)(v99 + 7);
        v100 = (unsigned int)(v99 + 6);
      }
    }
LABEL_129:
    v201 = v100;
    v239 = &v241;
    sub_2240A50(&v239, (unsigned int)(v97 + v93), 45, v100, v97);
    LODWORD(v101) = v201;
    v102 = &v239->m128i_i8[v92];
LABEL_130:
    for ( m = v95; ; m = v94 )
    {
      v104 = (unsigned __int64)(1374389535 * m) >> 37;
      v105 = v94 - 100 * v104;
      v106 = v94;
      v94 = v104;
      v107 = (unsigned int)(2 * v105);
      v108 = (unsigned int)(v107 + 1);
      LOBYTE(v107) = a00010203040506[v107];
      v102[(unsigned int)v101] = a00010203040506[v108];
      v109 = (unsigned int)(v101 - 1);
      v101 = (unsigned int)(v101 - 2);
      v102[v109] = v107;
      if ( v106 <= 0x270F )
        break;
    }
    if ( v106 <= 0x3E7 )
      goto LABEL_134;
  }
  v177 = 2 * v94;
  v178 = a00010203040506[v177];
  v102[1] = a00010203040506[(unsigned int)(v177 + 1)];
  *v102 = v178;
LABEL_135:
  v110 = v15[6];
  v111 = v110 >> 31;
  v112 = v110 ^ (v110 >> 31);
  v113 = (unsigned int)v110 >> 31;
  v114 = (int)v15[6] < 0;
  v115 = (unsigned int)(v112 - v111);
  if ( (unsigned int)v115 <= 9 )
  {
    v206 = v115;
    v230 = v232;
    sub_2240A50(&v230, v113 + 1, 45, v101, v115);
    v123 = v206;
    v125 = (char *)v230 + v114;
LABEL_149:
    v132 = v123 + 48;
    *v125 = v132;
    goto LABEL_150;
  }
  if ( (unsigned int)v115 <= 0x63 )
  {
    v203 = v115;
    v230 = v232;
    sub_2240A50(&v230, v113 + 2, 45, v101, v115);
    v123 = v203;
    v125 = (char *)v230 + v114;
  }
  else
  {
    if ( (unsigned int)v115 <= 0x3E7 )
    {
      v121 = 2;
      v118 = 3;
      v116 = (unsigned int)v115;
    }
    else
    {
      v116 = (unsigned int)v115;
      v117 = (unsigned int)v115;
      if ( (unsigned int)v115 <= 0x270F )
      {
        v121 = 3;
        v118 = 4;
      }
      else
      {
        v118 = 1;
        do
        {
          v119 = v117;
          v120 = v118;
          v118 += 4;
          v117 /= 0x2710u;
          if ( v119 <= 0x1869F )
          {
            v121 = v120 + 3;
            goto LABEL_145;
          }
          if ( (unsigned int)v117 <= 0x63 )
          {
            v191 = v115;
            v196 = v118;
            v205 = (unsigned int)v115;
            v230 = v232;
            sub_2240A50(&v230, v120 + v113 + 5, 45, (unsigned int)v115, v115);
            v124 = v205;
            v123 = v191;
            v125 = (char *)v230 + v114;
            v122 = v196;
            goto LABEL_147;
          }
          if ( (unsigned int)v117 <= 0x3E7 )
          {
            v118 = v120 + 6;
            v121 = v120 + 5;
            goto LABEL_145;
          }
        }
        while ( (unsigned int)v117 > 0x270F );
        v118 = v120 + 7;
        v121 = v120 + 6;
      }
    }
LABEL_145:
    v187 = v116;
    v194 = v115;
    v202 = v121;
    v230 = v232;
    sub_2240A50(&v230, v118 + v113, 45, v116, v115);
    v122 = v202;
    v123 = v194;
    v124 = v187;
    v125 = (char *)v230 + v114;
LABEL_147:
    while ( 1 )
    {
      v126 = (unsigned __int64)(1374389535 * v124) >> 37;
      v127 = v123 - 100 * v126;
      v128 = v123;
      v123 = v126;
      v129 = (unsigned int)(2 * v127);
      v130 = (unsigned int)(v129 + 1);
      LOBYTE(v129) = a00010203040506[v129];
      v125[v122] = a00010203040506[v130];
      v131 = (unsigned int)(v122 - 1);
      v122 -= 2;
      v125[v131] = v129;
      if ( v128 <= 0x270F )
        break;
      v124 = v123;
    }
    if ( v128 <= 0x3E7 )
      goto LABEL_149;
  }
  v132 = 2 * v123;
  v176 = a00010203040506[v132];
  v125[1] = a00010203040506[(unsigned int)(v132 + 1)];
  *v125 = v176;
LABEL_150:
  v133 = v15[5];
  v134 = v133 >> 31;
  v135 = v133 ^ (v133 >> 31);
  v136 = (unsigned int)v133 >> 31;
  v137 = v133 < 0;
  v138 = (unsigned int)(v135 - v134);
  if ( (unsigned int)v138 > 9 )
  {
    if ( (unsigned int)v138 <= 0x63 )
    {
      v189 = v138;
      v225[0] = v226;
      sub_2240A50(v225, v136 + 2, 45, v138, v132);
      v146 = v189;
      v147 = (char *)(v225[0] + v137);
    }
    else
    {
      if ( (unsigned int)v138 <= 0x3E7 )
      {
        v144 = 2;
        v141 = 3;
        v139 = (unsigned int)v138;
      }
      else
      {
        v139 = (unsigned int)v138;
        v140 = (unsigned int)v138;
        if ( (unsigned int)v138 <= 0x270F )
        {
          v144 = 3;
          v141 = 4;
        }
        else
        {
          v141 = 1;
          do
          {
            v142 = v140;
            v143 = v141;
            v141 += 4;
            v140 /= 0x2710u;
            if ( v142 <= 0x1869F )
            {
              v144 = (unsigned int)(v143 + 3);
              goto LABEL_160;
            }
            if ( (unsigned int)v140 <= 0x63 )
            {
              v186 = v138;
              v190 = v141;
              v225[0] = v226;
              sub_2240A50(v225, (unsigned int)v143 + v136 + 5, 45, v138, v143);
              v146 = v186;
              v147 = (char *)(v225[0] + v137);
              v145 = v190;
              goto LABEL_161;
            }
            if ( (unsigned int)v140 <= 0x3E7 )
            {
              v141 = v143 + 6;
              v144 = (unsigned int)(v143 + 5);
              goto LABEL_160;
            }
          }
          while ( (unsigned int)v140 > 0x270F );
          v141 = v143 + 7;
          v144 = (unsigned int)(v143 + 6);
        }
      }
LABEL_160:
      v185 = v138;
      v188 = v144;
      v225[0] = v226;
      sub_2240A50(v225, v141 + v136, 45, v138, v144);
      v145 = v188;
      v146 = v185;
      v147 = (char *)(v225[0] + v137);
LABEL_161:
      for ( n = v139; ; n = v146 )
      {
        v149 = (unsigned __int64)(1374389535 * n) >> 37;
        v150 = v146 - 100 * v149;
        v151 = v146;
        v146 = v149;
        v152 = (unsigned int)(2 * v150);
        v153 = (unsigned int)(v152 + 1);
        LOBYTE(v152) = a00010203040506[v152];
        v147[v145] = a00010203040506[v153];
        v154 = v145 - 1;
        v145 -= 2;
        v147[v154] = v152;
        if ( v151 <= 0x270F )
          break;
      }
      if ( v151 <= 0x3E7 )
        goto LABEL_165;
    }
    v155 = 2 * v146;
    v175 = a00010203040506[v155];
    v147[1] = a00010203040506[(unsigned int)(v155 + 1)];
    *v147 = v175;
    goto LABEL_166;
  }
  v192 = v138;
  v225[0] = v226;
  sub_2240A50(v225, v136 + 1, 45, v138, v132);
  v146 = v192;
  v147 = (char *)(v225[0] + v137);
LABEL_165:
  v155 = v146 + 48;
  *v147 = v155;
LABEL_166:
  if ( v225[1] == 0x3FFFFFFFFFFFFFFFLL )
    goto LABEL_286;
  v156 = (__m128i *)sub_2241490(v225, ",", 1, v155);
  v227 = &v229;
  if ( (__m128i *)v156->m128i_i64[0] == &v156[1] )
  {
    v229 = _mm_loadu_si128(v156 + 1);
  }
  else
  {
    v227 = (__m128i *)v156->m128i_i64[0];
    v229.m128i_i64[0] = v156[1].m128i_i64[0];
  }
  v157 = v156->m128i_i64[1];
  v156[1].m128i_i8[0] = 0;
  v228 = v157;
  v156->m128i_i64[0] = (__int64)v156[1].m128i_i64;
  v158 = v227;
  v156->m128i_i64[1] = 0;
  v159 = 15;
  v160 = 15;
  if ( v158 != &v229 )
    v160 = v229.m128i_i64[0];
  v161 = v228 + v231;
  if ( v228 + v231 <= v160 )
    goto LABEL_175;
  if ( v230 != v232 )
    v159 = v232[0];
  if ( v161 <= v159 )
    v162 = (__m128i *)sub_2241130(&v230, 0, 0, v158, v228);
  else
LABEL_175:
    v162 = (__m128i *)sub_2241490(&v227, v230, v231, v161);
  v233 = &v235;
  if ( (__m128i *)v162->m128i_i64[0] == &v162[1] )
  {
    v235 = _mm_loadu_si128(v162 + 1);
  }
  else
  {
    v233 = (__m128i *)v162->m128i_i64[0];
    v235.m128i_i64[0] = v162[1].m128i_i64[0];
  }
  v163 = v162->m128i_i64[1];
  v234 = v163;
  v162->m128i_i64[0] = (__int64)v162[1].m128i_i64;
  v162->m128i_i64[1] = 0;
  v162[1].m128i_i8[0] = 0;
  if ( v234 == 0x3FFFFFFFFFFFFFFFLL )
    goto LABEL_286;
  v164 = (__m128i *)sub_2241490(&v233, ",", 1, v163);
  v236 = &v238;
  if ( (__m128i *)v164->m128i_i64[0] == &v164[1] )
  {
    v238 = _mm_loadu_si128(v164 + 1);
  }
  else
  {
    v236 = (__m128i *)v164->m128i_i64[0];
    v238.m128i_i64[0] = v164[1].m128i_i64[0];
  }
  v237 = v164->m128i_i64[1];
  v164->m128i_i64[0] = (__int64)v164[1].m128i_i64;
  v164->m128i_i64[1] = 0;
  v164[1].m128i_i8[0] = 0;
  v165 = 15;
  v166 = 15;
  if ( v236 != &v238 )
    v166 = v238.m128i_i64[0];
  v167 = v237 + v240;
  if ( v237 + v240 <= v166 )
    goto LABEL_187;
  if ( v239 != &v241 )
    v165 = v241.m128i_i64[0];
  if ( v167 <= v165 )
    v168 = (__m128i *)sub_2241130(&v239, 0, 0, v236, v237);
  else
LABEL_187:
    v168 = (__m128i *)sub_2241490(&v236, v239, v240, v167);
  v264 = &v266;
  if ( (__m128i *)v168->m128i_i64[0] == &v168[1] )
  {
    v266 = _mm_loadu_si128(v168 + 1);
  }
  else
  {
    v264 = (__m128i *)v168->m128i_i64[0];
    v266.m128i_i64[0] = v168[1].m128i_i64[0];
  }
  v265 = v168->m128i_i64[1];
  v168->m128i_i64[0] = (__int64)v168[1].m128i_i64;
  v168->m128i_i64[1] = 0;
  v168[1].m128i_i8[0] = 0;
  sub_1562A10(v242, "nvvm.cluster_dim", 16, v264, v265);
  if ( v264 != &v266 )
    j_j___libc_free_0(v264, v266.m128i_i64[0] + 1);
  if ( v236 != &v238 )
    j_j___libc_free_0(v236, v238.m128i_i64[0] + 1);
  if ( v233 != &v235 )
    j_j___libc_free_0(v233, v235.m128i_i64[0] + 1);
  if ( v227 != &v229 )
    j_j___libc_free_0(v227, v229.m128i_i64[0] + 1);
  if ( (_QWORD *)v225[0] != v226 )
    j_j___libc_free_0(v225[0], v226[0] + 1LL);
  if ( v230 != v232 )
    j_j___libc_free_0(v230, v232[0] + 1LL);
  if ( v239 != &v241 )
    j_j___libc_free_0(v239, v241.m128i_i64[0] + 1);
LABEL_204:
  if ( (unsigned __int8)sub_1560CB0(v242) )
  {
    v172 = sub_1560CD0(*(_QWORD *)(a1 + 360), 0xFFFFFFFFLL, v242);
    v264 = (__m128i *)v172;
    v173 = a4[1];
    if ( v173 == a4[2] )
    {
      sub_129A5D0(a4, v173, &v264);
    }
    else
    {
      if ( v173 )
      {
        *(_QWORD *)v173 = v172;
        v173 = a4[1];
      }
      a4[1] = v173 + 8;
    }
  }
  sub_12973F0(v255);
  return sub_12973F0(v244);
}
