// Function: sub_2F81900
// Address: 0x2f81900
//
__int64 __fastcall sub_2F81900(__int64 *a1, __int64 a2)
{
  _QWORD *v2; // r14
  _QWORD *i; // r12
  _QWORD *v4; // rax
  _QWORD *v5; // r14
  _QWORD *v6; // r15
  _QWORD *v7; // rax
  _BYTE *v8; // rbx
  _BYTE *v9; // r12
  int v10; // eax
  _QWORD *v11; // rdi
  __int64 v13; // rax
  unsigned int v14; // ebx
  __int64 v15; // rax
  int v16; // r8d
  int v17; // ecx
  bool v18; // zf
  __int64 v19; // r8
  __int64 v20; // r9
  int v21; // edx
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  unsigned int v28; // r13d
  __int64 v29; // rax
  __int64 v30; // r12
  __int64 v31; // r15
  char v32; // r14
  __int64 *v33; // r8
  __int64 v34; // r9
  char v35; // r10
  __int64 v36; // rax
  __int64 v37; // r11
  unsigned __int64 v38; // rdx
  char v39; // cl
  unsigned __int64 v40; // rdx
  char *v41; // rbx
  size_t v42; // r13
  _QWORD *v43; // rax
  __m128i v44; // kr00_16
  __int64 v45; // rbx
  _QWORD **v46; // r13
  unsigned int *v47; // rax
  unsigned int *v48; // rsi
  unsigned int v49; // edx
  unsigned __int64 v50; // r13
  __int64 v51; // r8
  unsigned __int64 v52; // rcx
  __int64 v53; // rbx
  __int64 **v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 *v57; // rax
  char v58; // r10
  __int64 v59; // rsi
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 v62; // r13
  unsigned int *v63; // rax
  int v64; // edi
  unsigned int *v65; // rdx
  int v66; // edx
  __int64 v67; // r13
  __int64 v68; // rax
  unsigned __int64 v69; // rdx
  __int64 v70; // rax
  unsigned __int64 v71; // rdx
  __int64 v72; // rax
  unsigned __int64 v73; // rdx
  __int64 v74; // rax
  unsigned __int64 v75; // rdx
  __int64 v76; // rax
  unsigned __int64 v77; // rdx
  __int64 *v78; // rax
  __int64 v79; // rax
  __int64 v80; // rdx
  __int64 v81; // rbx
  __int64 v82; // rdx
  __int64 v83; // rax
  int v84; // ebx
  __int64 v85; // rax
  __int64 v86; // rdx
  int v87; // ecx
  _QWORD *v88; // rdx
  int v89; // esi
  __int64 v90; // r8
  __int64 v91; // rcx
  _QWORD *v92; // rax
  __int64 v93; // rcx
  __int64 v94; // r13
  __int64 v95; // r13
  _QWORD *v96; // rax
  __int64 *v97; // rax
  __int64 v98; // rax
  __int64 v99; // rax
  unsigned __int64 v100; // rdx
  __int64 v101; // r11
  __int64 v102; // rax
  char *v103; // r13
  int v104; // ecx
  __int64 *v105; // rsi
  unsigned __int64 v106; // r8
  char *v107; // rdi
  unsigned __int64 v108; // rsi
  __int64 v109; // r13
  char v110; // al
  char v111; // r10
  _BYTE *v112; // r13
  unsigned __int64 *v113; // rbx
  unsigned __int64 v114; // rdi
  unsigned int v115; // eax
  _QWORD *v116; // rdi
  unsigned __int64 v117; // rsi
  unsigned __int64 v118; // rbx
  __int64 v119; // [rsp+0h] [rbp-410h]
  char v120; // [rsp+Fh] [rbp-401h]
  __int64 v121; // [rsp+20h] [rbp-3F0h]
  __int64 v122; // [rsp+20h] [rbp-3F0h]
  _QWORD *v123; // [rsp+20h] [rbp-3F0h]
  char v125; // [rsp+38h] [rbp-3D8h]
  __int64 v126; // [rsp+38h] [rbp-3D8h]
  __int64 v127; // [rsp+38h] [rbp-3D8h]
  __int64 *v128; // [rsp+40h] [rbp-3D0h]
  char v129; // [rsp+40h] [rbp-3D0h]
  char v130; // [rsp+40h] [rbp-3D0h]
  char v131; // [rsp+40h] [rbp-3D0h]
  char v132; // [rsp+40h] [rbp-3D0h]
  char v133; // [rsp+40h] [rbp-3D0h]
  bool v134; // [rsp+48h] [rbp-3C8h]
  _QWORD *v135; // [rsp+58h] [rbp-3B8h]
  _QWORD *v136; // [rsp+60h] [rbp-3B0h]
  __int64 v137; // [rsp+60h] [rbp-3B0h]
  _QWORD *v138; // [rsp+68h] [rbp-3A8h]
  char v139; // [rsp+68h] [rbp-3A8h]
  char v140; // [rsp+68h] [rbp-3A8h]
  char v141; // [rsp+68h] [rbp-3A8h]
  char v142; // [rsp+68h] [rbp-3A8h]
  __int64 v143; // [rsp+68h] [rbp-3A8h]
  char v144; // [rsp+68h] [rbp-3A8h]
  char v145; // [rsp+68h] [rbp-3A8h]
  __int64 v146; // [rsp+80h] [rbp-390h]
  int v147; // [rsp+80h] [rbp-390h]
  __int64 v148; // [rsp+80h] [rbp-390h]
  char v149; // [rsp+80h] [rbp-390h]
  unsigned __int64 v150; // [rsp+80h] [rbp-390h]
  _BYTE *v151; // [rsp+80h] [rbp-390h]
  char v152; // [rsp+80h] [rbp-390h]
  char v153; // [rsp+80h] [rbp-390h]
  __int64 v154; // [rsp+80h] [rbp-390h]
  char v155; // [rsp+80h] [rbp-390h]
  char v156; // [rsp+80h] [rbp-390h]
  char v157; // [rsp+80h] [rbp-390h]
  char v158; // [rsp+80h] [rbp-390h]
  char v159; // [rsp+80h] [rbp-390h]
  __int64 v160; // [rsp+88h] [rbp-388h]
  int v161; // [rsp+98h] [rbp-378h] BYREF
  char v162; // [rsp+9Ch] [rbp-374h]
  __m128i v163; // [rsp+A0h] [rbp-370h] BYREF
  _QWORD v164[2]; // [rsp+B0h] [rbp-360h] BYREF
  char *v165[2]; // [rsp+C0h] [rbp-350h] BYREF
  __int64 v166; // [rsp+D0h] [rbp-340h] BYREF
  _BYTE *v167; // [rsp+E0h] [rbp-330h] BYREF
  __int64 v168; // [rsp+E8h] [rbp-328h]
  _BYTE v169[32]; // [rsp+F0h] [rbp-320h] BYREF
  char v170[32]; // [rsp+110h] [rbp-300h] BYREF
  __int16 v171; // [rsp+130h] [rbp-2E0h]
  _BYTE *v172; // [rsp+140h] [rbp-2D0h] BYREF
  __int64 v173; // [rsp+148h] [rbp-2C8h]
  _BYTE v174[48]; // [rsp+150h] [rbp-2C0h] BYREF
  _BYTE *v175; // [rsp+180h] [rbp-290h] BYREF
  __int64 v176; // [rsp+188h] [rbp-288h]
  _BYTE v177[48]; // [rsp+190h] [rbp-280h] BYREF
  _BYTE *v178; // [rsp+1C0h] [rbp-250h] BYREF
  __int64 v179; // [rsp+1C8h] [rbp-248h]
  _BYTE v180[64]; // [rsp+1D0h] [rbp-240h] BYREF
  _QWORD *v181; // [rsp+210h] [rbp-200h] BYREF
  __int64 v182; // [rsp+218h] [rbp-1F8h]
  _BYTE v183[64]; // [rsp+220h] [rbp-1F0h] BYREF
  unsigned int *v184; // [rsp+260h] [rbp-1B0h] BYREF
  __int64 v185; // [rsp+268h] [rbp-1A8h]
  _BYTE v186[32]; // [rsp+270h] [rbp-1A0h] BYREF
  __int64 v187; // [rsp+290h] [rbp-180h]
  _QWORD *v188; // [rsp+298h] [rbp-178h]
  __int16 v189; // [rsp+2A0h] [rbp-170h]
  __int64 v190; // [rsp+2A8h] [rbp-168h]
  void **v191; // [rsp+2B0h] [rbp-160h]
  void **v192; // [rsp+2B8h] [rbp-158h]
  __int64 v193; // [rsp+2C0h] [rbp-150h]
  int v194; // [rsp+2C8h] [rbp-148h]
  __int16 v195; // [rsp+2CCh] [rbp-144h]
  char v196; // [rsp+2CEh] [rbp-142h]
  __int64 v197; // [rsp+2D0h] [rbp-140h]
  __int64 v198; // [rsp+2D8h] [rbp-138h]
  void *v199; // [rsp+2E0h] [rbp-130h] BYREF
  void *v200; // [rsp+2E8h] [rbp-128h] BYREF
  __int64 v201; // [rsp+2F0h] [rbp-120h] BYREF
  unsigned int *v202; // [rsp+2F8h] [rbp-118h]
  int v203; // [rsp+300h] [rbp-110h]
  char v204; // [rsp+308h] [rbp-108h] BYREF
  __int64 *v205; // [rsp+388h] [rbp-88h]
  __int64 v206; // [rsp+398h] [rbp-78h] BYREF
  __int64 *v207; // [rsp+3A8h] [rbp-68h]
  __int64 v208; // [rsp+3B8h] [rbp-58h] BYREF
  char v209; // [rsp+3D0h] [rbp-40h]

  v2 = *(_QWORD **)(a2 + 80);
  v172 = v174;
  v173 = 0x600000000LL;
  if ( (_QWORD *)(a2 + 72) == v2 )
  {
    i = 0;
  }
  else
  {
    if ( !v2 )
      BUG();
    while ( 1 )
    {
      i = (_QWORD *)v2[4];
      if ( i != v2 + 3 )
        break;
      v2 = (_QWORD *)v2[1];
      if ( (_QWORD *)(a2 + 72) == v2 )
        break;
      if ( !v2 )
        BUG();
    }
  }
  v4 = v2;
  v5 = (_QWORD *)(a2 + 72);
  v6 = v4;
LABEL_8:
  if ( v6 != v5 )
  {
    while ( 1 )
    {
      if ( !i )
        BUG();
      if ( *((_BYTE *)i - 24) == 85 )
      {
        v13 = *(i - 7);
        if ( v13 )
        {
          if ( !*(_BYTE *)v13 && *(_QWORD *)(v13 + 24) == i[7] && (*(_BYTE *)(v13 + 33) & 0x20) != 0 )
          {
            v14 = *(_DWORD *)(v13 + 36);
            if ( v14 )
            {
              v15 = *(i - 2);
              v16 = *(unsigned __int8 *)(v15 + 8);
              v134 = (_BYTE)v16 != 7 && (unsigned int)(v16 - 17) > 1;
              if ( !v134 )
                break;
            }
          }
        }
      }
LABEL_11:
      for ( i = (_QWORD *)i[1]; ; i = (_QWORD *)v6[4] )
      {
        v7 = v6 - 3;
        if ( !v6 )
          v7 = 0;
        if ( i != v7 + 6 )
          break;
        v6 = (_QWORD *)v6[1];
        if ( v5 == v6 )
          goto LABEL_8;
        if ( !v6 )
          BUG();
      }
      if ( v5 == v6 )
        goto LABEL_19;
    }
    if ( (unsigned int)(v16 - 17) > 1 )
    {
      if ( v16 != 17 )
      {
        v161 = 0;
        v162 = 0;
        v128 = (__int64 *)v15;
        goto LABEL_34;
      }
      v128 = (__int64 *)*(i - 2);
    }
    else
    {
      v128 = **(__int64 ***)(v15 + 16);
    }
    v17 = *(_DWORD *)(v15 + 32);
    v162 = *(_BYTE *)(v15 + 8) == 18;
    v161 = v17;
LABEL_34:
    v168 = 0x300000000LL;
    v18 = *(_BYTE *)(v15 + 8) == 7;
    v167 = v169;
    if ( !v18 && sub_9B76D0(v14, 0xFFFFFFFF, 0) )
    {
      v76 = (unsigned int)v168;
      v77 = (unsigned int)v168 + 1LL;
      if ( v77 > HIDWORD(v168) )
      {
        sub_C8D5F0((__int64)&v167, v169, v77, 8u, v19, v20);
        v76 = (unsigned int)v168;
      }
      *(_QWORD *)&v167[8 * v76] = v128;
      LODWORD(v168) = v168 + 1;
    }
    v181 = v183;
    v182 = 0x800000000LL;
    v21 = *((unsigned __int8 *)i - 24);
    v160 = (__int64)(i - 3);
    if ( v21 == 40 )
    {
      v22 = -32 - 32LL * (unsigned int)sub_B491D0(v160);
    }
    else
    {
      v22 = -32;
      if ( v21 != 85 )
      {
        if ( v21 != 34 )
          goto LABEL_204;
        v22 = -96;
      }
    }
    if ( *((char *)i - 17) < 0 )
    {
      v23 = sub_BD2BC0(v160);
      v146 = v24 + v23;
      if ( *((char *)i - 17) >= 0 )
      {
        if ( (unsigned int)(v146 >> 4) )
LABEL_200:
          BUG();
      }
      else if ( (unsigned int)((v146 - sub_BD2BC0(v160)) >> 4) )
      {
        if ( *((char *)i - 17) >= 0 )
          goto LABEL_200;
        v147 = *(_DWORD *)(sub_BD2BC0(v160) + 8);
        if ( *((char *)i - 17) >= 0 )
          BUG();
        v25 = sub_BD2BC0(v160);
        v22 -= 32LL * (unsigned int)(*(_DWORD *)(v25 + v26 - 4) - v147);
      }
    }
    v27 = v22;
    v28 = 0;
    v29 = 32LL * (*((_DWORD *)i - 5) & 0x7FFFFFF);
    v148 = v160 + v27;
    if ( v160 - v29 != v160 + v27 )
    {
      v138 = v5;
      v136 = v6;
      v135 = i;
      v30 = v160 - v29;
      do
      {
        v31 = *(_QWORD *)(*(_QWORD *)v30 + 8LL);
        v32 = sub_9B76D0(v14, v28, 0);
        v35 = sub_9B75A0(v14, v28, 0);
        if ( v35 )
        {
          v70 = (unsigned int)v182;
          v71 = (unsigned int)v182 + 1LL;
          if ( v71 > HIDWORD(v182) )
          {
            sub_C8D5F0((__int64)&v181, v183, v71, 8u, (__int64)v33, v34);
            v70 = (unsigned int)v182;
          }
          v181[v70] = v31;
          LODWORD(v182) = v182 + 1;
          if ( v32 )
          {
            v72 = (unsigned int)v168;
            v73 = (unsigned int)v168 + 1LL;
            if ( v73 > HIDWORD(v168) )
            {
              sub_C8D5F0((__int64)&v167, v169, v73, 8u, (__int64)v33, v34);
              v72 = (unsigned int)v168;
            }
            *(_QWORD *)&v167[8 * v72] = v31;
            LODWORD(v168) = v168 + 1;
          }
        }
        else
        {
          if ( (unsigned int)*(unsigned __int8 *)(v31 + 8) - 17 > 1 )
            goto LABEL_90;
          v36 = (unsigned int)v182;
          v37 = *(_QWORD *)(v31 + 24);
          v38 = (unsigned int)v182 + 1LL;
          if ( v38 > HIDWORD(v182) )
          {
            v121 = *(_QWORD *)(v31 + 24);
            sub_C8D5F0((__int64)&v181, v183, v38, 8u, (__int64)v33, v34);
            v36 = (unsigned int)v182;
            v35 = 0;
            v37 = v121;
          }
          v181[v36] = v37;
          LODWORD(v182) = v182 + 1;
          if ( v32 )
          {
            v74 = (unsigned int)v168;
            v75 = (unsigned int)v168 + 1LL;
            if ( v75 > HIDWORD(v168) )
            {
              v120 = v35;
              v122 = v37;
              sub_C8D5F0((__int64)&v167, v169, v75, 8u, (__int64)v33, v34);
              v74 = (unsigned int)v168;
              v35 = v120;
              v37 = v122;
            }
            *(_QWORD *)&v167[8 * v74] = v37;
            LODWORD(v168) = v168 + 1;
          }
          v39 = *(_BYTE *)(v31 + 8) == 18;
          if ( v161 )
          {
            if ( v161 != *(_DWORD *)(v31 + 32) || v162 != v39 )
            {
LABEL_90:
              v5 = v138;
              v6 = v136;
              i = v135;
              goto LABEL_91;
            }
          }
          else
          {
            v161 = *(_DWORD *)(v31 + 32);
            v162 = v39;
          }
        }
        ++v28;
        v30 += 32;
      }
      while ( v30 != v148 );
      v5 = v138;
      v6 = v136;
      i = v135;
    }
    if ( (unsigned __int8)sub_B60C20(v14) )
    {
      v78 = (__int64 *)sub_B43CA0(v160);
      sub_B6E0E0(&v163, v14, (__int64)v167, (unsigned int)v168, v78, 0);
      v44 = v163;
      goto LABEL_65;
    }
    v41 = sub_B60C10(v14);
    v42 = v40;
    if ( !v41 )
    {
      LOBYTE(v164[0]) = 0;
      v163.m128i_i64[0] = (__int64)v164;
      v163.m128i_i64[1] = 0;
      v44 = (__m128i)(unsigned __int64)v164;
      goto LABEL_65;
    }
    v43 = v164;
    v201 = v40;
    v163.m128i_i64[0] = (__int64)v164;
    if ( v40 > 0xF )
    {
      v163.m128i_i64[0] = sub_22409D0((__int64)&v163, (unsigned __int64 *)&v201, 0);
      v116 = (_QWORD *)v163.m128i_i64[0];
      v164[0] = v201;
    }
    else
    {
      if ( v40 == 1 )
      {
        LOBYTE(v164[0]) = *v41;
LABEL_64:
        v163.m128i_i64[1] = v40;
        *((_BYTE *)v43 + v40) = 0;
        v44 = v163;
LABEL_65:
        v45 = sub_97F930(*a1, v44.m128i_i64[0], v44.m128i_u64[1], (__int64)&v161, 0);
        if ( !v45 )
        {
          v45 = sub_97F930(*a1, v163.m128i_i64[0], v163.m128i_u64[1], (__int64)&v161, 1);
          if ( !v45 )
          {
            v35 = 0;
            goto LABEL_174;
          }
        }
        v46 = (_QWORD **)sub_BCF480(v128, v181, (unsigned int)v182, 0);
        sub_97F000((__int64 *)v165, v45);
        sub_C0A940((__int64)&v201, v165[0], v165[1], (__int64)v46);
        v35 = v209;
        if ( !v209 )
          goto LABEL_172;
        v47 = v202;
        v33 = &v201;
        v48 = &v202[4 * v203];
        if ( v202 != v48 )
        {
          do
          {
            v49 = v47[1];
            if ( v49 != 10 )
            {
              v34 = *((_DWORD *)i - 5) & 0x7FFFFFF;
              if ( (v49 == 0) != (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(v160 + 32 * (*v47 - v34))
                                                                              + 8LL)
                                                                  + 8LL)
                               - 17 <= 1 )
                goto LABEL_165;
            }
            v47 += 4;
          }
          while ( v48 != v47 );
        }
        v149 = v209;
        v50 = sub_C0A1C0((int *)&v201, v46);
        if ( !v50 )
        {
          v35 = 0;
          goto LABEL_163;
        }
        v51 = *(i - 7);
        if ( v51 )
        {
          if ( *(_BYTE *)v51 )
          {
            v51 = 0;
          }
          else if ( *(_QWORD *)(v51 + 24) != i[7] )
          {
            v51 = 0;
          }
        }
        v52 = *(_QWORD *)(v45 + 24);
        v139 = v149;
        v53 = *(_QWORD *)(v45 + 16);
        v137 = v51;
        v150 = v52;
        v54 = (__int64 **)sub_B43CA0(v160);
        v151 = sub_2F81590(v54, v50, v53, v150, v137);
        v55 = sub_BD5C60(v160);
        v187 = 0;
        v190 = v55;
        v191 = &v199;
        v192 = &v200;
        v188 = 0;
        v189 = 0;
        v199 = &unk_49DA100;
        v193 = 0;
        v195 = 512;
        v194 = 0;
        v196 = 7;
        v197 = 0;
        v198 = 0;
        v200 = &unk_49DA0B0;
        v56 = i[2];
        v184 = (unsigned int *)v186;
        v185 = 0x200000000LL;
        v187 = v56;
        v188 = i;
        v57 = (__int64 *)sub_B46C60(v160);
        v58 = v139;
        v59 = *v57;
        v178 = (_BYTE *)*v57;
        if ( v178 && (sub_B96E90((__int64)&v178, v59, 1), v62 = (__int64)v178, v58 = v139, v178) )
        {
          v63 = v184;
          v64 = v185;
          v65 = &v184[4 * (unsigned int)v185];
          if ( v184 != v65 )
          {
            while ( 1 )
            {
              v60 = *v63;
              if ( !(_DWORD)v60 )
                break;
              v63 += 4;
              if ( v65 == v63 )
                goto LABEL_182;
            }
            *((_QWORD *)v63 + 1) = v178;
LABEL_83:
            v140 = v58;
            sub_B91220((__int64)&v178, v62);
            v58 = v140;
LABEL_84:
            v66 = *((unsigned __int8 *)i - 24);
            if ( v66 == 40 )
            {
              v145 = v58;
              v115 = sub_B491D0(v160);
              v58 = v145;
              v67 = -32 - 32LL * v115;
            }
            else
            {
              v67 = -32;
              if ( v66 != 85 )
              {
                if ( v66 != 34 )
LABEL_204:
                  BUG();
                v67 = -96;
              }
            }
            if ( *((char *)i - 17) < 0 )
            {
              v141 = v58;
              v79 = sub_BD2BC0(v160);
              v58 = v141;
              v81 = v79 + v80;
              v82 = 0;
              if ( *((char *)i - 17) < 0 )
              {
                v83 = sub_BD2BC0(v160);
                v58 = v141;
                v82 = v83;
              }
              if ( (unsigned int)((v81 - v82) >> 4) )
              {
                v142 = v58;
                if ( *((char *)i - 17) >= 0 )
                  BUG();
                v84 = *(_DWORD *)(sub_BD2BC0(v160) + 8);
                if ( *((char *)i - 17) >= 0 )
                  BUG();
                v85 = sub_BD2BC0(v160);
                v58 = v142;
                v67 -= 32LL * (unsigned int)(*(_DWORD *)(v85 + v86 - 4) - v84);
              }
            }
            v87 = *((_DWORD *)i - 5);
            v176 = 0x600000000LL;
            v88 = v177;
            v89 = 0;
            v90 = v160 + v67;
            v175 = v177;
            v91 = 32LL * (v87 & 0x7FFFFFF);
            v92 = (_QWORD *)(v160 - v91);
            v93 = v67 + v91;
            v94 = v93 >> 5;
            if ( (unsigned __int64)v93 > 0xC0 )
            {
              v123 = v92;
              v126 = v90;
              v132 = v58;
              sub_C8D5F0((__int64)&v175, v177, v93 >> 5, 8u, v90, (__int64)&v175);
              v89 = v176;
              v92 = v123;
              v90 = v126;
              v58 = v132;
              v88 = &v175[8 * (unsigned int)v176];
            }
            if ( v92 != (_QWORD *)v90 )
            {
              do
              {
                if ( v88 )
                  *v88 = *v92;
                v92 += 4;
                ++v88;
              }
              while ( (_QWORD *)v90 != v92 );
              v89 = v176;
            }
            LODWORD(v176) = v94 + v89;
            if ( v203 )
            {
              v95 = 0;
              while ( v202[4 * v95 + 1] != 10 )
              {
                if ( v203 == ++v95 )
                  goto LABEL_145;
              }
              v125 = v58;
              v96 = (_QWORD *)sub_BD5C60(v160);
              v97 = (__int64 *)sub_BCB2A0(v96);
              v98 = sub_BCE1B0(v97, v201);
              v99 = sub_AD62B0(v98);
              v100 = (unsigned __int64)v175;
              v101 = 8 * v95;
              v143 = v99;
              v102 = 8LL * (unsigned int)v176;
              v103 = &v175[8 * v95];
              v58 = v125;
              v104 = v176;
              v105 = (__int64 *)&v175[v102];
              if ( v103 == &v175[v102] )
              {
                if ( (unsigned __int64)(unsigned int)v176 + 1 > HIDWORD(v176) )
                {
                  sub_C8D5F0((__int64)&v175, v177, (unsigned int)v176 + 1LL, 8u, (unsigned int)v176, (__int64)&v175);
                  v58 = v125;
                  v105 = (__int64 *)&v175[8 * (unsigned int)v176];
                }
                *v105 = v143;
                LODWORD(v176) = v176 + 1;
              }
              else
              {
                v106 = (unsigned int)v176 + 1LL;
                if ( v106 > HIDWORD(v176) )
                {
                  v127 = v101;
                  v133 = v58;
                  sub_C8D5F0((__int64)&v175, v177, v106, 8u, v106, (__int64)&v175);
                  v100 = (unsigned __int64)v175;
                  v58 = v133;
                  v104 = v176;
                  v102 = 8LL * (unsigned int)v176;
                  v103 = &v175[v127];
                  v105 = (__int64 *)&v175[v102];
                }
                v107 = (char *)(v100 + v102 - 8);
                if ( v105 )
                {
                  *v105 = *(_QWORD *)v107;
                  v100 = (unsigned __int64)v175;
                  v104 = v176;
                  v102 = 8LL * (unsigned int)v176;
                  v107 = &v175[v102 - 8];
                }
                if ( v107 != v103 )
                {
                  v129 = v58;
                  memmove((void *)(v100 + v102 - (v107 - v103)), v103, v107 - v103);
                  v104 = v176;
                  v58 = v129;
                }
                LODWORD(v176) = v104 + 1;
                *(_QWORD *)v103 = v143;
              }
            }
LABEL_145:
            v130 = v58;
            v178 = v180;
            v179 = 0x100000000LL;
            sub_B56970(v160, (__int64)&v178);
            v108 = 0;
            v171 = 257;
            if ( v151 )
              v108 = *((_QWORD *)v151 + 3);
            v109 = sub_B33530(
                     &v184,
                     v108,
                     (int)v151,
                     (int)v175,
                     v176,
                     (__int64)v170,
                     (__int64)v178,
                     (unsigned int)v179,
                     0);
            sub_BD84D0(v160, v109);
            v110 = sub_920620(v109);
            v111 = v130;
            if ( v110 )
            {
              sub_B45230(v109, v160);
              v111 = v130;
            }
            v154 = (__int64)v178;
            v112 = &v178[56 * (unsigned int)v179];
            if ( v178 != v112 )
            {
              v131 = v111;
              v113 = (unsigned __int64 *)&v178[56 * (unsigned int)v179];
              do
              {
                v114 = *(v113 - 3);
                v113 -= 7;
                if ( v114 )
                  j_j___libc_free_0(v114);
                if ( (unsigned __int64 *)*v113 != v113 + 2 )
                  j_j___libc_free_0(*v113);
              }
              while ( (unsigned __int64 *)v154 != v113 );
              v111 = v131;
              v112 = v178;
            }
            if ( v112 != v180 )
            {
              v155 = v111;
              _libc_free((unsigned __int64)v112);
              v111 = v155;
            }
            if ( v175 != v177 )
            {
              v156 = v111;
              _libc_free((unsigned __int64)v175);
              v111 = v156;
            }
            v157 = v111;
            nullsub_61();
            v199 = &unk_49DA100;
            nullsub_63();
            v35 = v157;
            if ( v184 != (unsigned int *)v186 )
            {
              _libc_free((unsigned __int64)v184);
              v35 = v157;
            }
LABEL_163:
            if ( v209 )
            {
              v134 = v35;
LABEL_165:
              v209 = 0;
              if ( v207 != &v208 )
                j_j___libc_free_0((unsigned __int64)v207);
              if ( v205 != &v206 )
                j_j___libc_free_0((unsigned __int64)v205);
              if ( v202 != (unsigned int *)&v204 )
                _libc_free((unsigned __int64)v202);
              v35 = v134;
            }
LABEL_172:
            if ( (__int64 *)v165[0] != &v166 )
            {
              v158 = v35;
              j_j___libc_free_0((unsigned __int64)v165[0]);
              v35 = v158;
            }
LABEL_174:
            if ( (_QWORD *)v163.m128i_i64[0] != v164 )
            {
              v159 = v35;
              j_j___libc_free_0(v163.m128i_u64[0]);
              v35 = v159;
            }
LABEL_91:
            if ( v181 != (_QWORD *)v183 )
            {
              v152 = v35;
              _libc_free((unsigned __int64)v181);
              v35 = v152;
            }
            if ( v167 != v169 )
            {
              v153 = v35;
              _libc_free((unsigned __int64)v167);
              v35 = v153;
            }
            if ( v35 )
            {
              v68 = (unsigned int)v173;
              v69 = (unsigned int)v173 + 1LL;
              if ( v69 > HIDWORD(v173) )
              {
                sub_C8D5F0((__int64)&v172, v174, v69, 8u, (__int64)v33, v34);
                v68 = (unsigned int)v173;
              }
              *(_QWORD *)&v172[8 * v68] = v160;
              LODWORD(v173) = v173 + 1;
            }
            goto LABEL_11;
          }
LABEL_182:
          if ( (unsigned int)v185 >= (unsigned __int64)HIDWORD(v185) )
          {
            v117 = (unsigned int)v185 + 1LL;
            v118 = v119 & 0xFFFFFFFF00000000LL;
            v119 &= 0xFFFFFFFF00000000LL;
            if ( HIDWORD(v185) < v117 )
            {
              sub_C8D5F0((__int64)&v184, v186, v117, 0x10u, v60, v61);
              v58 = v139;
              v65 = &v184[4 * (unsigned int)v185];
            }
            *(_QWORD *)v65 = v118;
            *((_QWORD *)v65 + 1) = v62;
            v62 = (__int64)v178;
            LODWORD(v185) = v185 + 1;
          }
          else
          {
            if ( v65 )
            {
              *v65 = 0;
              *((_QWORD *)v65 + 1) = v62;
              v64 = v185;
              v62 = (__int64)v178;
            }
            LODWORD(v185) = v64 + 1;
          }
        }
        else
        {
          v144 = v58;
          sub_93FB40((__int64)&v184, 0);
          v62 = (__int64)v178;
          v58 = v144;
        }
        if ( !v62 )
          goto LABEL_84;
        goto LABEL_83;
      }
      if ( !v40 )
        goto LABEL_64;
      v116 = v164;
    }
    memcpy(v116, v41, v42);
    v40 = v201;
    v43 = (_QWORD *)v163.m128i_i64[0];
    goto LABEL_64;
  }
LABEL_19:
  v8 = v172;
  v9 = &v172[8 * (unsigned int)v173];
  v10 = v173;
  if ( v9 != v172 )
  {
    do
    {
      v11 = *(_QWORD **)v8;
      v8 += 8;
      sub_B43D60(v11);
    }
    while ( v9 != v8 );
    v8 = v172;
    v10 = v173;
  }
  LOBYTE(v9) = v10 != 0;
  if ( v8 != v174 )
    _libc_free((unsigned __int64)v8);
  return (unsigned int)v9;
}
