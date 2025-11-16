// Function: sub_2280510
// Address: 0x2280510
//
__int64 __fastcall sub_2280510(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  int v9; // r8d
  unsigned int i; // eax
  __int64 v11; // rsi
  unsigned int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int64 *v15; // rax
  unsigned __int64 *v16; // rax
  __int64 *v17; // rax
  _QWORD *v18; // rax
  __int64 v19; // rax
  int v20; // ebx
  __int64 *v21; // rsi
  __int64 v22; // r13
  __int64 v23; // r11
  __int64 v24; // r10
  __int64 v25; // r9
  int v26; // r8d
  unsigned int v27; // edx
  __int64 *v28; // rax
  __int64 v29; // rdi
  int v30; // eax
  __int64 v31; // rdx
  __int64 v32; // rcx
  _QWORD **v33; // rbx
  _QWORD **v34; // r12
  _QWORD *v35; // rdi
  _QWORD *v36; // rbx
  _QWORD *v37; // r12
  __int64 v38; // rax
  unsigned int v40; // r12d
  unsigned int v41; // edx
  __int64 *v42; // rax
  __int64 v43; // rcx
  int v44; // eax
  __int64 v45; // rdi
  __int64 v46; // rdx
  int v47; // r8d
  unsigned int v48; // ecx
  __int64 *v49; // rax
  __int64 v50; // r14
  int v51; // eax
  char v52; // al
  __int64 *v53; // r8
  int v54; // r10d
  __int64 v55; // rcx
  __int64 *v56; // rdx
  __int64 v57; // rdi
  __int64 v58; // rcx
  __int64 v59; // rsi
  __int64 v60; // rax
  unsigned __int64 v61; // rdx
  unsigned __int64 v62; // rsi
  __int64 v63; // rcx
  __int64 *v64; // r8
  int v65; // edi
  unsigned int v66; // edx
  __int64 *v67; // rax
  __int64 v68; // r10
  __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rbx
  __int64 j; // r12
  __int64 v73; // rax
  int v74; // ecx
  unsigned __int64 v75; // rsi
  __int64 *v76; // r12
  __int64 *v77; // r8
  int v78; // edi
  unsigned int v79; // edx
  __int64 *v80; // rax
  __int64 *v81; // r10
  __int64 v82; // rax
  __int64 v83; // rdx
  __int64 **v84; // rax
  __int64 **v85; // rdx
  unsigned int v86; // eax
  __int64 *v87; // rax
  __int64 *v88; // rdx
  __int64 v89; // rcx
  __int64 v90; // rdx
  __int64 v91; // rcx
  __int64 v92; // r8
  __int64 v93; // r9
  __int64 v94; // rdx
  __int64 v95; // rcx
  __int64 v96; // r8
  __int64 v97; // r9
  __int64 v98; // rcx
  __int64 **v99; // rax
  __int64 **v100; // rdx
  _QWORD *v101; // r13
  _QWORD *v102; // r12
  __int64 v103; // rbx
  _QWORD *v104; // rdi
  __int64 v105; // rdx
  __int64 v106; // rsi
  void (__fastcall *v107)(_QWORD *, __int64, __int64, __int64 *); // r8
  _QWORD *v108; // rbx
  __int64 v109; // r13
  __int64 *v110; // rax
  __int64 *v111; // rdi
  _QWORD *v112; // rdi
  __int64 v113; // rdx
  __int64 v114; // rsi
  int v115; // eax
  int v116; // eax
  int v117; // eax
  int v118; // esi
  unsigned int v119; // esi
  unsigned int v120; // eax
  unsigned int v121; // edx
  unsigned int v122; // ecx
  __int64 v123; // rdx
  __int64 *v124; // rax
  unsigned int v125; // eax
  unsigned int v126; // ebx
  char v127; // al
  __int64 v128; // rdi
  __int64 v129; // rax
  __int64 *v130; // rax
  __int64 *v131; // rdx
  int v132; // r11d
  __int64 *v133; // rsi
  int v134; // eax
  int v135; // esi
  __int64 *v136; // rax
  __int64 *v137; // rdx
  int v138; // eax
  int v139; // ecx
  __int64 v140; // rax
  __int64 v141; // [rsp+28h] [rbp-678h]
  __int64 v142; // [rsp+48h] [rbp-658h]
  __int64 v143; // [rsp+60h] [rbp-640h]
  __int64 v145; // [rsp+88h] [rbp-618h]
  __int64 *v146; // [rsp+90h] [rbp-610h]
  __int64 v148; // [rsp+A0h] [rbp-600h]
  _QWORD *v149; // [rsp+A8h] [rbp-5F8h]
  __int64 v150; // [rsp+B0h] [rbp-5F0h] BYREF
  __int64 *v151; // [rsp+B8h] [rbp-5E8h] BYREF
  _BYTE *v152; // [rsp+C0h] [rbp-5E0h] BYREF
  __int64 v153; // [rsp+C8h] [rbp-5D8h]
  _BYTE v154[32]; // [rsp+D0h] [rbp-5D0h] BYREF
  __int64 v155; // [rsp+F0h] [rbp-5B0h] BYREF
  __int64 **v156; // [rsp+F8h] [rbp-5A8h]
  __int64 v157; // [rsp+100h] [rbp-5A0h]
  int v158; // [rsp+108h] [rbp-598h]
  char v159; // [rsp+10Ch] [rbp-594h]
  char v160; // [rsp+110h] [rbp-590h] BYREF
  __int64 v161; // [rsp+130h] [rbp-570h] BYREF
  __int64 v162; // [rsp+138h] [rbp-568h]
  __int64 *v163; // [rsp+140h] [rbp-560h] BYREF
  unsigned int v164; // [rsp+148h] [rbp-558h]
  __int64 v165; // [rsp+180h] [rbp-520h] BYREF
  unsigned __int64 v166; // [rsp+188h] [rbp-518h]
  char v167; // [rsp+19Ch] [rbp-504h]
  unsigned __int64 v168; // [rsp+1B8h] [rbp-4E8h]
  char v169; // [rsp+1CCh] [rbp-4D4h]
  __int64 v170; // [rsp+1E0h] [rbp-4C0h] BYREF
  __int64 v171; // [rsp+1E8h] [rbp-4B8h]
  __int64 *v172; // [rsp+1F0h] [rbp-4B0h] BYREF
  unsigned int v173; // [rsp+1F8h] [rbp-4A8h]
  _BYTE *v174; // [rsp+230h] [rbp-470h] BYREF
  __int64 v175; // [rsp+238h] [rbp-468h]
  _BYTE v176[16]; // [rsp+240h] [rbp-460h] BYREF
  __int64 v177; // [rsp+250h] [rbp-450h] BYREF
  __int64 v178; // [rsp+258h] [rbp-448h]
  __int64 *v179; // [rsp+260h] [rbp-440h] BYREF
  unsigned int v180; // [rsp+268h] [rbp-438h]
  _BYTE *v181; // [rsp+2A0h] [rbp-400h] BYREF
  __int64 v182; // [rsp+2A8h] [rbp-3F8h]
  _BYTE v183[16]; // [rsp+2B0h] [rbp-3F0h] BYREF
  _QWORD v184[124]; // [rsp+2C0h] [rbp-3E0h] BYREF

  v148 = *(_QWORD *)(sub_BC0510(a4, &unk_4FDADC0, a3) + 8);
  v6 = sub_BC0510(a4, &unk_4F86C48, a3);
  v7 = *(unsigned int *)(a4 + 88);
  v8 = *(_QWORD *)(a4 + 72);
  v141 = v6;
  v145 = v6 + 8;
  if ( !(_DWORD)v7 )
    goto LABEL_229;
  v9 = 1;
  for ( i = (v7 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F82418 >> 9) ^ ((unsigned int)&unk_4F82418 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v7 - 1) & v12 )
  {
    v11 = v8 + 24LL * i;
    if ( *(_UNKNOWN **)v11 == &unk_4F82418 && a3 == *(_QWORD *)(v11 + 8) )
      break;
    if ( *(_QWORD *)v11 == -4096 && *(_QWORD *)(v11 + 8) == -4096 )
      goto LABEL_229;
    v12 = v9 + i;
    ++v9;
  }
  if ( v11 == v8 + 24 * v7 || (v13 = *(_QWORD *)(*(_QWORD *)(v11 + 16) + 24LL)) == 0 )
LABEL_229:
    BUG();
  v14 = *(_QWORD *)(v13 + 8);
  v170 = 0;
  v171 = 1;
  v143 = v14;
  v15 = (unsigned __int64 *)&v172;
  do
  {
    *v15 = -4096;
    v15 += 2;
  }
  while ( v15 != (unsigned __int64 *)&v174 );
  v177 = 0;
  v174 = v176;
  v175 = 0x100000000LL;
  v16 = (unsigned __int64 *)&v179;
  v178 = 1;
  do
  {
    *v16 = -4096;
    v16 += 2;
  }
  while ( v16 != (unsigned __int64 *)&v181 );
  v155 = 0;
  v181 = v183;
  v182 = 0x100000000LL;
  v156 = (__int64 **)&v160;
  v17 = (__int64 *)&v163;
  v157 = 4;
  v158 = 0;
  v159 = 1;
  v161 = 0;
  v162 = 1;
  do
  {
    *v17 = -4096;
    v17 += 2;
    *(v17 - 1) = -4096;
  }
  while ( v17 != &v165 );
  v152 = v154;
  v153 = 0x400000000LL;
  memset(v184, 0, 0x3A8u);
  LODWORD(v184[5]) = 2;
  v184[0] = &v177;
  v184[1] = &v155;
  v184[4] = &v184[7];
  v184[10] = &v184[13];
  BYTE4(v184[6]) = 1;
  LODWORD(v184[11]) = 2;
  BYTE4(v184[12]) = 1;
  sub_AE6EC0((__int64)&v184[3], (__int64)&qword_4F82400);
  v184[15] = &v161;
  memset(&v184[17], 0, 0x320u);
  v184[16] = &v152;
  v18 = &v184[19];
  LOBYTE(v184[18]) = 1;
  do
  {
    *v18 = -4096;
    v18 += 2;
  }
  while ( v18 != &v184[51] );
  v184[51] = &v184[53];
  v184[52] = 0x1000000000LL;
  v19 = *(_QWORD *)(sub_BC0510(a4, &qword_4F8A320, a3) + 8);
  *(_QWORD *)a1 = 0;
  v150 = v19;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 16) = 2;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  sub_AE6EC0(a1, (__int64)&qword_4F82400);
  sub_D2AD40(v145, &qword_4F82400);
  v20 = *(_DWORD *)(v141 + 448);
  if ( !v20 )
    goto LABEL_26;
  v21 = *(__int64 **)(v141 + 440);
  v22 = *v21;
  if ( !*v21 )
    goto LABEL_26;
  v23 = *(_QWORD *)(v141 + 592);
  v24 = *(unsigned int *)(v141 + 608);
  v25 = v23 + 16 * v24;
  v26 = v24 - 1;
  while ( !*(_DWORD *)(v22 + 16) )
  {
    if ( (_DWORD)v24 )
    {
      v27 = v26 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v28 = (__int64 *)(v23 + 16LL * v27);
      v29 = *v28;
      if ( *v28 == v22 )
      {
LABEL_21:
        v30 = *((_DWORD *)v28 + 2) + 1;
        if ( v20 == v30 )
          goto LABEL_26;
        goto LABEL_22;
      }
      v138 = 1;
      while ( v29 != -4096 )
      {
        v139 = v138 + 1;
        v140 = v26 & (v27 + v138);
        v27 = v140;
        v28 = (__int64 *)(v23 + 16 * v140);
        v29 = *v28;
        if ( *v28 == v22 )
          goto LABEL_21;
        v138 = v139;
      }
    }
    v30 = *(_DWORD *)(v25 + 8) + 1;
    if ( v20 == v30 )
      goto LABEL_26;
LABEL_22:
    v22 = v21[v30];
    if ( !v22 )
      goto LABEL_26;
  }
  v40 = v175;
  if ( (_DWORD)v24 )
  {
LABEL_56:
    v41 = (v24 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
    v42 = (__int64 *)(v23 + 16LL * v41);
    v43 = *v42;
    if ( *v42 == v22 )
      goto LABEL_57;
    v134 = 1;
    while ( v43 != -4096 )
    {
      v135 = v134 + 1;
      v41 = (v24 - 1) & (v134 + v41);
      v42 = (__int64 *)(v23 + 16LL * v41);
      v43 = *v42;
      if ( *v42 == v22 )
        goto LABEL_57;
      v134 = v135;
    }
  }
  while ( 2 )
  {
    v42 = (__int64 *)(v23 + 16LL * (unsigned int)v24);
LABEL_57:
    v44 = *((_DWORD *)v42 + 2) + 1;
    if ( v44 == v20 )
    {
LABEL_66:
      v142 = 0;
      goto LABEL_67;
    }
    v45 = *(_QWORD *)(v141 + 440);
    v46 = *(_QWORD *)(v45 + 8LL * v44);
    v142 = v46;
    if ( !v46 )
      goto LABEL_67;
    v47 = v24 - 1;
    v25 = v23 + 16LL * (unsigned int)v24;
    while ( 2 )
    {
      if ( !*(_DWORD *)(v46 + 16) )
      {
        if ( (_DWORD)v24 )
        {
          v48 = v47 & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
          v49 = (__int64 *)(v23 + 16LL * v48);
          v50 = *v49;
          if ( *v49 != v46 )
          {
            v117 = 1;
            while ( v50 != -4096 )
            {
              v118 = v117 + 1;
              v48 = v47 & (v117 + v48);
              v49 = (__int64 *)(v23 + 16LL * v48);
              v50 = *v49;
              if ( *v49 == v46 )
                goto LABEL_61;
              v117 = v118;
            }
            goto LABEL_65;
          }
LABEL_61:
          v51 = *((_DWORD *)v49 + 2) + 1;
          if ( v20 == v51 )
            goto LABEL_66;
        }
        else
        {
LABEL_65:
          v51 = *(_DWORD *)(v25 + 8) + 1;
          if ( v20 == v51 )
            goto LABEL_66;
        }
        v46 = *(_QWORD *)(v45 + 8LL * v51);
        if ( !v46 )
          break;
        continue;
      }
      break;
    }
    v142 = v46;
LABEL_67:
    v165 = v22;
    v166 = v40;
    v52 = v171;
    if ( (v171 & 1) != 0 )
    {
      v53 = (__int64 *)&v172;
      v54 = 3;
      goto LABEL_69;
    }
    v119 = v173;
    v53 = v172;
    v54 = v173 - 1;
    if ( !v173 )
    {
      v120 = v171;
      ++v170;
      v151 = 0;
      v121 = ((unsigned int)v171 >> 1) + 1;
      goto LABEL_174;
    }
LABEL_69:
    LODWORD(v55) = v54 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
    v56 = &v53[2 * (unsigned int)v55];
    v57 = *v56;
    if ( *v56 == v22 )
    {
LABEL_70:
      v58 = v40;
      v59 = v56[1];
      if ( v59 != v40 - 1LL )
      {
        *(_QWORD *)&v174[8 * v59] = 0;
        v56[1] = (unsigned int)v175;
        goto LABEL_72;
      }
      goto LABEL_75;
    }
    v132 = 1;
    v133 = 0;
    while ( v57 != -4096 )
    {
      if ( v57 == -8192 && !v133 )
        v133 = v56;
      v25 = (unsigned int)(v132 + 1);
      v55 = v54 & (unsigned int)(v55 + v132);
      v56 = &v53[2 * v55];
      v57 = *v56;
      if ( *v56 == v22 )
        goto LABEL_70;
      ++v132;
    }
    v120 = v171;
    if ( v133 )
      v56 = v133;
    ++v170;
    v151 = v56;
    v121 = ((unsigned int)v171 >> 1) + 1;
    if ( (v171 & 1) == 0 )
    {
      v119 = v173;
LABEL_174:
      if ( 3 * v119 > 4 * v121 )
        goto LABEL_175;
LABEL_200:
      v119 *= 2;
      goto LABEL_201;
    }
    v119 = 4;
    if ( 4 * v121 >= 0xC )
      goto LABEL_200;
LABEL_175:
    v122 = v119 - HIDWORD(v171) - v121;
    v123 = v22;
    if ( v122 <= v119 >> 3 )
    {
LABEL_201:
      sub_227FAF0((__int64)&v170, v119);
      sub_227C380((__int64)&v170, &v165, &v151);
      v123 = v165;
      v120 = v171;
    }
    LODWORD(v171) = (2 * (v120 >> 1) + 2) | v120 & 1;
    v124 = v151;
    if ( *v151 != -4096 )
      --HIDWORD(v171);
    *v151 = v123;
    v124[1] = v166;
LABEL_72:
    v60 = (unsigned int)v175;
    v61 = (unsigned int)v175 + 1LL;
    if ( v61 > HIDWORD(v175) )
    {
      sub_C8D5F0((__int64)&v174, v176, v61, 8u, (__int64)v53, v25);
      v60 = (unsigned int)v175;
    }
    *(_QWORD *)&v174[8 * v60] = v22;
    v52 = v171;
    LODWORD(v175) = v175 + 1;
    v58 = (unsigned int)v175;
    v40 = v175;
LABEL_75:
    v62 = (unsigned __int64)v174;
    v63 = *(_QWORD *)&v174[8 * v58 - 8];
    if ( (v52 & 1) != 0 )
    {
LABEL_76:
      v64 = (__int64 *)&v172;
      v65 = 3;
LABEL_77:
      v66 = v65 & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
      v67 = &v64[2 * v66];
      v68 = *v67;
      if ( *v67 == v63 )
      {
LABEL_78:
        *v67 = -8192;
        ++HIDWORD(v171);
        v62 = (unsigned __int64)v174;
        v40 = v175;
        LODWORD(v171) = (2 * ((unsigned int)v171 >> 1) - 2) | v171 & 1;
      }
      else
      {
        v116 = 1;
        while ( v68 != -4096 )
        {
          v25 = (unsigned int)(v116 + 1);
          v66 = v65 & (v116 + v66);
          v67 = &v64[2 * v66];
          v68 = *v67;
          if ( v63 == *v67 )
            goto LABEL_78;
          v116 = v25;
        }
      }
    }
    else
    {
LABEL_110:
      v64 = v172;
      if ( v173 )
      {
        v65 = v173 - 1;
        goto LABEL_77;
      }
    }
    v69 = v40 - 1;
    v70 = v62 + 8 * v69 - 8;
    while ( 1 )
    {
      LODWORD(v175) = v69;
      if ( !(_DWORD)v69 )
        break;
      v70 -= 8;
      if ( *(_QWORD *)(v70 + 8) )
        break;
      LODWORD(v69) = v69 - 1;
    }
    v71 = *(_QWORD *)(v63 + 8);
    for ( j = v71 + 8LL * *(unsigned int *)(v63 + 16); v71 != j; j -= 8 )
    {
      v73 = *(_QWORD *)(j - 8);
      v165 = v73;
      sub_22801B0((__int64)&v177, &v165);
    }
    v146 = 0;
    v74 = v182;
    while ( 1 )
    {
LABEL_86:
      v75 = (unsigned __int64)v181;
      v76 = *(__int64 **)&v181[8 * v74 - 8];
      if ( (v178 & 1) != 0 )
      {
        v77 = (__int64 *)&v179;
        v78 = 3;
      }
      else
      {
        v77 = v179;
        if ( !v180 )
          goto LABEL_90;
        v78 = v180 - 1;
      }
      v79 = v78 & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
      v80 = &v77[2 * v79];
      v81 = (__int64 *)*v80;
      if ( v76 == (__int64 *)*v80 )
      {
LABEL_89:
        *v80 = -8192;
        ++HIDWORD(v178);
        v75 = (unsigned __int64)v181;
        v74 = v182;
        LODWORD(v178) = (2 * ((unsigned int)v178 >> 1) - 2) | v178 & 1;
      }
      else
      {
        v115 = 1;
        while ( v81 != (__int64 *)-4096LL )
        {
          v25 = (unsigned int)(v115 + 1);
          v79 = v78 & (v115 + v79);
          v80 = &v77[2 * v79];
          v81 = (__int64 *)*v80;
          if ( v76 == (__int64 *)*v80 )
            goto LABEL_89;
          v115 = v25;
        }
      }
LABEL_90:
      v82 = (unsigned int)(v74 - 1);
      v83 = v75 + 8 * v82 - 8;
      while ( 1 )
      {
        LODWORD(v182) = v82;
        v74 = v82;
        if ( !(_DWORD)v82 )
          break;
        v83 -= 8;
        if ( *(_QWORD *)(v83 + 8) )
          break;
        LODWORD(v82) = v82 - 1;
      }
      if ( !v159 )
        break;
      v84 = v156;
      v85 = &v156[HIDWORD(v157)];
      if ( v156 == v85 )
        goto LABEL_113;
      while ( *v84 != v76 )
      {
        if ( v85 == ++v84 )
          goto LABEL_113;
      }
      if ( !v74 )
        goto LABEL_100;
    }
    if ( sub_C8CA60((__int64)&v155, (__int64)v76) )
      goto LABEL_114;
LABEL_113:
    if ( v146 == v76 )
      goto LABEL_114;
    *(_QWORD *)(sub_227ED20(v148, &qword_4FDADA8, v76, v145) + 8) = v143;
    sub_227C930(v148, (__int64)v76, (__int64)&v184[3], v89);
    v146 = (__int64 *)v184[2];
    while ( 2 )
    {
      v184[2] = 0;
      if ( !(unsigned __int8)sub_227B670(&v150, *a2, (__int64)v76) )
        goto LABEL_154;
      (*(void (__fastcall **)(__int64 *, __int64, __int64 *, __int64, __int64, _QWORD *))(*(_QWORD *)*a2 + 16LL))(
        &v165,
        *a2,
        v76,
        v148,
        v145,
        v184);
      if ( v184[2] )
      {
        v76 = (__int64 *)v184[2];
        *(_QWORD *)(sub_227ED20(v148, &qword_4FDADA8, (__int64 *)v184[2], v145) + 8) = v143;
      }
      sub_227AD80((__int64)&v184[3], (__int64)&v165, v90, v91, v92, v93);
      sub_227AD80(a1, (__int64)&v165, v94, v95, v96, v97);
      if ( !*(_BYTE *)(v184[1] + 28LL) )
      {
        if ( sub_C8CA60(v184[1], (__int64)v76) )
          goto LABEL_127;
LABEL_138:
        sub_227C930(v148, (__int64)v76, (__int64)&v165, v98);
        if ( v150 )
        {
          v108 = *(_QWORD **)(v150 + 432);
          v149 = &v108[4 * *(unsigned int *)(v150 + 440)];
          if ( v108 != v149 )
          {
            v109 = *a2;
            do
            {
              v151 = 0;
              v110 = (__int64 *)sub_22077B0(0x10u);
              if ( v110 )
              {
                v110[1] = (__int64)v76;
                *v110 = (__int64)&unk_4A08BA8;
              }
              v111 = v151;
              v151 = v110;
              if ( v111 )
                (*(void (__fastcall **)(__int64 *))(*v111 + 8))(v111);
              v112 = v108;
              v114 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v109 + 32LL))(v109);
              if ( (v108[3] & 2) == 0 )
                v112 = (_QWORD *)*v108;
              (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64 **, __int64 *))(v108[3] & 0xFFFFFFFFFFFFFFF8LL))(
                v112,
                v114,
                v113,
                &v151,
                &v165);
              if ( v151 )
                (*(void (__fastcall **)(__int64 *))(*v151 + 8))(v151);
              v108 += 4;
            }
            while ( v149 != v108 );
          }
        }
        if ( !v169 )
          _libc_free(v168);
        if ( !v167 )
          _libc_free(v166);
LABEL_154:
        if ( !v184[2] )
          goto LABEL_114;
        v146 = (__int64 *)v184[2];
        continue;
      }
      break;
    }
    v99 = *(__int64 ***)(v184[1] + 8LL);
    v100 = &v99[*(unsigned int *)(v184[1] + 20LL)];
    if ( v99 == v100 )
      goto LABEL_138;
    while ( v76 != *v99 )
    {
      if ( v100 == ++v99 )
        goto LABEL_138;
    }
LABEL_127:
    if ( v150 )
    {
      v101 = *(_QWORD **)(v150 + 576);
      v102 = &v101[4 * *(unsigned int *)(v150 + 584)];
      if ( v101 != v102 )
      {
        v103 = *a2;
        do
        {
          v104 = v101;
          v106 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v103 + 32LL))(v103);
          v107 = *(void (__fastcall **)(_QWORD *, __int64, __int64, __int64 *))(v101[3] & 0xFFFFFFFFFFFFFFF8LL);
          if ( (v101[3] & 2) == 0 )
            v104 = (_QWORD *)*v101;
          v101 += 4;
          v107(v104, v106, v105, &v165);
        }
        while ( v102 != v101 );
      }
    }
    if ( !v169 )
      _libc_free(v168);
    if ( !v167 )
      _libc_free(v166);
LABEL_114:
    v74 = v182;
    if ( (_DWORD)v182 )
      goto LABEL_86;
LABEL_100:
    ++v161;
    v86 = (unsigned int)v162 >> 1;
    if ( !((unsigned int)v162 >> 1) && !HIDWORD(v162) )
      goto LABEL_108;
    if ( (v162 & 1) != 0 )
    {
      v87 = (__int64 *)&v163;
      v88 = &v165;
      goto LABEL_106;
    }
    if ( 4 * v86 < v164 && v164 > 0x40 )
    {
      if ( v86 && (v125 = v86 - 1) != 0 )
      {
        _BitScanReverse(&v125, v125);
        v126 = 1 << (33 - (v125 ^ 0x1F));
        if ( v126 - 5 <= 0x3A )
        {
          v126 = 64;
          sub_C7D6A0((__int64)v163, 16LL * v164, 8);
          v127 = v162;
          v128 = 1024;
LABEL_184:
          LOBYTE(v162) = v127 & 0xFE;
          v129 = sub_C7D670(v128, 8);
          v164 = v126;
          v163 = (__int64 *)v129;
          goto LABEL_185;
        }
        if ( v164 == v126 )
        {
          v162 &= 1u;
          if ( v162 )
          {
            v136 = (__int64 *)&v163;
            v137 = &v165;
          }
          else
          {
            v136 = v163;
            v137 = &v163[2 * v164];
          }
          do
          {
            if ( v136 )
            {
              *v136 = -4096;
              v136[1] = -4096;
            }
            v136 += 2;
          }
          while ( v136 != v137 );
          goto LABEL_108;
        }
        sub_C7D6A0((__int64)v163, 16LL * v164, 8);
        v127 = v162 | 1;
        LOBYTE(v162) = v162 | 1;
        if ( v126 > 4 )
        {
          v128 = 16LL * v126;
          goto LABEL_184;
        }
      }
      else
      {
        sub_C7D6A0((__int64)v163, 16LL * v164, 8);
        LOBYTE(v162) = v162 | 1;
      }
LABEL_185:
      v162 &= 1u;
      if ( v162 )
      {
        v130 = (__int64 *)&v163;
        v131 = &v165;
      }
      else
      {
        v130 = v163;
        v131 = &v163[2 * v164];
        if ( v163 == v131 )
          goto LABEL_108;
      }
      do
      {
        if ( v130 )
        {
          *v130 = -4096;
          v130[1] = -4096;
        }
        v130 += 2;
      }
      while ( v131 != v130 );
      goto LABEL_108;
    }
    v87 = v163;
    v88 = &v163[2 * v164];
    if ( v163 != v88 )
    {
      do
      {
LABEL_106:
        *v87 = -4096;
        v87 += 2;
        *(v87 - 1) = -4096;
      }
      while ( v87 != v88 );
    }
    v162 &= 1u;
LABEL_108:
    v40 = v175;
    if ( (_DWORD)v175 )
    {
      v62 = (unsigned __int64)v174;
      v63 = *(_QWORD *)&v174[8 * (unsigned int)v175 - 8];
      if ( (v171 & 1) != 0 )
        goto LABEL_76;
      goto LABEL_110;
    }
    if ( v142 )
    {
      v22 = v142;
      LODWORD(v24) = *(_DWORD *)(v141 + 608);
      v20 = *(_DWORD *)(v141 + 448);
      v23 = *(_QWORD *)(v141 + 592);
      if ( (_DWORD)v24 )
        goto LABEL_56;
      continue;
    }
    break;
  }
LABEL_26:
  sub_D2DA90(v145, (__int64)v152, (unsigned int)v153);
  v33 = (_QWORD **)v152;
  v34 = (_QWORD **)&v152[8 * (unsigned int)v153];
  if ( v34 != (_QWORD **)v152 )
  {
    do
    {
      v35 = *v33++;
      sub_B2E860(v35);
    }
    while ( v34 != v33 );
  }
  if ( *(_DWORD *)(a1 + 68) != *(_DWORD *)(a1 + 72)
    || !(unsigned __int8)sub_B19060(a1, (__int64)&qword_4F82400, v31, v32) )
  {
    sub_AE6EC0(a1, (__int64)&unk_4FDADC8);
  }
  sub_227AC60(a1, (__int64)&unk_4F86C48);
  sub_227AC60(a1, (__int64)&unk_4FDADC0);
  sub_227AC60(a1, (__int64)&unk_4F82418);
  v36 = (_QWORD *)v184[51];
  v37 = (_QWORD *)(v184[51] + 32LL * LODWORD(v184[52]));
  if ( (_QWORD *)v184[51] != v37 )
  {
    do
    {
      v38 = *(v37 - 1);
      v37 -= 4;
      if ( v38 != -4096 && v38 != 0 && v38 != -8192 )
        sub_BD60C0(v37 + 1);
    }
    while ( v36 != v37 );
    v37 = (_QWORD *)v184[51];
  }
  if ( v37 != &v184[53] )
    _libc_free((unsigned __int64)v37);
  if ( (v184[18] & 1) == 0 )
    sub_C7D6A0(v184[19], 16LL * LODWORD(v184[20]), 8);
  sub_227AD40((__int64)&v184[3]);
  if ( v152 != v154 )
    _libc_free((unsigned __int64)v152);
  if ( (v162 & 1) == 0 )
    sub_C7D6A0((__int64)v163, 16LL * v164, 8);
  if ( !v159 )
    _libc_free((unsigned __int64)v156);
  if ( v181 != v183 )
    _libc_free((unsigned __int64)v181);
  if ( (v178 & 1) == 0 )
    sub_C7D6A0((__int64)v179, 16LL * v180, 8);
  if ( v174 != v176 )
    _libc_free((unsigned __int64)v174);
  if ( (v171 & 1) == 0 )
    sub_C7D6A0((__int64)v172, 16LL * v173, 8);
  return a1;
}
