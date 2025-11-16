// Function: sub_34B7280
// Address: 0x34b7280
//
__int64 __fastcall sub_34B7280(__int64 a1, __int64 *a2, __int64 a3, unsigned __int64 a4, int a5, __int64 *a6)
{
  unsigned __int64 *v6; // r13
  unsigned __int64 *v7; // rbx
  __int64 v8; // rax
  char v9; // di
  unsigned __int64 v10; // r12
  __int64 v11; // rax
  unsigned __int64 v12; // r13
  _QWORD *v13; // rax
  __int64 *v14; // rdx
  _QWORD *v15; // rdi
  _QWORD *v16; // rsi
  _QWORD *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  int v22; // r13d
  unsigned int v23; // ebx
  unsigned __int64 v24; // r14
  _QWORD *v25; // rax
  _QWORD *v26; // rdx
  __int64 v27; // rax
  __int64 j; // r14
  __int64 v30; // r13
  __int64 v31; // r12
  __int64 i; // rbx
  unsigned int v33; // r14d
  __int64 v34; // r8
  unsigned __int64 **v35; // rax
  unsigned __int64 **v36; // rsi
  unsigned __int64 *v37; // rcx
  unsigned __int64 *v38; // rdx
  unsigned __int64 *v39; // rax
  unsigned __int64 v40; // rbx
  __int64 v41; // r12
  __int64 v42; // rax
  __int64 v43; // r9
  unsigned __int64 *v44; // r14
  unsigned __int64 *v45; // rax
  char v46; // dl
  unsigned __int64 v47; // rdi
  __int64 v48; // r8
  __int64 v49; // r9
  _QWORD *v50; // rax
  _QWORD *v51; // rcx
  bool v52; // zf
  __int64 v53; // rax
  _QWORD *v54; // rdi
  _QWORD *v55; // r12
  __int64 **v56; // r13
  __int64 v57; // r15
  __int64 v58; // rax
  unsigned int v59; // r14d
  _QWORD *v60; // rbx
  __int64 v61; // rax
  unsigned __int64 v62; // rax
  int *v63; // rsi
  __int64 v64; // rcx
  __int64 v65; // rdx
  unsigned int v66; // eax
  __int64 v67; // rax
  unsigned __int64 v68; // rdx
  unsigned __int64 v69; // rcx
  unsigned __int64 v70; // r15
  unsigned __int64 v71; // rsi
  __int64 v72; // rdi
  __int64 v73; // rsi
  __int64 v74; // rcx
  __int64 v75; // rdx
  __int64 v76; // rcx
  unsigned int v77; // edx
  unsigned int v78; // eax
  unsigned int *v79; // rbx
  char v80; // r14
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 v83; // rdx
  _QWORD *v84; // r12
  __int64 v85; // rbx
  __int64 v86; // rax
  __int64 v87; // rcx
  __int64 v88; // rdx
  unsigned int v89; // r15d
  _QWORD *v90; // r14
  int v91; // r13d
  unsigned __int64 v92; // r12
  unsigned int v93; // eax
  _QWORD *v94; // rdx
  __int64 v95; // r15
  __int64 v96; // rax
  __int64 v97; // rdx
  __int64 v98; // r15
  __int64 *v99; // r9
  unsigned __int64 v100; // rsi
  unsigned __int64 **v101; // rax
  unsigned __int64 *v102; // rcx
  unsigned __int64 *v103; // rdx
  __int64 v104; // r9
  int v105; // esi
  __int64 v106; // r10
  __int64 v107; // rax
  __int64 v108; // rcx
  __int64 v109; // rdx
  __int16 v110; // dx
  __int64 v111; // rdi
  __int64 v112; // rdx
  __int64 v113; // [rsp+10h] [rbp-280h]
  __int64 v114; // [rsp+18h] [rbp-278h]
  __int64 v115; // [rsp+20h] [rbp-270h]
  __int64 v116; // [rsp+28h] [rbp-268h]
  __int64 v117; // [rsp+38h] [rbp-258h]
  __int64 v118; // [rsp+48h] [rbp-248h]
  unsigned __int64 v120; // [rsp+68h] [rbp-228h]
  unsigned __int64 v121; // [rsp+70h] [rbp-220h]
  unsigned __int64 *v122; // [rsp+78h] [rbp-218h]
  _QWORD *v123; // [rsp+80h] [rbp-210h]
  __int64 v124; // [rsp+90h] [rbp-200h]
  unsigned __int64 v125; // [rsp+98h] [rbp-1F8h]
  __int64 v126; // [rsp+98h] [rbp-1F8h]
  __int64 v128; // [rsp+C0h] [rbp-1D0h]
  __int64 v131; // [rsp+D0h] [rbp-1C0h]
  unsigned int *v132; // [rsp+D0h] [rbp-1C0h]
  unsigned __int64 v133; // [rsp+D0h] [rbp-1C0h]
  int v134; // [rsp+D0h] [rbp-1C0h]
  unsigned int v135; // [rsp+D8h] [rbp-1B8h]
  int v137; // [rsp+DCh] [rbp-1B4h]
  __int64 v139; // [rsp+E0h] [rbp-1B0h]
  unsigned __int64 *v140; // [rsp+E8h] [rbp-1A8h]
  unsigned __int64 v141; // [rsp+E8h] [rbp-1A8h]
  unsigned __int64 **v142; // [rsp+E8h] [rbp-1A8h]
  __int64 v143; // [rsp+E8h] [rbp-1A8h]
  unsigned int v144; // [rsp+F8h] [rbp-198h] BYREF
  unsigned int v145; // [rsp+FCh] [rbp-194h] BYREF
  unsigned __int64 v146; // [rsp+100h] [rbp-190h] BYREF
  unsigned __int64 *v147; // [rsp+108h] [rbp-188h] BYREF
  __int64 **v148; // [rsp+110h] [rbp-180h] BYREF
  unsigned __int64 **v149; // [rsp+118h] [rbp-178h]
  unsigned __int64 **v150; // [rsp+120h] [rbp-170h]
  __int64 v151; // [rsp+130h] [rbp-160h] BYREF
  int v152; // [rsp+138h] [rbp-158h] BYREF
  unsigned __int64 v153; // [rsp+140h] [rbp-150h]
  int *v154; // [rsp+148h] [rbp-148h]
  int *v155; // [rsp+150h] [rbp-140h]
  __int64 v156; // [rsp+158h] [rbp-138h]
  __int64 v157; // [rsp+160h] [rbp-130h] BYREF
  __int64 v158; // [rsp+168h] [rbp-128h] BYREF
  unsigned __int64 **v159; // [rsp+170h] [rbp-120h]
  __int64 *v160; // [rsp+178h] [rbp-118h]
  __int64 *v161; // [rsp+180h] [rbp-110h]
  __int64 v162; // [rsp+188h] [rbp-108h]
  __int64 v163; // [rsp+190h] [rbp-100h] BYREF
  int v164; // [rsp+198h] [rbp-F8h] BYREF
  unsigned __int64 v165; // [rsp+1A0h] [rbp-F0h]
  int *v166; // [rsp+1A8h] [rbp-E8h]
  int *v167; // [rsp+1B0h] [rbp-E0h]
  __int64 v168; // [rsp+1B8h] [rbp-D8h]
  void *v169; // [rsp+1C0h] [rbp-D0h] BYREF
  __int64 v170; // [rsp+1C8h] [rbp-C8h]
  _BYTE v171[48]; // [rsp+1D0h] [rbp-C0h] BYREF
  int v172; // [rsp+200h] [rbp-90h]
  unsigned __int64 *v173; // [rsp+210h] [rbp-80h] BYREF
  __int64 v174; // [rsp+218h] [rbp-78h] BYREF
  unsigned __int64 v175; // [rsp+220h] [rbp-70h] BYREF
  __int64 *v176; // [rsp+228h] [rbp-68h]
  __int64 *v177; // [rsp+230h] [rbp-60h] BYREF
  __int64 v178; // [rsp+238h] [rbp-58h] BYREF
  unsigned __int64 v179; // [rsp+240h] [rbp-50h]
  __int64 *v180; // [rsp+248h] [rbp-48h]
  __int64 *v181; // [rsp+250h] [rbp-40h]
  __int64 v182; // [rsp+258h] [rbp-38h]

  v6 = (unsigned __int64 *)a2[1];
  v7 = (unsigned __int64 *)*a2;
  if ( (unsigned __int64 *)*a2 == v6 )
    return 0;
  v8 = *(_QWORD *)(a1 + 120);
  v152 = 0;
  v153 = 0;
  v118 = v8;
  v154 = &v152;
  v155 = &v152;
  v156 = 0;
  LODWORD(v158) = 0;
  v159 = 0;
  v160 = &v158;
  v161 = &v158;
  v162 = 0;
  v140 = v6;
  do
  {
    while ( 1 )
    {
      v10 = *v7;
      v11 = sub_22077B0(0x30u);
      *(_QWORD *)(v11 + 32) = v10;
      v12 = v11;
      *(_QWORD *)(v11 + 40) = v7;
      v13 = sub_34B6150((__int64)&v157, (unsigned __int64 *)(v11 + 32));
      if ( v14 )
        break;
      v7 += 32;
      j_j___libc_free_0(v12);
      if ( v140 == v7 )
        goto LABEL_9;
    }
    v9 = v13 || v14 == &v158 || v10 < v14[4];
    v7 += 32;
    sub_220F040(v9, v12, v14, &v158);
    ++v162;
  }
  while ( v140 != v7 );
LABEL_9:
  v15 = *(_QWORD **)(a1 + 48);
  v16 = &v15[*(unsigned int *)(a1 + 56)];
  v17 = sub_34B2960(v15, (__int64)v16);
  if ( (_QWORD *)v19 == v17 )
  {
    v124 = 0;
    v121 = 0;
  }
  else
  {
    v30 = a2[1];
    v31 = *a2;
    if ( *a2 == v30 )
      BUG();
    for ( i = *a2 + 256; i != v30; i += 256 )
    {
      if ( v31 )
      {
        if ( (*(_BYTE *)(i + 254) & 1) == 0 )
          sub_2F8F5D0(i, v16, v18, v19, v20, v21);
        v33 = *(_DWORD *)(i + 240) + *(unsigned __int16 *)(i + 252);
        if ( (*(_BYTE *)(v31 + 254) & 1) == 0 )
          sub_2F8F5D0(v31, v16, v18, v19, v20, v21);
        if ( v33 > *(_DWORD *)(v31 + 240) + (unsigned int)*(unsigned __int16 *)(v31 + 252) )
          v31 = i;
      }
      else
      {
        v31 = i;
      }
    }
    v121 = v31;
    v124 = *(_QWORD *)v31;
  }
  v22 = *(_DWORD *)(*(_QWORD *)(a1 + 32) + 16LL);
  v169 = v171;
  v23 = (unsigned int)(v22 + 63) >> 6;
  v170 = 0x600000000LL;
  if ( v23 > 6 )
  {
    sub_C8D5F0((__int64)&v169, v171, v23, 8u, v20, v21);
    memset(v169, 0, 8LL * v23);
    LODWORD(v170) = (unsigned int)(v22 + 63) >> 6;
  }
  else
  {
    if ( v23 && 8LL * v23 )
      memset(v171, 0, 8LL * v23);
    LODWORD(v170) = (unsigned int)(v22 + 63) >> 6;
  }
  v24 = a4;
  v172 = v22;
  v135 = 0;
  v137 = a5 - 1;
  if ( a4 == a3 )
    goto LABEL_26;
  v139 = a1;
  do
  {
    v25 = (_QWORD *)(*(_QWORD *)v24 & 0xFFFFFFFFFFFFFFF8LL);
    v26 = v25;
    if ( !v25 )
      BUG();
    v24 = *(_QWORD *)v24 & 0xFFFFFFFFFFFFFFF8LL;
    v27 = *v25;
    if ( (v27 & 4) == 0 && (*((_BYTE *)v26 + 44) & 4) != 0 )
    {
      for ( j = v27; ; j = *(_QWORD *)v24 )
      {
        v24 = j & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v24 + 44) & 4) == 0 )
          break;
      }
    }
    if ( (unsigned __int16)(*(_WORD *)(v24 + 68) - 14) <= 4u )
      goto LABEL_25;
    v166 = &v164;
    v167 = &v164;
    v164 = 0;
    v165 = 0;
    v168 = 0;
    sub_34B3C60(v139, v24, &v163);
    sub_34B48B0(v139, v24, v137, (__int64)&v163);
    v35 = v159;
    v148 = 0;
    v147 = (unsigned __int64 *)v24;
    v36 = (unsigned __int64 **)&v158;
    v149 = 0;
    v150 = 0;
    if ( !v159 )
      goto LABEL_49;
    do
    {
      while ( 1 )
      {
        v37 = v35[2];
        v38 = v35[3];
        if ( (unsigned __int64)v35[4] >= v24 )
          break;
        v35 = (unsigned __int64 **)v35[3];
        if ( !v38 )
          goto LABEL_47;
      }
      v36 = v35;
      v35 = (unsigned __int64 **)v35[2];
    }
    while ( v37 );
LABEL_47:
    if ( v36 == (unsigned __int64 **)&v158 || (unsigned __int64)v36[4] > v24 )
    {
LABEL_49:
      v173 = (unsigned __int64 *)&v147;
      v36 = (unsigned __int64 **)sub_34B62F0(&v157, (__int64)v36, &v173);
    }
    v39 = v36[5];
    LODWORD(v178) = 0;
    v173 = &v175;
    v174 = 0x400000000LL;
    v179 = 0;
    v180 = &v178;
    v181 = &v178;
    v182 = 0;
    v40 = v39[5];
    v122 = v39;
    v41 = 16LL * *((unsigned int *)v39 + 12);
    v141 = v40 + v41;
    if ( v40 != v40 + v41 )
    {
      v125 = v24;
      while ( 1 )
      {
        while ( 1 )
        {
          v42 = (*(__int64 *)v40 >> 1) & 3;
          if ( v42 != 1 && v42 != 2 )
            goto LABEL_53;
          v43 = *(unsigned int *)(v40 + 8);
          LODWORD(v147) = *(_DWORD *)(v40 + 8);
          if ( !v182 )
            break;
          v36 = &v147;
          sub_2DCBE50((__int64)&v177, (unsigned int *)&v147);
          if ( v46 )
            goto LABEL_66;
LABEL_53:
          v40 += 16LL;
          if ( v141 == v40 )
            goto LABEL_70;
        }
        v44 = (unsigned __int64 *)((char *)v173 + 4 * (unsigned int)v174);
        if ( v173 == v44 )
        {
          if ( (unsigned int)v174 > 3uLL )
            goto LABEL_132;
        }
        else
        {
          v45 = v173;
          while ( (_DWORD)v43 != *(_DWORD *)v45 )
          {
            v45 = (unsigned __int64 *)((char *)v45 + 4);
            if ( v44 == v45 )
              goto LABEL_61;
          }
          if ( v45 != v44 )
            goto LABEL_53;
LABEL_61:
          if ( (unsigned int)v174 > 3uLL )
          {
            v120 = v40;
            v79 = (unsigned int *)v173;
            v132 = (unsigned int *)v173 + (unsigned int)v174;
            do
            {
              v82 = sub_2DCC990(&v177, (__int64)&v178, v79);
              v84 = (_QWORD *)v83;
              if ( v83 )
              {
                v80 = v82 || (__int64 *)v83 == &v178 || *v79 < *(_DWORD *)(v83 + 32);
                v81 = sub_22077B0(0x28u);
                *(_DWORD *)(v81 + 32) = *v79;
                sub_220F040(v80, v81, v84, &v178);
                ++v182;
              }
              ++v79;
            }
            while ( v132 != v79 );
            v40 = v120;
LABEL_132:
            LODWORD(v174) = 0;
            sub_2DCBE50((__int64)&v177, (unsigned int *)&v147);
            goto LABEL_66;
          }
        }
        if ( (unsigned __int64)(unsigned int)v174 + 1 > HIDWORD(v174) )
        {
          v134 = v43;
          sub_C8D5F0((__int64)&v173, &v175, (unsigned int)v174 + 1LL, 4u, v34, v43);
          LODWORD(v43) = v134;
          v44 = (unsigned __int64 *)((char *)v173 + 4 * (unsigned int)v174);
        }
        *(_DWORD *)v44 = v43;
        LODWORD(v174) = v174 + 1;
LABEL_66:
        v147 = (unsigned __int64 *)v40;
        v36 = v149;
        if ( v149 == v150 )
        {
          sub_34B5D70((__int64)&v148, v149, &v147);
          goto LABEL_53;
        }
        if ( v149 )
        {
          *v149 = (unsigned __int64 *)v40;
          v36 = v149;
        }
        ++v36;
        v40 += 16LL;
        v149 = v36;
        if ( v141 == v40 )
        {
LABEL_70:
          v24 = v125;
          v47 = v179;
          goto LABEL_71;
        }
      }
    }
    v47 = 0;
LABEL_71:
    sub_34B2A20(v47);
    if ( v173 != &v175 )
      _libc_free((unsigned __int64)v173);
    if ( v24 == v124 )
    {
      if ( !v121 )
        goto LABEL_152;
      v85 = *(_QWORD *)(v121 + 40);
      v86 = 16LL * *(unsigned int *)(v121 + 48);
      v87 = v85 + v86;
      if ( v85 == v85 + v86 )
        goto LABEL_152;
      v88 = 0;
      v133 = v24;
      v89 = 0;
      v90 = 0;
      do
      {
        while ( 1 )
        {
          v91 = *(_DWORD *)(v85 + 12);
          v92 = *(_QWORD *)v85 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v92 + 254) & 1) == 0 )
          {
            v143 = v87;
            sub_2F8F5D0(*(_QWORD *)v85 & 0xFFFFFFFFFFFFFFF8LL, v36, v88, v87, v48, v49);
            v87 = v143;
          }
          v93 = v91 + *(_DWORD *)(v92 + 240);
          if ( v89 >= v93 )
            break;
          v90 = (_QWORD *)v85;
          v85 += 16;
          v89 = v91 + *(_DWORD *)(v92 + 240);
          if ( v87 == v85 )
            goto LABEL_146;
        }
        if ( v89 == v93 && ((*(__int64 *)v85 >> 1) & 3) == 1 )
          v90 = (_QWORD *)v85;
        v85 += 16;
      }
      while ( v87 != v85 );
LABEL_146:
      v94 = v90;
      v24 = v133;
      if ( v94 && (v121 = *v94 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v54 = 0;
        v124 = *(_QWORD *)(*v94 & 0xFFFFFFFFFFFFFFF8LL);
      }
      else
      {
LABEL_152:
        v124 = 0;
        v54 = 0;
        v121 = 0;
      }
    }
    else
    {
      v50 = sub_34B2960(*(_QWORD **)(v139 + 48), *(_QWORD *)(v139 + 48) + 8LL * *(unsigned int *)(v139 + 56));
      v52 = v51 == v50;
      v53 = 0;
      if ( !v52 )
        v53 = v139 + 48;
      v54 = (_QWORD *)v53;
    }
    if ( *(_WORD *)(v24 + 68) == 7 )
      goto LABEL_78;
    v142 = v149;
    if ( v148 == (__int64 **)v149 )
      goto LABEL_78;
    v131 = v24;
    v55 = v54;
    v56 = v148;
    while ( 2 )
    {
      v57 = **v56;
      v58 = (v57 >> 1) & 3;
      if ( v58 != 2 && v58 != 1 )
        goto LABEL_84;
      v59 = *((_DWORD *)*v56 + 2);
      v60 = *(_QWORD **)(v139 + 16);
      if ( !*(_BYTE *)(*(_QWORD *)(*(_QWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v60 + 16LL) + 200LL))(*(_QWORD *)(*v60 + 16LL))
                                             + 248)
                                 + 16LL)
                     + v59) )
        goto LABEL_84;
      v61 = v59 >> 6;
      if ( (*(_QWORD *)(v60[48] + 8 * v61) & (1LL << v59)) != 0
        || v55 && (*(_QWORD *)(*v55 + 8 * v61) & (1LL << v59)) != 0 )
      {
        goto LABEL_84;
      }
      v62 = v165;
      if ( v165 )
      {
        v63 = &v164;
        do
        {
          while ( 1 )
          {
            v64 = *(_QWORD *)(v62 + 16);
            v65 = *(_QWORD *)(v62 + 24);
            if ( v59 <= *(_DWORD *)(v62 + 32) )
              break;
            v62 = *(_QWORD *)(v62 + 24);
            if ( !v65 )
              goto LABEL_95;
          }
          v63 = (int *)v62;
          v62 = *(_QWORD *)(v62 + 16);
        }
        while ( v64 );
LABEL_95:
        if ( v63 != &v164 && v59 >= v63[8] )
          goto LABEL_84;
      }
      v66 = sub_2E8E710(v131, v59, 0, 0, 0);
      if ( v66 == -1 )
        goto LABEL_84;
      v67 = *(_QWORD *)(v131 + 32) + 40LL * v66;
      if ( !v67 || (*(_BYTE *)(v67 + 3) & 0x20) != 0 )
        goto LABEL_84;
      v68 = v122[5];
      v69 = v68 + 16LL * *((unsigned int *)v122 + 12);
      if ( v69 != v68 )
      {
        v70 = v57 & 0xFFFFFFFFFFFFFFF8LL;
        v71 = v122[5];
        while ( 1 )
        {
          v72 = (*(__int64 *)v71 >> 1) & 3;
          if ( v70 != (*(_QWORD *)v71 & 0xFFFFFFFFFFFFFFF8LL) )
            break;
          if ( v72 != 1 || v59 != *(_DWORD *)(v71 + 8) )
            goto LABEL_108;
LABEL_104:
          v71 += 16LL;
          if ( v69 == v71 )
            goto LABEL_112;
        }
        if ( v72 || v59 != *(_DWORD *)(v71 + 8) )
          goto LABEL_104;
LABEL_108:
        v59 = 0;
LABEL_112:
        while ( 1 )
        {
          v73 = (*(__int64 *)v68 >> 1) & 3;
          if ( v70 != (*(_QWORD *)v68 & 0xFFFFFFFFFFFFFFF8LL) )
            break;
          if ( v73 == 1 )
          {
LABEL_111:
            v68 += 16LL;
            if ( v69 == v68 )
              goto LABEL_116;
          }
          else
          {
            if ( v73 != 2 )
              goto LABEL_84;
            v68 += 16LL;
            if ( v69 == v68 )
              goto LABEL_116;
          }
        }
        if ( !v73 && *(_DWORD *)(v68 + 8) == v59 )
          goto LABEL_84;
        goto LABEL_111;
      }
LABEL_116:
      if ( !v59 )
        goto LABEL_84;
      v74 = *(_QWORD *)(v139 + 120);
      v75 = *(_QWORD *)(v74 + 32);
      v76 = *(_QWORD *)(v74 + 8);
      v77 = *(_DWORD *)(v75 + 4LL * v59);
      do
      {
        v78 = v77;
        v77 = *(_DWORD *)(v76 + 4LL * v77);
      }
      while ( v78 != v77 );
      if ( !v77 )
        goto LABEL_84;
      LODWORD(v174) = 0;
      v175 = 0;
      v176 = &v174;
      v177 = &v174;
      v178 = 0;
      if ( !(unsigned __int8)sub_34B6620((__int64 *)v139, v59, v77, &v151, &v173) )
        goto LABEL_121;
      v126 = (__int64)v176;
      v123 = (_QWORD *)(v118 + 56);
      if ( v176 == &v174 )
        goto LABEL_182;
      v95 = v118;
      do
      {
        v144 = *(_DWORD *)(v126 + 32);
        v145 = *(_DWORD *)(v126 + 36);
        v96 = sub_34B4480((__int64)v123, &v144);
        v128 = v97;
        if ( v97 == v96 )
          goto LABEL_181;
        v117 = v95;
        v98 = v96;
        do
        {
          sub_2EAB0C0(*(_QWORD *)(v98 + 40), v145);
          v99 = &v158;
          v100 = *(_QWORD *)(*(_QWORD *)(v98 + 40) + 16LL);
          v101 = v159;
          v146 = v100;
          if ( !v159 )
            goto LABEL_165;
          do
          {
            while ( 1 )
            {
              v102 = v101[2];
              v103 = v101[3];
              if ( (unsigned __int64)v101[4] >= v100 )
                break;
              v101 = (unsigned __int64 **)v101[3];
              if ( !v103 )
                goto LABEL_163;
            }
            v99 = (__int64 *)v101;
            v101 = (unsigned __int64 **)v101[2];
          }
          while ( v102 );
LABEL_163:
          if ( v99 == &v158 || v99[4] > v100 )
          {
LABEL_165:
            v147 = &v146;
            v99 = (__int64 *)sub_34B62F0(&v157, (__int64)v99, &v147);
          }
          if ( v99[5] )
          {
            v104 = *a6;
            v105 = v145;
            v106 = *(_QWORD *)(*(_QWORD *)(v98 + 40) + 16LL);
            v107 = a6[1];
            if ( *a6 != v107 )
            {
              v108 = 0;
              while ( 1 )
              {
                v109 = *(_QWORD *)(v107 - 8);
                if ( v106 != v109 && v108 != v109 )
                {
                  if ( v108 )
                    goto LABEL_179;
                  goto LABEL_171;
                }
                v108 = *(_QWORD *)(v107 - 16);
                v110 = *(_WORD *)(v108 + 68);
                if ( v110 == 14 )
                  break;
                if ( v110 == 15 )
                {
                  v112 = *(_QWORD *)(v108 + 32);
                  if ( *(_BYTE *)(v112 + 80) )
                    goto LABEL_171;
                  v111 = v112 + 80;
                  if ( v59 != *(_DWORD *)(v112 + 88) )
                    goto LABEL_171;
LABEL_178:
                  v113 = v104;
                  v114 = *(_QWORD *)(v107 - 16);
                  v115 = v107;
                  v116 = v106;
                  sub_2EAB0C0(v111, v105);
                  v104 = v113;
                  v106 = v116;
                  v107 = v115 - 16;
                  v108 = v114;
                  if ( v113 == v115 - 16 )
                    goto LABEL_179;
                }
                else
                {
                  if ( v110 != 17 )
                    BUG();
                  v111 = *(_QWORD *)(v108 + 32);
                  if ( !*(_BYTE *)v111 && *(_DWORD *)(v111 + 8) == v59 )
                    goto LABEL_178;
LABEL_171:
                  v107 -= 16;
                  if ( v104 == v107 )
                    goto LABEL_179;
                }
              }
              v111 = *(_QWORD *)(v108 + 32);
              if ( *(_BYTE *)v111 || v59 != *(_DWORD *)(v111 + 8) )
                goto LABEL_171;
              goto LABEL_178;
            }
          }
LABEL_179:
          v98 = sub_220EEE0(v98);
        }
        while ( v128 != v98 );
        v95 = v117;
LABEL_181:
        sub_34B3410(*(_QWORD **)(v139 + 120), v145, 0);
        sub_34B4520(v123, &v145);
        *(_DWORD *)(*(_QWORD *)(v95 + 128) + 4LL * v145) = *(_DWORD *)(*(_QWORD *)(v95 + 128) + 4LL * v144);
        *(_DWORD *)(*(_QWORD *)(v95 + 104) + 4LL * v145) = *(_DWORD *)(*(_QWORD *)(v95 + 104) + 4LL * v144);
        sub_34B3410(*(_QWORD **)(v139 + 120), v144, 0);
        sub_34B4520(v123, &v144);
        *(_DWORD *)(*(_QWORD *)(v95 + 128) + 4LL * v144) = *(_DWORD *)(*(_QWORD *)(v95 + 104) + 4LL * v144);
        *(_DWORD *)(*(_QWORD *)(v95 + 104) + 4LL * v144) = -1;
        v126 = sub_220EEE0(v126);
      }
      while ( (__int64 *)v126 != &v174 );
LABEL_182:
      ++v135;
LABEL_121:
      sub_34B5950((__int64)&v173, v175);
LABEL_84:
      if ( v142 != (unsigned __int64 **)++v56 )
        continue;
      break;
    }
    v24 = v131;
LABEL_78:
    sub_34B4E10((_QWORD *)v139, v24, v137);
    if ( v148 )
      j_j___libc_free_0((unsigned __int64)v148);
    sub_34B4200((__int64)&v163, v165);
LABEL_25:
    --v137;
  }
  while ( a3 != v24 );
LABEL_26:
  if ( v169 != v171 )
    _libc_free((unsigned __int64)v169);
  sub_34B5770((__int64)&v157, (unsigned __int64)v159);
  sub_34B5590((__int64)&v151, v153);
  return v135;
}
