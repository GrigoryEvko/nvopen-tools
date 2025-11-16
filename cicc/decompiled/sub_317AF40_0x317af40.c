// Function: sub_317AF40
// Address: 0x317af40
//
__int64 __fastcall sub_317AF40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v6; // r13
  _QWORD *v7; // rbx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // r12
  __int64 v13; // r13
  char v14; // al
  __int64 *v15; // rax
  _BYTE *v16; // rbx
  unsigned __int64 v17; // r14
  __int64 v18; // r12
  __int64 v19; // r13
  _QWORD **v20; // rdi
  unsigned __int64 v21; // rsi
  unsigned __int64 v22; // rdx
  __int64 *v23; // rcx
  __int64 v24; // r8
  __int64 *v25; // r9
  __int64 v26; // rax
  __int64 *v27; // rax
  __int64 v28; // r13
  unsigned int v29; // ebx
  int v30; // eax
  int v31; // edx
  unsigned int v32; // eax
  unsigned int *v33; // rbx
  unsigned int v34; // edx
  __int64 v35; // rax
  __int64 v36; // rbx
  __int64 v37; // r13
  unsigned __int64 v38; // rdi
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 *v42; // r11
  _QWORD *v43; // r14
  __int64 v44; // rax
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // rbx
  __int64 v48; // rax
  __int64 v49; // rbx
  __int64 *v50; // rsi
  __int64 v51; // rax
  __int64 *v52; // r8
  void (__fastcall *v53)(__int64 *, __int64, __int64); // rax
  __int64 v54; // rax
  unsigned __int64 v55; // rbx
  int v56; // r13d
  __int64 *v57; // r12
  signed __int64 v58; // rax
  int v59; // edx
  bool v60; // of
  unsigned __int8 *v61; // rdx
  __int64 v62; // rsi
  unsigned __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // r8
  __int64 v66; // r9
  __int64 v67; // rcx
  unsigned __int64 v68; // rax
  unsigned int v69; // ebx
  unsigned __int64 v70; // rax
  int v71; // edx
  __int64 *v72; // rax
  __int64 v73; // rcx
  int v74; // eax
  __int64 v75; // rdx
  __int64 v76; // rbx
  __int64 v77; // rax
  __int64 v78; // rcx
  __int64 v79; // rax
  __int64 v80; // rbx
  __int64 v81; // rax
  __int64 v82; // r8
  __int64 v83; // r9
  __int64 v84; // rax
  bool v85; // zf
  __int64 v86; // r8
  _DWORD *v87; // rax
  int v88; // esi
  int v89; // edx
  __int64 v90; // rcx
  __int64 v91; // r9
  __int64 v92; // rdx
  int v93; // r11d
  __int64 v94; // r9
  int v95; // r11d
  int v96; // esi
  __int64 v97; // rcx
  unsigned int v98; // eax
  __int64 v99; // rdx
  __int64 *v100; // rdi
  _QWORD *v101; // rax
  __int64 v102; // rdx
  __int64 v103; // rdx
  __int64 v104; // rcx
  __int64 v105; // rbx
  __int64 v106; // r8
  __int64 v107; // r9
  __int64 v108; // rax
  __int64 v109; // rdx
  int v110; // eax
  int v111; // eax
  __int64 v112; // rax
  unsigned int v113; // eax
  int v114; // edx
  __int64 v115; // [rsp+8h] [rbp-288h]
  __int64 v116; // [rsp+8h] [rbp-288h]
  __int64 v117; // [rsp+10h] [rbp-280h]
  unsigned int v118; // [rsp+10h] [rbp-280h]
  _DWORD *v119; // [rsp+10h] [rbp-280h]
  unsigned __int64 v120; // [rsp+18h] [rbp-278h]
  __int64 *v121; // [rsp+20h] [rbp-270h]
  __int64 *v122; // [rsp+28h] [rbp-268h]
  __int64 v123; // [rsp+28h] [rbp-268h]
  __int64 v124; // [rsp+30h] [rbp-260h]
  __int64 *v125; // [rsp+30h] [rbp-260h]
  int v126; // [rsp+30h] [rbp-260h]
  unsigned int v127; // [rsp+30h] [rbp-260h]
  int v128; // [rsp+30h] [rbp-260h]
  __int64 v129; // [rsp+38h] [rbp-258h]
  __int64 v130; // [rsp+38h] [rbp-258h]
  __int64 v131; // [rsp+38h] [rbp-258h]
  unsigned int v132; // [rsp+38h] [rbp-258h]
  int v133; // [rsp+38h] [rbp-258h]
  int v134; // [rsp+38h] [rbp-258h]
  unsigned int v136; // [rsp+54h] [rbp-23Ch]
  __int64 *v138; // [rsp+68h] [rbp-228h] BYREF
  unsigned int v139; // [rsp+74h] [rbp-21Ch] BYREF
  __int64 v140; // [rsp+78h] [rbp-218h] BYREF
  __int64 v141; // [rsp+80h] [rbp-210h] BYREF
  __int64 v142; // [rsp+88h] [rbp-208h]
  __int64 v143; // [rsp+90h] [rbp-200h]
  unsigned int v144; // [rsp+98h] [rbp-1F8h]
  __int64 v145[2]; // [rsp+A0h] [rbp-1F0h] BYREF
  void (__fastcall *v146)(__int64 *, __int64, __int64); // [rsp+B0h] [rbp-1E0h]
  __int64 v147; // [rsp+B8h] [rbp-1D8h]
  _BYTE *v148; // [rsp+C0h] [rbp-1D0h] BYREF
  __int64 v149; // [rsp+C8h] [rbp-1C8h]
  _BYTE v150[48]; // [rsp+D0h] [rbp-1C0h] BYREF
  unsigned int v151; // [rsp+100h] [rbp-190h] BYREF
  __int64 *v152; // [rsp+108h] [rbp-188h] BYREF
  __int64 v153; // [rsp+110h] [rbp-180h]
  _BYTE v154[72]; // [rsp+118h] [rbp-178h] BYREF
  __int64 v155[2]; // [rsp+160h] [rbp-130h] BYREF
  void (__fastcall *v156)(__int64 *, __int64 *, __int64); // [rsp+170h] [rbp-120h]
  __int64 v157; // [rsp+178h] [rbp-118h]
  __int64 v158; // [rsp+180h] [rbp-110h]
  __int64 v159; // [rsp+188h] [rbp-108h]
  __int64 v160; // [rsp+190h] [rbp-100h]
  __int64 v161; // [rsp+198h] [rbp-F8h]
  __int64 v162; // [rsp+1A0h] [rbp-F0h]
  __int64 v163; // [rsp+1A8h] [rbp-E8h]
  __int64 v164; // [rsp+1B0h] [rbp-E0h]
  unsigned int v165; // [rsp+1B8h] [rbp-D8h]
  __int64 v166; // [rsp+1C0h] [rbp-D0h]
  __int64 v167; // [rsp+1C8h] [rbp-C8h]
  __int64 v168; // [rsp+1D0h] [rbp-C0h]
  __int64 v169; // [rsp+1D8h] [rbp-B8h]
  __int64 v170; // [rsp+1E0h] [rbp-B0h]
  __int64 v171; // [rsp+1E8h] [rbp-A8h]
  __int64 v172; // [rsp+1F0h] [rbp-A0h]
  __int64 v173; // [rsp+1F8h] [rbp-98h]
  _BYTE *v174; // [rsp+200h] [rbp-90h]
  __int64 v175; // [rsp+208h] [rbp-88h]
  _BYTE v176[48]; // [rsp+210h] [rbp-80h] BYREF
  __int64 v177; // [rsp+240h] [rbp-50h]
  __int64 v178; // [rsp+248h] [rbp-48h]
  __int64 v179; // [rsp+250h] [rbp-40h]
  __int64 v180; // [rsp+258h] [rbp-38h]

  v148 = v150;
  v138 = (__int64 *)a2;
  v136 = a3;
  v141 = 0;
  v142 = 0;
  v143 = 0;
  v144 = 0;
  v149 = 0x600000000LL;
  if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
  {
    sub_B2C6D0(a2, a2, a3, a4);
    v6 = *(_QWORD **)(a2 + 96);
    v7 = &v6[5 * *(_QWORD *)(a2 + 104)];
    if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
    {
      sub_B2C6D0(a2, a2, v40, v41);
      v6 = *(_QWORD **)(a2 + 96);
    }
  }
  else
  {
    v6 = *(_QWORD **)(a2 + 96);
    v7 = &v6[5 * *(_QWORD *)(a2 + 104)];
  }
  for ( ; v7 != v6; LODWORD(v149) = v149 + 1 )
  {
    while ( !(unsigned __int8)sub_3176D40((__int64 *)a1, v6) )
    {
      v6 += 5;
      if ( v7 == v6 )
        goto LABEL_10;
    }
    v10 = (unsigned int)v149;
    v11 = (unsigned int)v149 + 1LL;
    if ( v11 > HIDWORD(v149) )
    {
      sub_C8D5F0((__int64)&v148, v150, v11, 8u, v8, v9);
      v10 = (unsigned int)v149;
    }
    *(_QWORD *)&v148[8 * v10] = v6;
    v6 += 5;
  }
LABEL_10:
  if ( !(_DWORD)v149 )
  {
    LODWORD(v12) = 0;
    goto LABEL_39;
  }
  v12 = v138[2];
  if ( !v12 )
    goto LABEL_38;
  while ( 2 )
  {
    v13 = *(_QWORD *)(v12 + 24);
    v14 = *(_BYTE *)v13;
    if ( *(_BYTE *)v13 <= 0x1Cu || v14 != 85 && v14 != 34 )
      goto LABEL_14;
    v15 = *(__int64 **)(v13 - 32);
    if ( v15 )
    {
      if ( *(_BYTE *)v15 )
      {
        v15 = 0;
      }
      else if ( *(_QWORD *)(v13 + 80) != v15[3] )
      {
        v15 = 0;
      }
    }
    if ( v138 != v15 )
      goto LABEL_14;
    if ( (unsigned __int8)sub_A73ED0((_QWORD *)(v13 + 72), 18) )
      goto LABEL_14;
    if ( (unsigned __int8)sub_B49560(v13, 18) )
      goto LABEL_14;
    if ( !(unsigned __int8)sub_2A64220(*(__int64 **)a1, *(_QWORD *)(v13 + 40)) )
      goto LABEL_14;
    v151 = 0;
    v152 = (__int64 *)v154;
    v153 = 0x400000000LL;
    v16 = &v148[8 * (unsigned int)v149];
    if ( v16 == v148 )
      goto LABEL_14;
    v129 = v12;
    v17 = (unsigned __int64)v148;
    v18 = v13;
    do
    {
      v19 = *(_QWORD *)v17;
      v20 = (_QWORD **)a1;
      v21 = *(_QWORD *)(v18
                      + 32
                      * (*(unsigned int *)(*(_QWORD *)v17 + 32LL) - (unsigned __int64)(*(_DWORD *)(v18 + 4) & 0x7FFFFFF)));
      v25 = (__int64 *)sub_3176FF0((__int64 **)a1, (_BYTE *)v21);
      if ( v25 )
      {
        v26 = (unsigned int)v153;
        v23 = (__int64 *)HIDWORD(v153);
        v22 = (unsigned int)v153 + 1LL;
        if ( v22 > HIDWORD(v153) )
        {
          v21 = (unsigned __int64)v154;
          v20 = &v152;
          v122 = v25;
          sub_C8D5F0((__int64)&v152, v154, v22, 0x10u, v24, (__int64)v25);
          v26 = (unsigned int)v153;
          v25 = v122;
        }
        v27 = &v152[2 * v26];
        *v27 = v19;
        v27[1] = (__int64)v25;
        LODWORD(v153) = v153 + 1;
      }
      v17 += 8LL;
    }
    while ( v16 != (_BYTE *)v17 );
    v28 = v18;
    v12 = v129;
    if ( !(_DWORD)v153 )
      goto LABEL_59;
    v29 = v144;
    if ( !v144 )
      goto LABEL_62;
    v130 = v142;
    v155[0] = sub_3177C10(v152, &v152[2 * (unsigned int)v153]);
    v145[0] = 0x9DDFEA08EB382D69LL
            * (((0x9DDFEA08EB382D69LL
               * ((0x9DDFEA08EB382D69LL * ((_QWORD)sub_C64CA0 + 8 * v151))
                ^ ((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 + 8 * v151)) >> 47))) >> 47)
             ^ (0x9DDFEA08EB382D69LL
              * ((0x9DDFEA08EB382D69LL * ((_QWORD)sub_C64CA0 + 8 * v151))
               ^ ((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 + 8 * v151)) >> 47))));
    v30 = sub_C41E80(v145, v155);
    v21 = (unsigned int)v153;
    v23 = v152;
    v24 = 1;
    v31 = v30;
    v32 = v29 - 1;
    v22 = (v29 - 1) & v31;
    v25 = &v152[2 * (unsigned int)v153];
    while ( 1 )
    {
      v33 = (unsigned int *)(v130 + 96LL * (unsigned int)v22);
      v20 = (_QWORD **)*v33;
      if ( v151 != (_DWORD)v20 || (unsigned int)v153 != (unsigned __int64)v33[4] )
        goto LABEL_36;
      v43 = (_QWORD *)*((_QWORD *)v33 + 1);
      if ( v152 == v25 )
        break;
      v42 = v152;
      while ( *v42 == *v43 && v42[1] == v43[1] )
      {
        v42 += 2;
        v43 += 2;
        if ( v25 == v42 )
          goto LABEL_54;
      }
LABEL_36:
      if ( (_DWORD)v20 == -1 )
      {
        v20 = (_QWORD **)v33[4];
        if ( !(_DWORD)v20 )
          goto LABEL_62;
      }
      v34 = v24 + v22;
      v24 = (unsigned int)(v24 + 1);
      v22 = v32 & v34;
    }
LABEL_54:
    if ( v33 != (unsigned int *)(v142 + 96LL * v144) )
    {
      v44 = sub_B43CB0(v28);
      if ( v138 != (__int64 *)v44 )
      {
        v47 = *(_QWORD *)a4 + 176LL * v33[22];
        v48 = *(unsigned int *)(v47 + 120);
        if ( v48 + 1 > (unsigned __int64)*(unsigned int *)(v47 + 124) )
        {
          sub_C8D5F0(v47 + 112, (const void *)(v47 + 128), v48 + 1, 8u, v45, v46);
          v48 = *(unsigned int *)(v47 + 120);
        }
        *(_QWORD *)(*(_QWORD *)(v47 + 112) + 8 * v48) = v28;
        ++*(_DWORD *)(v47 + 120);
      }
      goto LABEL_59;
    }
LABEL_62:
    v49 = (__int64)v138;
    if ( !*(_QWORD *)(a1 + 104) )
      sub_4263D6(v20, v21, v22);
    v50 = v138;
    v51 = (*(__int64 (__fastcall **)(__int64, __int64 *, unsigned __int64, __int64 *, __int64, __int64 *))(a1 + 112))(
            a1 + 88,
            v138,
            v22,
            v23,
            v24,
            v25);
    v146 = 0;
    v123 = v51;
    v131 = *(_QWORD *)a1;
    v124 = *(_QWORD *)(a1 + 8) + 312LL;
    v53 = *(void (__fastcall **)(__int64 *, __int64, __int64))(a1 + 40);
    if ( v53 )
    {
      v50 = (__int64 *)(a1 + 24);
      v53(v145, a1 + 24, 2);
      v54 = *(_QWORD *)(a1 + 48);
      v52 = v145;
      v156 = 0;
      v147 = v54;
      v146 = *(void (__fastcall **)(__int64 *, __int64, __int64))(a1 + 40);
      v53 = v146;
      if ( v146 )
      {
        v50 = v145;
        v146(v155, (__int64)v145, 2);
        v157 = v147;
        v53 = v146;
        v156 = (void (__fastcall *)(__int64 *, __int64 *, __int64))v146;
      }
    }
    else
    {
      v156 = 0;
    }
    v158 = v49;
    v162 = 0;
    v159 = v124;
    v163 = 0;
    v160 = v123;
    v164 = 0;
    v161 = v131;
    v174 = v176;
    v165 = 0;
    v166 = 0;
    v167 = 0;
    v168 = 0;
    v169 = 0;
    v170 = 0;
    v171 = 0;
    v172 = 0;
    v173 = 0;
    v175 = 0x600000000LL;
    v177 = 0;
    v178 = -1;
    v179 = 0;
    v180 = 0;
    if ( v53 )
    {
      v50 = v145;
      ((void (__fastcall *)(__int64 *, __int64 *, __int64, __int64, __int64 *))v53)(v145, v145, 3, 0x600000000LL, v52);
    }
    v125 = &v152[2 * (unsigned int)v153];
    if ( v125 == v152 )
    {
      v113 = sub_317ACD0((__int64)v155);
      v64 = (unsigned int)(v114 - 1);
      if ( !(_DWORD)v64 )
      {
        v132 = 0;
        goto LABEL_79;
      }
      v139 = v113;
      v69 = v113;
      v132 = 0;
      v127 = v136 - v113;
      if ( (_BYTE)qword_5034E48 )
        goto LABEL_89;
LABEL_81:
      if ( (unsigned int)qword_5034828 * v136 / 0x64 <= v69 )
      {
        v70 = sub_3175260((__int64)v155, (__int64)v50, v64);
        if ( v71 )
          v70 = v120;
        v120 = v70;
        v118 = v70;
        LODWORD(v145[0]) = v70;
        if ( (unsigned int)v70 >= dword_5034748 * v136 / 0x64
          && (*(_DWORD *)sub_9E0980(a1 + 760, (__int64 *)&v138) + v127) / v136 <= dword_5034908 )
        {
          v72 = (__int64 *)&v139;
          if ( v118 > v69 )
            v72 = v145;
          v132 += *(_DWORD *)v72;
          goto LABEL_89;
        }
      }
    }
    else
    {
      v132 = 0;
      v55 = 0;
      v117 = v28;
      v56 = 0;
      v115 = v12;
      v57 = v152;
      do
      {
        v58 = sub_317AE00((__int64)v155, *v57, v57[1]);
        if ( v59 == 1 )
          v56 = 1;
        v60 = __OFADD__(v58, v55);
        v55 += v58;
        if ( v60 )
        {
          v55 = 0x8000000000000000LL;
          if ( v58 > 0 )
            v55 = 0x7FFFFFFFFFFFFFFFLL;
        }
        v61 = (unsigned __int8 *)v57[1];
        v62 = *v57;
        v57 += 2;
        v132 += sub_3176AB0(a1, v62, v61);
      }
      while ( v125 != v57 );
      v126 = v56;
      v12 = v115;
      v28 = v117;
      v63 = sub_317ACD0((__int64)v155);
      v67 = v63;
      if ( (_DWORD)v64 != 1 )
      {
        v68 = v63 + v55;
        if ( __OFADD__(v67, v55) )
        {
          v68 = 0x8000000000000000LL;
          if ( v67 > 0 )
            v68 = 0x7FFFFFFFFFFFFFFFLL;
        }
        if ( v126 )
          v68 = (unsigned __int64)v121;
        v121 = (__int64 *)v68;
      }
LABEL_79:
      v50 = v121;
      v139 = (unsigned int)v121;
      v69 = (unsigned int)v121;
      v127 = v136 - (_DWORD)v121;
      if ( !(_BYTE)qword_5034E48 && (unsigned int)qword_5034668 * v136 / 0x64 >= v132 )
        goto LABEL_81;
LABEL_89:
      v73 = *(unsigned int *)(a4 + 8);
      v74 = v73;
      if ( *(_DWORD *)(a4 + 12) <= (unsigned int)v73 )
      {
        v105 = sub_C8D7D0(a4, a4 + 16, 0, 0xB0u, (unsigned __int64 *)v145, v66);
        v108 = v105 + 176LL * *(unsigned int *)(a4 + 8);
        if ( v108 )
        {
          v109 = (__int64)v138;
          *(_QWORD *)(v108 + 8) = 0;
          *(_QWORD *)v108 = v109;
          LODWORD(v109) = v151;
          *(_QWORD *)(v108 + 32) = 0x400000000LL;
          *(_DWORD *)(v108 + 16) = v109;
          v85 = (_DWORD)v153 == 0;
          *(_QWORD *)(v108 + 24) = v108 + 40;
          if ( !v85 )
          {
            v116 = v108;
            sub_3174B60(v108 + 24, (__int64)&v152, v108 + 40, 0x400000000LL, v106, v107);
            v108 = v116;
          }
          v103 = v108 + 128;
          *(_QWORD *)(v108 + 112) = v108 + 128;
          *(_DWORD *)(v108 + 104) = v132;
          *(_DWORD *)(v108 + 108) = v127;
          v104 = 0x600000000LL;
          *(_QWORD *)(v108 + 120) = 0x600000000LL;
        }
        sub_3178480((__int64 *)a4, v105, v103, v104, v106, v107);
        v110 = v145[0];
        if ( a4 + 16 != *(_QWORD *)a4 )
        {
          v134 = v145[0];
          _libc_free(*(_QWORD *)a4);
          v110 = v134;
        }
        *(_DWORD *)(a4 + 12) = v110;
        v111 = *(_DWORD *)(a4 + 8);
        *(_QWORD *)a4 = v105;
        v112 = (unsigned int)(v111 + 1);
        *(_DWORD *)(a4 + 8) = v112;
        v80 = v105 + 176 * v112 - 176;
      }
      else
      {
        v75 = *(_QWORD *)a4;
        v76 = *(_QWORD *)a4 + 176 * v73;
        if ( v76 )
        {
          v77 = (__int64)v138;
          *(_QWORD *)(v76 + 8) = 0;
          *(_QWORD *)v76 = v77;
          *(_DWORD *)(v76 + 16) = v151;
          v78 = (unsigned int)v153;
          *(_QWORD *)(v76 + 24) = v76 + 40;
          *(_QWORD *)(v76 + 32) = 0x400000000LL;
          if ( (_DWORD)v78 )
            sub_3174B60(v76 + 24, (__int64)&v152, v75, v78, v65, v66);
          *(_DWORD *)(v76 + 104) = v132;
          *(_DWORD *)(v76 + 108) = v127;
          *(_QWORD *)(v76 + 112) = v76 + 128;
          *(_QWORD *)(v76 + 120) = 0x600000000LL;
          v74 = *(_DWORD *)(a4 + 8);
          v75 = *(_QWORD *)a4;
        }
        v79 = (unsigned int)(v74 + 1);
        *(_DWORD *)(a4 + 8) = v79;
        v80 = v75 + 176 * v79 - 176;
      }
      v81 = sub_B43CB0(v28);
      if ( v138 != (__int64 *)v81 )
      {
        v84 = *(unsigned int *)(v80 + 120);
        if ( v84 + 1 > (unsigned __int64)*(unsigned int *)(v80 + 124) )
        {
          sub_C8D5F0(v80 + 112, (const void *)(v80 + 128), v84 + 1, 8u, v82, v83);
          v84 = *(unsigned int *)(v80 + 120);
        }
        *(_QWORD *)(*(_QWORD *)(v80 + 112) + 8 * v84) = v28;
        ++*(_DWORD *)(v80 + 120);
      }
      v128 = *(_DWORD *)(a4 + 8);
      v133 = v128 - 1;
      v85 = (unsigned __int8)sub_3178240((__int64)&v141, &v151, &v140) == 0;
      v87 = (_DWORD *)v140;
      if ( v85 )
      {
        v88 = v144;
        v145[0] = v140;
        ++v141;
        v89 = v143 + 1;
        if ( 4 * ((int)v143 + 1) >= 3 * v144 )
        {
          v88 = 2 * v144;
        }
        else
        {
          v90 = v144 - HIDWORD(v143) - v89;
          v91 = v144 >> 3;
          if ( (unsigned int)v90 > (unsigned int)v91 )
          {
LABEL_102:
            LODWORD(v143) = v89;
            if ( *v87 != -1 || v87[4] )
              --HIDWORD(v143);
            v92 = v151;
            v119 = v87;
            *v87 = v151;
            sub_3174B60((__int64)(v87 + 2), (__int64)&v152, v92, v90, v86, v91);
            v87 = v119;
            v119[22] = 0;
            goto LABEL_106;
          }
        }
        sub_3178B50((__int64)&v141, v88);
        sub_3178240((__int64)&v141, &v151, v145);
        v89 = v143 + 1;
        v87 = (_DWORD *)v145[0];
        goto LABEL_102;
      }
LABEL_106:
      v87[22] = v133;
      v93 = *(_DWORD *)(a5 + 24);
      if ( v93 )
      {
        v94 = *(_QWORD *)(a5 + 8);
        v95 = v93 - 1;
        v96 = 1;
        v97 = 0;
        v98 = v95 & (((unsigned int)v138 >> 9) ^ ((unsigned int)v138 >> 4));
        v99 = v94 + 16LL * v98;
        v100 = *(__int64 **)v99;
        if ( v138 == *(__int64 **)v99 )
        {
LABEL_108:
          *(_DWORD *)(v99 + 12) = v128;
          goto LABEL_109;
        }
        while ( v100 != (__int64 *)-4096LL )
        {
          if ( !v97 && v100 == (__int64 *)-8192LL )
            v97 = v99;
          v98 = v95 & (v96 + v98);
          v99 = v94 + 16LL * v98;
          v100 = *(__int64 **)v99;
          if ( v138 == *(__int64 **)v99 )
            goto LABEL_108;
          ++v96;
        }
        if ( v97 )
          v99 = v97;
      }
      else
      {
        v99 = 0;
      }
      v101 = sub_31789B0(a5, &v138, (_QWORD *)v99);
      v102 = (__int64)v138;
      *((_DWORD *)v101 + 2) = v133;
      *v101 = v102;
      *((_DWORD *)v101 + 3) = v128;
    }
LABEL_109:
    if ( v174 != v176 )
      _libc_free((unsigned __int64)v174);
    sub_C7D6A0(v171, 8LL * (unsigned int)v173, 8);
    sub_C7D6A0(v167, 8LL * (unsigned int)v169, 8);
    sub_C7D6A0(v163, 16LL * v165, 8);
    if ( v156 )
      v156(v155, v155, 3);
LABEL_59:
    if ( v152 != (__int64 *)v154 )
      _libc_free((unsigned __int64)v152);
LABEL_14:
    v12 = *(_QWORD *)(v12 + 8);
    if ( v12 )
      continue;
    break;
  }
LABEL_38:
  LOBYTE(v12) = (_DWORD)v143 != 0;
LABEL_39:
  if ( v148 != v150 )
    _libc_free((unsigned __int64)v148);
  v35 = v144;
  if ( v144 )
  {
    v36 = v142;
    v37 = v142 + 96LL * v144;
    do
    {
      v38 = *(_QWORD *)(v36 + 8);
      if ( v38 != v36 + 24 )
        _libc_free(v38);
      v36 += 96;
    }
    while ( v37 != v36 );
    v35 = v144;
  }
  sub_C7D6A0(v142, 96 * v35, 8);
  return (unsigned int)v12;
}
