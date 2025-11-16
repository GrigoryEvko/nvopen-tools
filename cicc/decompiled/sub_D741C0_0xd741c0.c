// Function: sub_D741C0
// Address: 0xd741c0
//
__int64 __fastcall sub_D741C0(__int64 a1, __int64 *a2, char a3)
{
  __int64 v3; // rbx
  unsigned int v4; // eax
  __int64 v5; // rdx
  _QWORD *v6; // r13
  _QWORD *v7; // r12
  __int64 v8; // rax
  __int64 *v9; // r12
  char v10; // r14
  __int64 v11; // rdi
  __int64 v12; // r8
  __int64 v13; // r9
  _QWORD *v14; // r15
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rdx
  _QWORD *v18; // r12
  unsigned __int64 *v19; // r13
  unsigned int v20; // r15d
  __int64 i; // r13
  __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // r9
  _QWORD *v25; // r15
  _QWORD *v26; // r12
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // r15
  __int64 v30; // rcx
  _QWORD *v31; // r14
  _QWORD *v32; // r15
  unsigned __int64 v33; // r13
  __int64 v34; // r12
  int v35; // eax
  unsigned __int64 *v36; // r12
  unsigned __int64 v37; // rax
  char *v38; // r12
  char *v39; // r12
  char *v40; // r13
  char *v41; // r14
  __int64 v42; // rdx
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 v54; // r9
  char *v55; // r12
  char *v56; // r13
  __int64 v57; // rbx
  _QWORD *v58; // r12
  __int64 v59; // rax
  _QWORD *v60; // rbx
  __int64 result; // rax
  _QWORD *v62; // r12
  __int64 v63; // r13
  __int64 v64; // r14
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // r13
  __int64 k; // r14
  __int64 v69; // rax
  __int64 v70; // r14
  __int64 *v71; // r13
  char v72; // r15
  __int64 v73; // rax
  __int64 v74; // rdi
  __int64 v75; // rax
  __int64 v76; // rdx
  __int64 v77; // rcx
  unsigned int v78; // eax
  __int64 v79; // rax
  __int64 *v80; // rdx
  __int64 v81; // rcx
  __int64 v82; // r8
  __int64 v83; // r9
  __int64 v84; // r13
  __int64 j; // r14
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 v88; // rdi
  __int64 *v89; // rbx
  __int64 v90; // r14
  __int64 v91; // rax
  __int64 v92; // r12
  __int64 v93; // rdx
  __int64 v94; // r9
  __int64 v95; // rsi
  __int64 v96; // rcx
  __int64 v97; // r8
  __int64 v98; // rax
  __int64 v99; // rcx
  __int64 v100; // r8
  __int64 v101; // r9
  __int64 v102; // r14
  __int64 v103; // rdx
  __int64 v104; // r15
  __int64 v105; // r13
  unsigned __int64 v106; // rax
  __int64 v107; // rsi
  _QWORD *v108; // r15
  _QWORD *v109; // r13
  __int64 v110; // rax
  _QWORD *v111; // r12
  _QWORD *v112; // r13
  __int64 v113; // rdx
  _QWORD *v114; // rsi
  _QWORD *v115; // r14
  _QWORD *v116; // rbx
  __int64 v117; // rax
  __int64 v118; // r8
  __int64 v119; // r9
  unsigned __int64 v120; // rdx
  unsigned __int64 v121; // rsi
  int v122; // eax
  char *v123; // rcx
  char *v124; // r8
  unsigned __int64 *v125; // rdi
  _QWORD *v126; // rax
  __int64 v127; // rdx
  _QWORD *v128; // rcx
  signed __int64 v129; // rdx
  _QWORD *v130; // rdx
  char *v131; // [rsp+8h] [rbp-498h]
  __int64 v134; // [rsp+50h] [rbp-450h]
  __int64 *v135; // [rsp+58h] [rbp-448h]
  unsigned int v137; // [rsp+70h] [rbp-430h]
  __int64 *v138; // [rsp+70h] [rbp-430h]
  char *v139; // [rsp+70h] [rbp-430h]
  unsigned __int64 v140; // [rsp+78h] [rbp-428h]
  unsigned int v141; // [rsp+78h] [rbp-428h]
  __int64 v142; // [rsp+78h] [rbp-428h]
  __int64 *v143; // [rsp+78h] [rbp-428h]
  __int64 v144; // [rsp+78h] [rbp-428h]
  __int64 *v145; // [rsp+78h] [rbp-428h]
  char *v146; // [rsp+78h] [rbp-428h]
  unsigned __int64 v147[4]; // [rsp+80h] [rbp-420h] BYREF
  __int64 v148; // [rsp+A0h] [rbp-400h] BYREF
  _QWORD *v149; // [rsp+A8h] [rbp-3F8h]
  __int64 v150; // [rsp+B0h] [rbp-3F0h]
  unsigned int v151; // [rsp+B8h] [rbp-3E8h]
  _QWORD v152[2]; // [rsp+C0h] [rbp-3E0h] BYREF
  char v153; // [rsp+D0h] [rbp-3D0h]
  __int64 *v154; // [rsp+E0h] [rbp-3C0h]
  __int64 v155; // [rsp+F0h] [rbp-3B0h] BYREF
  char *v156; // [rsp+F8h] [rbp-3A8h]
  __int64 v157; // [rsp+100h] [rbp-3A0h]
  int v158; // [rsp+108h] [rbp-398h]
  char v159; // [rsp+10Ch] [rbp-394h]
  char v160; // [rsp+110h] [rbp-390h] BYREF
  char *v161; // [rsp+120h] [rbp-380h] BYREF
  __int64 v162; // [rsp+128h] [rbp-378h]
  _BYTE v163[96]; // [rsp+130h] [rbp-370h] BYREF
  _BYTE *v164; // [rsp+190h] [rbp-310h] BYREF
  __int64 v165; // [rsp+198h] [rbp-308h]
  _BYTE v166[192]; // [rsp+1A0h] [rbp-300h] BYREF
  _BYTE *v167; // [rsp+260h] [rbp-240h] BYREF
  __int64 v168; // [rsp+268h] [rbp-238h]
  _BYTE v169[192]; // [rsp+270h] [rbp-230h] BYREF
  __int64 v170; // [rsp+330h] [rbp-170h] BYREF
  __int64 v171; // [rsp+338h] [rbp-168h] BYREF
  _QWORD *v172; // [rsp+340h] [rbp-160h]
  __int64 *v173; // [rsp+348h] [rbp-158h]
  __int64 *v174; // [rsp+350h] [rbp-150h]
  __int64 v175; // [rsp+358h] [rbp-148h]
  __int64 *v176; // [rsp+360h] [rbp-140h] BYREF
  __int64 v177; // [rsp+368h] [rbp-138h]
  __int64 v178; // [rsp+370h] [rbp-130h] BYREF
  int v179; // [rsp+378h] [rbp-128h]
  char v180; // [rsp+37Ch] [rbp-124h]
  char v181; // [rsp+380h] [rbp-120h] BYREF

  v3 = a1;
  ++*(_QWORD *)(a1 + 408);
  if ( !*(_BYTE *)(a1 + 436) )
  {
    v4 = 4 * (*(_DWORD *)(a1 + 428) - *(_DWORD *)(a1 + 432));
    v5 = *(unsigned int *)(a1 + 424);
    if ( v4 < 0x20 )
      v4 = 32;
    if ( (unsigned int)v5 > v4 )
    {
      sub_C8C990(a1 + 408, (__int64)a2);
      goto LABEL_7;
    }
    memset(*(void **)(a1 + 416), -1, 8 * v5);
  }
  *(_QWORD *)(a1 + 428) = 0;
LABEL_7:
  v6 = *(_QWORD **)(a1 + 8);
  v7 = &v6[3 * *(unsigned int *)(a1 + 16)];
  while ( v6 != v7 )
  {
    while ( 1 )
    {
      v8 = *(v7 - 1);
      v7 -= 3;
      if ( v8 == -4096 || v8 == 0 || v8 == -8192 )
        break;
      sub_BD60C0(v7);
      if ( v6 == v7 )
        goto LABEL_12;
    }
  }
LABEL_12:
  *(_DWORD *)(a1 + 16) = 0;
  v9 = (__int64 *)sub_D735C0((__int64 *)a1, (__int64)a2);
  if ( v9[8] != a2[8] )
    goto LABEL_13;
  if ( *(_BYTE *)v9 != 28 )
  {
LABEL_166:
    v10 = 1;
    v176 = a2;
    sub_BD79D0(v9, a2, (unsigned __int8 (__fastcall *)(__int64, __int64 *))sub_D67790, (__int64)&v176);
    goto LABEL_14;
  }
  v126 = *(_QWORD **)(a1 + 8);
  v127 = 3LL * *(unsigned int *)(a1 + 16);
  v128 = &v126[v127];
  v129 = 0xAAAAAAAAAAAAAAABLL * ((v127 * 8) >> 3);
  if ( v129 >> 2 )
  {
    v130 = &v126[12 * (v129 >> 2)];
    while ( v9 != (__int64 *)v126[2] )
    {
      if ( v9 == (__int64 *)v126[5] )
      {
        v126 += 3;
        goto LABEL_165;
      }
      if ( v9 == (__int64 *)v126[8] )
      {
        v126 += 6;
        goto LABEL_165;
      }
      if ( v9 == (__int64 *)v126[11] )
      {
        v126 += 9;
        goto LABEL_165;
      }
      v126 += 12;
      if ( v130 == v126 )
      {
        v129 = 0xAAAAAAAAAAAAAAABLL * (v128 - v126);
        goto LABEL_177;
      }
    }
    goto LABEL_165;
  }
LABEL_177:
  if ( v129 == 2 )
    goto LABEL_187;
  if ( v129 == 3 )
  {
    if ( v9 == (__int64 *)v126[2] )
      goto LABEL_165;
    v126 += 3;
LABEL_187:
    if ( v9 == (__int64 *)v126[2] )
      goto LABEL_165;
    v126 += 3;
    goto LABEL_180;
  }
  if ( v129 != 1 )
    goto LABEL_166;
LABEL_180:
  if ( v9 != (__int64 *)v126[2] )
    goto LABEL_166;
LABEL_165:
  if ( v128 == v126 )
    goto LABEL_166;
LABEL_13:
  v10 = 0;
LABEL_14:
  v11 = (__int64)(a2 - 8);
  if ( *(_BYTE *)a2 == 26 )
    v11 = (__int64)(a2 - 4);
  sub_AC2B30(v11, (__int64)v9);
  v14 = *(_QWORD **)(v3 + 8);
  v15 = 0;
  v16 = 0xAAAAAAAAAAAAAAABLL;
  v17 = 3LL * *(unsigned int *)(v3 + 16);
  v164 = v166;
  v18 = &v14[v17];
  v165 = 0x800000000LL;
  v140 = 0xAAAAAAAAAAAAAAABLL * ((v17 * 8) >> 3);
  v19 = (unsigned __int64 *)v166;
  if ( (unsigned __int64)v17 > 24 )
  {
    sub_D6B130((__int64)&v164, v140, v17 * 8, 0xAAAAAAAAAAAAAAABLL, v12, v13);
    v15 = (unsigned int)v165;
    v16 = 3LL * (unsigned int)v165;
    v19 = (unsigned __int64 *)&v164[24 * (unsigned int)v165];
  }
  if ( v14 != v18 )
  {
    do
    {
      if ( v19 )
        sub_D68CD0(v19, 2u, v14);
      v14 += 3;
      v19 += 3;
    }
    while ( v18 != v14 );
    v15 = (unsigned int)v165;
  }
  LODWORD(v171) = 0;
  v167 = v169;
  v168 = 0x800000000LL;
  LODWORD(v165) = v15 + v140;
  v172 = 0;
  v173 = &v171;
  v174 = &v171;
  v175 = 0;
  if ( v10 )
  {
    if ( !((_DWORD)v15 + (_DWORD)v140) )
      goto LABEL_58;
    v141 = 0;
    v20 = *(_DWORD *)(v3 + 16);
    v137 = v20;
LABEL_26:
    for ( i = v20; ; i = *(unsigned int *)(v3 + 16) )
    {
      v15 = (__int64)&v164;
      sub_D738B0(v3, (__int64)&v164);
      v25 = v164;
      v26 = &v164[24 * (unsigned int)v165];
      while ( v25 != v26 )
      {
        while ( 1 )
        {
          v27 = *(v26 - 1);
          v26 -= 3;
          LOBYTE(v22) = v27 != 0;
          if ( v27 == 0 || v27 == -4096 || v27 == -8192 )
            break;
          sub_BD60C0(v26);
          if ( v25 == v26 )
            goto LABEL_32;
        }
      }
LABEL_32:
      v28 = *(unsigned int *)(v3 + 16);
      v29 = *(_QWORD *)(v3 + 8);
      LODWORD(v165) = 0;
      v30 = 24 * i;
      v28 *= 24;
      v31 = (_QWORD *)(v29 + v28);
      v32 = (_QWORD *)(24 * i + v29);
      v33 = 0xAAAAAAAAAAAAAAABLL * ((v28 - 24 * i) >> 3);
      if ( v33 > HIDWORD(v165) )
      {
        v15 = v33;
        sub_D6B130((__int64)&v164, v33, v22, v30, v23, v24);
        v35 = v165;
        v34 = 24LL * (unsigned int)v165;
      }
      else
      {
        v34 = 0;
        v35 = 0;
      }
      v36 = (unsigned __int64 *)&v164[v34];
      if ( v31 != v32 )
      {
        do
        {
          if ( v36 )
          {
            *v36 = 4;
            v36[1] = 0;
            v37 = v32[2];
            v36[2] = v37;
            if ( v37 != -4096 && v37 != 0 && v37 != -8192 )
            {
              v15 = *v32 & 0xFFFFFFFFFFFFFFF8LL;
              sub_BD6050(v36, v15);
            }
          }
          v32 += 3;
          v36 += 3;
        }
        while ( v31 != v32 );
        v35 = v165;
      }
      LODWORD(v165) = v35 + v33;
      if ( !(v35 + (_DWORD)v33) )
        break;
    }
    goto LABEL_57;
  }
  v155 = 0;
  v156 = &v160;
  v159 = 1;
  v157 = 2;
  v80 = (__int64 *)a2[8];
  v158 = 0;
  sub_D695C0((__int64)&v176, (__int64)&v155, v80, v16, v12, v13);
  v84 = *(_QWORD *)(v3 + 8);
  for ( j = v84 + 24LL * *(unsigned int *)(v3 + 16); j != v84; v84 += 24 )
  {
    v86 = *(_QWORD *)(v84 + 16);
    if ( v86 )
      sub_D695C0((__int64)&v176, (__int64)&v155, *(__int64 **)(v86 + 64), v81, v82, v83);
  }
  v40 = (char *)&v148;
  v87 = *(_QWORD *)(*(_QWORD *)v3 + 8LL);
  v153 = 0;
  v152[1] = 0;
  v152[0] = v87;
  v176 = &v178;
  v177 = 0x2000000000LL;
  v154 = &v155;
  sub_D6A180((__int64)v152, (__int64)&v176);
  v161 = v163;
  v162 = 0x400000000LL;
  v135 = &v176[(unsigned int)v177];
  if ( v176 == v135 )
  {
LABEL_175:
    v137 = *(_DWORD *)(v3 + 16);
    goto LABEL_47;
  }
  v88 = v3 + 504;
  v138 = (__int64 *)v3;
  v40 = (char *)&v148;
  v89 = v176;
  v134 = v88;
  do
  {
    v90 = *v138;
    v142 = *v89;
    v91 = sub_D68B40(*v138, *v89);
    v92 = v91;
    if ( v91 )
    {
      sub_D68D20((__int64)v40, 2u, v91);
      if ( !v175 )
      {
        v95 = (unsigned int)v168;
        v96 = (__int64)v167;
        v97 = (__int64)&v167[24 * (unsigned int)v168];
        if ( v167 == (_BYTE *)v97 )
        {
          if ( (unsigned int)v168 <= 7uLL )
          {
LABEL_156:
            sub_D6B260((__int64)&v167, v40, v93, (__int64)v167, v97, v94);
            goto LABEL_117;
          }
        }
        else
        {
          v93 = v150;
          v98 = (__int64)v167;
          while ( *(_QWORD *)(v98 + 16) != v150 )
          {
            v98 += 24;
            if ( v97 == v98 )
              goto LABEL_155;
          }
          if ( v98 != v97 )
            goto LABEL_117;
LABEL_155:
          if ( (unsigned int)v168 <= 7uLL )
            goto LABEL_156;
          v144 = v92;
          v111 = &v167[24 * (unsigned int)v168];
          v131 = v40;
          v112 = v167;
          do
          {
            v114 = sub_D6C6F0(&v170, &v171, (__int64)v112);
            if ( v113 )
              sub_D681D0((__int64)&v170, (__int64)v114, v113, v112);
            v112 += 3;
          }
          while ( v111 != v112 );
          v92 = v144;
          v40 = v131;
          v96 = (__int64)v167;
          v95 = (unsigned int)v168;
        }
        if ( v96 != v96 + 24 * v95 )
        {
          v145 = v89;
          v115 = (_QWORD *)(v96 + 24 * v95);
          v116 = (_QWORD *)v96;
          do
          {
            v117 = *(v115 - 1);
            v115 -= 3;
            if ( v117 != -4096 && v117 != 0 && v117 != -8192 )
              sub_BD60C0(v115);
          }
          while ( v116 != v115 );
          v89 = v145;
        }
        LODWORD(v168) = 0;
      }
      sub_D6B8A0((__int64)&v170, v40);
LABEL_117:
      sub_D68D70(v40);
      goto LABEL_118;
    }
    v92 = sub_10420D0(v90, v142);
    sub_D68D20((__int64)v40, 0, v92);
    v120 = (unsigned int)v162;
    v121 = (unsigned int)v162 + 1LL;
    v122 = v162;
    if ( v121 > HIDWORD(v162) )
    {
      if ( v161 > v40 || (v120 = (unsigned __int64)&v161[24 * (unsigned int)v162], (unsigned __int64)v40 >= v120) )
      {
        sub_D6B530((__int64)&v161, v121, v120, HIDWORD(v162), v118, v119);
        v120 = (unsigned int)v162;
        v123 = v161;
        v124 = v40;
        v122 = v162;
      }
      else
      {
        v146 = v161;
        sub_D6B530((__int64)&v161, v121, v120, HIDWORD(v162), v118, v119);
        v123 = v161;
        v120 = (unsigned int)v162;
        v124 = &v161[v40 - v146];
        v122 = v162;
      }
    }
    else
    {
      v123 = v161;
      v124 = v40;
    }
    v125 = (unsigned __int64 *)&v123[24 * v120];
    if ( v125 )
    {
      sub_D68CD0(v125, 0, v124);
      v122 = v162;
    }
    LODWORD(v162) = v122 + 1;
    sub_D68D70(v40);
LABEL_118:
    ++v89;
    sub_D68D20((__int64)v147, 0, v92);
    sub_D6C8F0((__int64)v40, v134, v147, v99, v100, v101);
    sub_D68D70(v147);
  }
  while ( v135 != v89 );
  v38 = v161;
  v3 = (__int64)v138;
  v139 = &v161[24 * (unsigned int)v162];
  if ( v161 == v139 )
    goto LABEL_175;
  v143 = (__int64 *)v40;
  do
  {
    v148 = *(_QWORD *)(*(_QWORD *)(*((_QWORD *)v38 + 2) + 64LL) + 16LL);
    sub_D4B000(v143);
    v102 = v148;
    if ( v148 )
    {
      v103 = *(_QWORD *)(v148 + 24);
LABEL_123:
      v104 = *(_QWORD *)(v103 + 40);
      v148 = 0;
      v149 = 0;
      v150 = 0;
      v151 = 0;
      v105 = *((_QWORD *)v38 + 2);
      v106 = sub_D740C0((__int64 *)v3, v104, (__int64)v143);
      sub_D689D0(v105, v106, v104);
      v107 = v151;
      if ( v151 )
      {
        v108 = v149;
        v109 = &v149[4 * v151];
        do
        {
          if ( *v108 != -8192 && *v108 != -4096 )
          {
            v110 = v108[3];
            if ( v110 != -4096 && v110 != 0 && v110 != -8192 )
              sub_BD60C0(v108 + 1);
          }
          v108 += 4;
        }
        while ( v109 != v108 );
        v107 = v151;
      }
      sub_C7D6A0((__int64)v149, 32 * v107, 8);
      while ( 1 )
      {
        v102 = *(_QWORD *)(v102 + 8);
        if ( !v102 )
          break;
        v103 = *(_QWORD *)(v102 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v103 - 30) <= 0xAu )
          goto LABEL_123;
      }
    }
    v38 += 24;
  }
  while ( v139 != v38 );
  v39 = v161;
  v40 = (char *)v143;
  v41 = &v161[24 * (unsigned int)v162];
  v137 = *(_DWORD *)(v3 + 16);
  if ( v161 != v41 )
  {
    do
    {
      v42 = *((_QWORD *)v39 + 2);
      v39 += 24;
      sub_D68D20((__int64)v143, 2u, v42);
      sub_D6B260(v3 + 8, (char *)v143, v43, v44, v45, v46);
      sub_D68D70(v143);
      sub_D68D20((__int64)v143, 2u, *((_QWORD *)v39 - 1));
      sub_D6B260((__int64)&v164, (char *)v143, v47, v48, v49, v50);
      sub_D68D70(v143);
    }
    while ( v41 != v39 );
  }
LABEL_47:
  sub_D68D20((__int64)v40, 2u, (__int64)a2);
  v15 = (__int64)v40;
  sub_D6B260((__int64)&v164, v40, v51, v52, v53, v54);
  sub_D68D70(v40);
  v55 = v161;
  v56 = &v161[24 * (unsigned int)v162];
  if ( v161 != v56 )
  {
    do
    {
      v56 -= 24;
      sub_D68D70(v56);
    }
    while ( v55 != v56 );
    v56 = v161;
  }
  if ( v56 != v163 )
    _libc_free(v56, v15);
  if ( v176 != &v178 )
    _libc_free(v176, v15);
  if ( !v159 )
    _libc_free(v156, v15);
  v20 = *(_DWORD *)(v3 + 16);
  v141 = v20 - v137;
  if ( (_DWORD)v165 )
    goto LABEL_26;
LABEL_57:
  if ( v141 )
  {
    v15 = *(_QWORD *)(v3 + 8) + 24LL * v137;
    sub_D6FF00(v3, v15, v141);
    if ( a3 )
      goto LABEL_77;
  }
  else
  {
LABEL_58:
    if ( !a3 )
      goto LABEL_59;
LABEL_77:
    v63 = *(_QWORD *)v3;
    v176 = 0;
    v178 = 16;
    v64 = a2[8];
    v179 = 0;
    v177 = (__int64)&v181;
    v180 = 1;
    v65 = *(_QWORD *)(sub_D68C20(v63, v64) + 8);
    if ( !v65 )
      BUG();
    v66 = v65 - 48;
    if ( *(_BYTE *)(v65 - 48) == 27 )
      v66 = *(_QWORD *)(v65 - 112);
    v15 = v64;
    sub_D68C90(v63, v64, v66, (__int64)&v176);
    v67 = *(_QWORD *)(v3 + 8);
    for ( k = v67 + 24LL * *(unsigned int *)(v3 + 16); k != v67; v67 += 24 )
    {
      v69 = *(_QWORD *)(v67 + 16);
      if ( v69 && *(_BYTE *)v69 == 28 )
      {
        v15 = *(_QWORD *)(v69 + 64);
        sub_D68C90(*(_QWORD *)v3, v15, 0, (__int64)&v176);
      }
    }
    if ( v175 )
    {
      v70 = (__int64)v173;
      v71 = &v171;
      v72 = 0;
    }
    else
    {
      v70 = (__int64)v167;
      v72 = a3;
      v71 = (__int64 *)&v167[24 * (unsigned int)v168];
    }
    if ( !v72 )
      goto LABEL_90;
    while ( v71 != (__int64 *)v70 )
    {
      v79 = *(_QWORD *)(v70 + 16);
      if ( v79 && *(_BYTE *)v79 == 28 )
      {
        v74 = *(_QWORD *)v3;
        v75 = *(_QWORD *)(v79 + 64);
        v76 = *(_QWORD *)(*(_QWORD *)v3 + 8LL);
        if ( v75 )
        {
LABEL_94:
          v77 = (unsigned int)(*(_DWORD *)(v75 + 44) + 1);
          v78 = *(_DWORD *)(v75 + 44) + 1;
        }
        else
        {
LABEL_103:
          v77 = 0;
          v78 = 0;
        }
        v15 = 0;
        if ( v78 < *(_DWORD *)(v76 + 32) )
          v15 = *(_QWORD *)(*(_QWORD *)(v76 + 24) + 8 * v77);
        sub_103C0D0(v74, v15, 0, &v176, 1, 1);
        if ( !v72 )
        {
          while ( 1 )
          {
            v70 = sub_220EF30(v70);
LABEL_90:
            if ( v71 == (__int64 *)v70 )
              goto LABEL_169;
            v73 = *(_QWORD *)(v70 + 48);
            if ( v73 && *(_BYTE *)v73 == 28 )
            {
              v74 = *(_QWORD *)v3;
              v75 = *(_QWORD *)(v73 + 64);
              v76 = *(_QWORD *)(*(_QWORD *)v3 + 8LL);
              if ( !v75 )
                goto LABEL_103;
              goto LABEL_94;
            }
          }
        }
      }
      v70 += 24;
    }
LABEL_169:
    if ( !v180 )
      _libc_free(v177, v15);
  }
LABEL_59:
  sub_D68DA0(v172);
  v57 = (__int64)v167;
  v58 = &v167[24 * (unsigned int)v168];
  if ( v167 != (_BYTE *)v58 )
  {
    do
    {
      v59 = *(v58 - 1);
      v58 -= 3;
      if ( v59 != 0 && v59 != -4096 && v59 != -8192 )
        sub_BD60C0(v58);
    }
    while ( (_QWORD *)v57 != v58 );
    v58 = v167;
  }
  if ( v58 != (_QWORD *)v169 )
    _libc_free(v58, v15);
  v60 = v164;
  result = 3LL * (unsigned int)v165;
  v62 = &v164[24 * (unsigned int)v165];
  if ( v164 != (_BYTE *)v62 )
  {
    do
    {
      result = *(v62 - 1);
      v62 -= 3;
      if ( result != 0 && result != -4096 && result != -8192 )
        result = sub_BD60C0(v62);
    }
    while ( v60 != v62 );
    v62 = v164;
  }
  if ( v62 != (_QWORD *)v166 )
    return _libc_free(v62, v15);
  return result;
}
