// Function: sub_1B534C0
// Address: 0x1b534c0
//
__int64 __fastcall sub_1B534C0(_QWORD *a1, __int64 a2, __int64 **a3)
{
  __int64 v3; // r14
  unsigned __int64 v4; // rax
  __int64 *v5; // r9
  __int64 v6; // rax
  __m128i *v7; // rax
  __int64 v8; // r8
  __int64 v9; // r15
  int v10; // ebx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 *v13; // r13
  __int64 v14; // r15
  __int64 v15; // r14
  char *v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // r14
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 *v22; // rbx
  __int64 v23; // r12
  _QWORD *m128i_i64; // rdi
  int v25; // esi
  unsigned int v26; // edx
  __int64 *v27; // rax
  int v28; // edx
  char v29; // dl
  __int64 v30; // rax
  _QWORD *v31; // rbx
  __m128i *v32; // r12
  __m128i *v33; // r13
  __int64 v34; // rax
  __m128i *v35; // rbx
  int v36; // r15d
  __int64 v37; // rax
  __int64 v38; // rdx
  char v39; // si
  __int64 v40; // r12
  __int64 v41; // rax
  char v42; // r11
  __int64 v43; // rcx
  __int64 v44; // rax
  __int64 v45; // rdi
  __int64 v46; // rdi
  __int64 v47; // r13
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rcx
  char v52; // al
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  int v56; // r9d
  unsigned int v57; // r13d
  __int64 v58; // rax
  unsigned __int64 *v59; // rbx
  __int64 v60; // r15
  __int64 v61; // rdi
  unsigned __int64 *v62; // r13
  unsigned __int64 v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // r13
  __int64 v68; // rsi
  __int64 v69; // rax
  __int64 v70; // rcx
  __int64 v71; // rdx
  _QWORD *v72; // r12
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // r12
  unsigned int v76; // eax
  unsigned int v77; // esi
  unsigned int v79; // edx
  unsigned int v80; // edi
  __int64 v81; // rax
  __int64 v82; // rax
  int v83; // r11d
  _QWORD *v84; // rdi
  int v85; // esi
  unsigned int v86; // ecx
  __int64 *v87; // rdx
  _QWORD *v88; // rdi
  int v89; // esi
  unsigned int v90; // ecx
  int v91; // r10d
  char v92; // al
  __int64 v93; // rdx
  __int64 v94; // rcx
  __int64 v95; // r8
  int v96; // r9d
  __int64 v97; // r11
  int v98; // r13d
  __int64 v99; // rdx
  __int64 v100; // rcx
  __int64 v101; // r8
  int v102; // r9d
  int v103; // eax
  _QWORD *v104; // r12
  __int64 v105; // rax
  __int64 *v106; // r10
  __int64 v107; // rax
  __int64 v108; // rsi
  __int64 v109; // rax
  __int64 v110; // rdx
  _QWORD *v111; // rax
  __int64 v112; // r13
  __int64 v113; // rax
  _QWORD **v114; // rbx
  _QWORD **v115; // r13
  _QWORD *v116; // rdi
  int v117; // r10d
  __int64 v118; // [rsp+8h] [rbp-338h]
  __int64 v119; // [rsp+10h] [rbp-330h]
  __int64 v120; // [rsp+18h] [rbp-328h]
  __int64 v124; // [rsp+48h] [rbp-2F8h]
  __int64 v125; // [rsp+58h] [rbp-2E8h]
  __int64 v126; // [rsp+60h] [rbp-2E0h]
  int v127; // [rsp+68h] [rbp-2D8h]
  unsigned __int8 v128; // [rsp+87h] [rbp-2B9h]
  __int64 v129; // [rsp+88h] [rbp-2B8h]
  unsigned __int64 v130; // [rsp+90h] [rbp-2B0h]
  int v131; // [rsp+90h] [rbp-2B0h]
  __int64 v132; // [rsp+98h] [rbp-2A8h]
  __int64 v133; // [rsp+98h] [rbp-2A8h]
  __int64 v134; // [rsp+A0h] [rbp-2A0h]
  __int64 v135; // [rsp+A0h] [rbp-2A0h]
  __int64 v136; // [rsp+A0h] [rbp-2A0h]
  __int64 v137; // [rsp+A8h] [rbp-298h]
  int v138; // [rsp+A8h] [rbp-298h]
  __int64 v139; // [rsp+A8h] [rbp-298h]
  unsigned __int64 *v140; // [rsp+A8h] [rbp-298h]
  __int64 v141; // [rsp+A8h] [rbp-298h]
  _BYTE *v142; // [rsp+B0h] [rbp-290h] BYREF
  __int64 v143; // [rsp+B8h] [rbp-288h]
  _BYTE v144[32]; // [rsp+C0h] [rbp-280h] BYREF
  __m128i v145; // [rsp+E0h] [rbp-260h]
  _BYTE v146[16]; // [rsp+F0h] [rbp-250h] BYREF
  void (__fastcall *v147)(_BYTE *, _BYTE *, __int64); // [rsp+100h] [rbp-240h]
  __m128i v148; // [rsp+110h] [rbp-230h]
  _BYTE v149[16]; // [rsp+120h] [rbp-220h] BYREF
  void (__fastcall *v150)(_BYTE *, _BYTE *, __int64); // [rsp+130h] [rbp-210h]
  __m128i v151; // [rsp+140h] [rbp-200h]
  _BYTE v152[16]; // [rsp+150h] [rbp-1F0h] BYREF
  void (__fastcall *v153)(_BYTE *, _BYTE *, __int64); // [rsp+160h] [rbp-1E0h]
  __m128i v154; // [rsp+170h] [rbp-1D0h]
  _BYTE v155[16]; // [rsp+180h] [rbp-1C0h] BYREF
  void (__fastcall *v156)(_BYTE *, _BYTE *, __int64); // [rsp+190h] [rbp-1B0h]
  unsigned __int8 (__fastcall *v157)(_BYTE *); // [rsp+198h] [rbp-1A8h]
  __m128i v158; // [rsp+1A0h] [rbp-1A0h]
  _BYTE v159[16]; // [rsp+1B0h] [rbp-190h] BYREF
  void (__fastcall *v160)(_BYTE *, _BYTE *, __int64); // [rsp+1C0h] [rbp-180h]
  __m128i v161; // [rsp+1D0h] [rbp-170h] BYREF
  _BYTE v162[16]; // [rsp+1E0h] [rbp-160h] BYREF
  void (__fastcall *v163)(_BYTE *, _BYTE *, __int64); // [rsp+1F0h] [rbp-150h]
  __int64 v164; // [rsp+200h] [rbp-140h] BYREF
  __int64 v165; // [rsp+208h] [rbp-138h]
  __m128i *v166; // [rsp+210h] [rbp-130h] BYREF
  unsigned int v167; // [rsp+218h] [rbp-128h]
  __m128i v168; // [rsp+250h] [rbp-F0h] BYREF
  char v169; // [rsp+260h] [rbp-E0h] BYREF
  char v170; // [rsp+261h] [rbp-DFh]
  void (__fastcall *v171)(char *, char *, __int64); // [rsp+270h] [rbp-D0h]
  unsigned __int8 (__fastcall *v172)(char *, __int64); // [rsp+278h] [rbp-C8h]
  __m128i v173; // [rsp+280h] [rbp-C0h]
  _BYTE v174[16]; // [rsp+290h] [rbp-B0h] BYREF
  void (__fastcall *v175)(_BYTE *, _BYTE *, __int64); // [rsp+2A0h] [rbp-A0h]
  __m128i v176; // [rsp+2B0h] [rbp-90h] BYREF
  _QWORD *v177; // [rsp+2C0h] [rbp-80h] BYREF
  __int64 v178; // [rsp+2C8h] [rbp-78h]
  void (__fastcall *v179)(_QWORD **, _QWORD **, __int64); // [rsp+2D0h] [rbp-70h]
  int v180; // [rsp+2D8h] [rbp-68h]
  __m128i v181; // [rsp+2E0h] [rbp-60h]
  _BYTE v182[16]; // [rsp+2F0h] [rbp-50h] BYREF
  void (__fastcall *v183)(_BYTE *, _BYTE *, __int64); // [rsp+300h] [rbp-40h]

  v120 = *(a1 - 9);
  if ( *(_BYTE *)(v120 + 16) == 76 )
    return 0;
  v3 = a1[5];
  v4 = sub_157EBA0(a2);
  v164 = 0;
  v118 = sub_15F4DF0(v4, 0);
  v6 = *(a1 - 3);
  v165 = 1;
  v119 = v6;
  v7 = (__m128i *)&v166;
  do
  {
    v7->m128i_i64[0] = -8;
    ++v7;
  }
  while ( v7 != &v168 );
  LODWORD(v8) = 0;
  v134 = 0;
  v142 = v144;
  v143 = 0x400000000LL;
  v129 = 0;
  v9 = *(_QWORD *)(a2 + 48);
  v130 = *(_QWORD *)(a2 + 40) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v9 != v130 )
  {
    v10 = 0;
    while ( 1 )
    {
      if ( !v9 )
LABEL_259:
        BUG();
      v137 = v9 - 24;
      if ( *(_BYTE *)(v9 - 8) != 78 )
        break;
      v81 = *(_QWORD *)(v9 - 48);
      if ( *(_BYTE *)(v81 + 16) || (*(_BYTE *)(v81 + 33) & 0x20) == 0 || (unsigned int)(*(_DWORD *)(v81 + 36) - 35) > 3 )
        break;
      v82 = (unsigned int)v143;
      if ( (unsigned int)v143 >= HIDWORD(v143) )
      {
        sub_16CD150((__int64)&v142, v144, 0, 8, v8, (int)v5);
        v82 = (unsigned int)v143;
      }
      *(_QWORD *)&v142[8 * v82] = v137;
      LODWORD(v143) = v143 + 1;
LABEL_78:
      v9 = *(_QWORD *)(v9 + 8);
      if ( v9 == v130 )
      {
        LODWORD(v8) = v10;
        goto LABEL_80;
      }
    }
    if ( v10 == 1 )
      goto LABEL_158;
    if ( (unsigned __int8)sub_14AF470(v137, 0, 0, 0) )
    {
      if ( v129 )
      {
        v134 = v9 - 24;
        v13 = (__int64 *)(v9 - 24);
      }
      else
      {
        v13 = (__int64 *)(v9 - 24);
        if ( (unsigned int)sub_1B44750(v137, a3, v11, v12, v8, (int)v5) > dword_4FB7680 )
          goto LABEL_158;
      }
LABEL_64:
      v21 = 3LL * (*(_DWORD *)(v9 - 4) & 0xFFFFFFF);
      v22 = (__int64 *)(v137 - v21 * 8);
      if ( (*(_BYTE *)(v9 - 1) & 0x40) != 0 )
      {
        v22 = *(__int64 **)(v9 - 32);
        v13 = &v22[v21];
      }
      while ( 1 )
      {
        if ( v22 == v13 )
        {
          v10 = 1;
          goto LABEL_78;
        }
        v23 = *v22;
        if ( *(_BYTE *)(*v22 + 16) > 0x17u
          && v3 == *(_QWORD *)(v23 + 40)
          && !(unsigned __int8)sub_15F3040(*v22)
          && !sub_15F3330(v23) )
        {
          break;
        }
LABEL_66:
        v22 += 3;
      }
      if ( (v165 & 1) != 0 )
      {
        m128i_i64 = &v166;
        v25 = 3;
      }
      else
      {
        v77 = v167;
        m128i_i64 = v166->m128i_i64;
        if ( !v167 )
        {
          v79 = v165;
          ++v164;
          v27 = 0;
          v80 = ((unsigned int)v165 >> 1) + 1;
          goto LABEL_165;
        }
        v25 = v167 - 1;
      }
      v26 = v25 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v27 = &m128i_i64[2 * v26];
      v8 = *v27;
      if ( v23 == *v27 )
      {
        v28 = *((_DWORD *)v27 + 2) + 1;
        goto LABEL_76;
      }
      v83 = 1;
      v5 = 0;
      while ( v8 != -8 )
      {
        if ( v5 || v8 != -16 )
          v27 = v5;
        LODWORD(v5) = v83 + 1;
        v26 = v25 & (v83 + v26);
        v106 = &m128i_i64[2 * v26];
        v8 = *v106;
        if ( v23 == *v106 )
        {
          v28 = *((_DWORD *)v106 + 2) + 1;
          v27 = v106;
          goto LABEL_76;
        }
        ++v83;
        v5 = v27;
        v27 = &m128i_i64[2 * v26];
      }
      v79 = v165;
      LODWORD(v8) = 12;
      v77 = 4;
      if ( v5 )
        v27 = v5;
      ++v164;
      v80 = ((unsigned int)v165 >> 1) + 1;
      if ( (v165 & 1) != 0 )
      {
LABEL_166:
        if ( 4 * v80 >= (unsigned int)v8 )
        {
          sub_1A278A0((__int64)&v164, 2 * v77);
          if ( (v165 & 1) != 0 )
          {
            v84 = &v166;
            v85 = 3;
          }
          else
          {
            v84 = v166->m128i_i64;
            if ( !v167 )
              goto LABEL_258;
            v85 = v167 - 1;
          }
          v79 = v165;
          v86 = v85 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v27 = &v84[2 * v86];
          v8 = *v27;
          if ( *v27 == v23 )
            goto LABEL_168;
          LODWORD(v5) = 1;
          v87 = 0;
          while ( v8 != -8 )
          {
            if ( !v87 && v8 == -16 )
              v87 = v27;
            v117 = (_DWORD)v5 + 1;
            LODWORD(v5) = v86 + (_DWORD)v5;
            v86 = v85 & (unsigned int)v5;
            v27 = &v84[2 * (v85 & (unsigned int)v5)];
            v8 = *v27;
            if ( v23 == *v27 )
              goto LABEL_200;
            LODWORD(v5) = v117;
          }
        }
        else
        {
          if ( v77 - HIDWORD(v165) - v80 > v77 >> 3 )
            goto LABEL_168;
          sub_1A278A0((__int64)&v164, v77);
          if ( (v165 & 1) != 0 )
          {
            v88 = &v166;
            v89 = 3;
          }
          else
          {
            v88 = v166->m128i_i64;
            if ( !v167 )
            {
LABEL_258:
              LODWORD(v165) = (2 * ((unsigned int)v165 >> 1) + 2) | v165 & 1;
              BUG();
            }
            v89 = v167 - 1;
          }
          v79 = v165;
          v90 = v89 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v27 = &v88[2 * v90];
          v8 = *v27;
          if ( v23 == *v27 )
            goto LABEL_168;
          LODWORD(v5) = 1;
          v87 = 0;
          while ( v8 != -8 )
          {
            if ( !v87 && v8 == -16 )
              v87 = v27;
            v91 = (_DWORD)v5 + 1;
            LODWORD(v5) = v90 + (_DWORD)v5;
            v90 = v89 & (unsigned int)v5;
            v27 = &v88[2 * (v89 & (unsigned int)v5)];
            v8 = *v27;
            if ( v23 == *v27 )
              goto LABEL_200;
            LODWORD(v5) = v91;
          }
        }
        if ( v87 )
          v27 = v87;
LABEL_200:
        v79 = v165;
LABEL_168:
        LODWORD(v165) = (2 * (v79 >> 1) + 2) | v79 & 1;
        if ( *v27 != -8 )
          --HIDWORD(v165);
        *v27 = v23;
        v28 = 1;
        *((_DWORD *)v27 + 2) = 0;
LABEL_76:
        *((_DWORD *)v27 + 2) = v28;
        goto LABEL_66;
      }
      v77 = v167;
LABEL_165:
      LODWORD(v8) = 3 * v77;
      goto LABEL_166;
    }
    if ( !byte_4FB73E0 )
      goto LABEL_158;
    if ( *(_BYTE *)(v9 - 8) != 55 )
      goto LABEL_158;
    v13 = (__int64 *)(v9 - 24);
    if ( sub_15F32D0(v137) )
      goto LABEL_158;
    v128 = *(_BYTE *)(v9 - 6) & 1;
    if ( v128 )
      goto LABEL_158;
    v124 = *(_QWORD *)(v9 - 48);
    sub_1580910(&v168);
    v148 = v168;
    sub_1974F30((__int64)v149, (__int64)&v169);
    v176 = v148;
    sub_1974F30((__int64)&v177, (__int64)v149);
    v145 = v176;
    sub_1974F30((__int64)v146, (__int64)&v177);
    if ( v179 )
      v179(&v177, &v177, 3);
    v154 = v173;
    sub_1974F30((__int64)v155, (__int64)v174);
    v176 = v154;
    sub_1974F30((__int64)&v177, (__int64)v155);
    v151 = v176;
    sub_1974F30((__int64)v152, (__int64)&v177);
    if ( v179 )
      v179(&v177, &v177, 3);
    v161 = v145;
    sub_1974F30((__int64)v162, (__int64)v146);
    v158 = v151;
    sub_1974F30((__int64)v159, (__int64)v152);
    v176 = v158;
    sub_1974F30((__int64)&v177, (__int64)v159);
    v181 = v161;
    sub_1974F30((__int64)v182, (__int64)v162);
    if ( v160 )
      v160(v159, v159, 3);
    if ( v163 )
      v163(v162, v162, 3);
    if ( v153 )
      v153(v152, v152, 3);
    if ( v156 )
      v156(v155, v155, 3);
    if ( v147 )
      v147(v146, v146, 3);
    if ( v150 )
      v150(v149, v149, 3);
    if ( v175 )
      v175(v174, v174, 3);
    if ( v171 )
      v171(&v169, &v169, 3);
    v154 = v176;
    sub_1974F30((__int64)v155, (__int64)&v177);
    v158 = v181;
    sub_1974F30((__int64)v159, (__int64)v182);
    v126 = v3;
    v127 = 10;
    v125 = v9;
LABEL_35:
    v168 = v158;
    sub_1974F30((__int64)&v169, (__int64)v159);
    v161 = v154;
    sub_1974F30((__int64)v162, (__int64)v155);
    v14 = v161.m128i_i64[0];
    v15 = v168.m128i_i64[0];
    if ( v163 )
      v163(v162, v162, 3);
    if ( v171 )
      v171(&v169, &v169, 3);
    if ( v14 == v15 )
      goto LABEL_202;
    v16 = &v169;
    v168 = v154;
    sub_1974F30((__int64)&v169, (__int64)v155);
    do
    {
      v168.m128i_i64[0] = *(_QWORD *)v168.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
      v18 = v168.m128i_i64[0];
      if ( v168.m128i_i64[0] )
        v18 = v168.m128i_i64[0] - 24;
      if ( !v171 )
        goto LABEL_184;
      v16 = &v169;
    }
    while ( !v172(&v169, v18) );
    v19 = v168.m128i_i64[0];
    if ( v168.m128i_i64[0] )
      v19 = v168.m128i_i64[0] - 24;
    if ( v171 )
      v171(&v169, &v169, 3);
    if ( !--v127 )
    {
LABEL_202:
      sub_A17130((__int64)v159);
      sub_A17130((__int64)v155);
      sub_A17130((__int64)v182);
      sub_A17130((__int64)&v177);
      goto LABEL_159;
    }
    if ( (unsigned __int8)sub_15F3040(v19) || (v16 = (char *)v19, sub_15F3330(v19)) )
    {
      v20 = v19;
      v9 = v125;
      v3 = v126;
      if ( *(_BYTE *)(v20 + 16) != 55 )
        goto LABEL_53;
    }
    else
    {
      if ( *(_BYTE *)(v19 + 16) != 55 )
      {
        while ( 1 )
        {
          v154.m128i_i64[0] = *(_QWORD *)v154.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
          v18 = v154.m128i_i64[0];
          if ( v154.m128i_i64[0] )
            v18 = v154.m128i_i64[0] - 24;
          if ( !v156 )
            break;
          v16 = v155;
          if ( v157(v155) )
            goto LABEL_35;
        }
LABEL_184:
        sub_4263D6(v16, v18, v17);
      }
      v20 = v19;
      v9 = v125;
      v3 = v126;
    }
    if ( v124 == *(_QWORD *)(v20 - 24) )
    {
      v129 = *(_QWORD *)(v20 - 48);
LABEL_54:
      if ( v160 )
        v160(v159, v159, 3);
      if ( v156 )
        v156(v155, v155, 3);
      if ( v183 )
        v183(v182, v182, 3);
      if ( v179 )
        v179(&v177, &v177, 3);
      if ( v129 )
      {
        v134 = v137;
        goto LABEL_64;
      }
LABEL_158:
      v128 = 0;
      goto LABEL_159;
    }
LABEL_53:
    v129 = 0;
    goto LABEL_54;
  }
LABEL_80:
  v29 = v165 & 1;
  if ( (unsigned int)v165 >> 1 )
  {
    if ( v29 )
    {
      v33 = &v168;
      v32 = (__m128i *)&v166;
    }
    else
    {
      v30 = v167;
      v31 = v166->m128i_i64;
      v32 = v166;
      v33 = &v166[v167];
      if ( v166 == v33 )
        goto LABEL_88;
    }
    do
    {
      if ( v32->m128i_i64[0] != -16 && v32->m128i_i64[0] != -8 )
        break;
      ++v32;
    }
    while ( v32 != v33 );
  }
  else
  {
    if ( v29 )
    {
      v104 = &v166;
      v105 = 8;
    }
    else
    {
      v104 = v166->m128i_i64;
      v105 = 2LL * v167;
    }
    v32 = (__m128i *)&v104[v105];
    v33 = v32;
  }
  if ( v29 )
  {
    v31 = &v166;
    v34 = 8;
    goto LABEL_89;
  }
  v31 = v166->m128i_i64;
  v30 = v167;
LABEL_88:
  v34 = 2 * v30;
LABEL_89:
  v35 = (__m128i *)&v31[v34];
  v36 = v8;
  if ( v35 != v32 )
  {
    do
    {
      if ( (unsigned int)sub_1648EF0(v32->m128i_i64[0]) == v32->m128i_i32[2] )
      {
        if ( v36 == 1 )
          goto LABEL_158;
        v36 = 1;
      }
      do
        ++v32;
      while ( v33 != v32 && (v32->m128i_i64[0] == -8 || v32->m128i_i64[0] == -16) );
    }
    while ( v32 != v35 );
    LODWORD(v8) = v36;
  }
  v138 = v8;
  v37 = sub_157F280(v118);
  v132 = v38;
  if ( v37 == v38 )
    goto LABEL_236;
  v39 = 0;
  v40 = v37;
  v131 = v138;
  do
  {
    v41 = 0x17FFFFFFE8LL;
    v42 = *(_BYTE *)(v40 + 23) & 0x40;
    v43 = *(_DWORD *)(v40 + 20) & 0xFFFFFFF;
    if ( (*(_DWORD *)(v40 + 20) & 0xFFFFFFF) != 0 )
    {
      v38 = 24LL * *(unsigned int *)(v40 + 56) + 8;
      v44 = 0;
      do
      {
        v45 = v40 - 24LL * (unsigned int)v43;
        if ( v42 )
          v45 = *(_QWORD *)(v40 - 8);
        if ( v3 == *(_QWORD *)(v45 + v38) )
        {
          v41 = 24 * v44;
          goto LABEL_107;
        }
        ++v44;
        v38 += 8;
      }
      while ( (_DWORD)v43 != (_DWORD)v44 );
      v41 = 0x17FFFFFFE8LL;
    }
LABEL_107:
    if ( v42 )
    {
      v46 = *(_QWORD *)(v40 - 8);
    }
    else
    {
      v38 = 24LL * (unsigned int)v43;
      v46 = v40 - v38;
    }
    v47 = *(_QWORD *)(v46 + v41);
    v48 = 0x17FFFFFFE8LL;
    if ( (_DWORD)v43 )
    {
      v49 = 0;
      v38 = v46 + 24LL * *(unsigned int *)(v40 + 56);
      do
      {
        if ( a2 == *(_QWORD *)(v38 + 8 * v49 + 8) )
        {
          v48 = 24 * v49;
          goto LABEL_114;
        }
        ++v49;
      }
      while ( (_DWORD)v43 != (_DWORD)v49 );
      v48 = 0x17FFFFFFE8LL;
    }
LABEL_114:
    v139 = *(_QWORD *)(v46 + v48);
    if ( v47 == v139 )
      goto LABEL_124;
    if ( sub_1B43710(v47, v40, v38, v43) || sub_1B43710(v139, v40, v50, v51) )
      goto LABEL_158;
    v52 = *(_BYTE *)(v139 + 16);
    if ( *(_BYTE *)(v47 + 16) == 5 )
    {
      if ( v52 != 5 )
      {
        if ( !(unsigned __int8)sub_14AF470(v47, 0, 0, 0) )
          goto LABEL_158;
        v57 = sub_1B44750(v47, a3, v53, v54, v55, v56);
        goto LABEL_121;
      }
      if ( !(unsigned __int8)sub_14AF470(v139, 0, 0, 0) || !(unsigned __int8)sub_14AF470(v47, 0, 0, 0) )
        goto LABEL_158;
      v103 = sub_1B44750(v47, a3, v99, v100, v101, v102);
      v97 = v139;
      v98 = v103;
    }
    else
    {
      v39 = 1;
      if ( v52 != 5 )
        goto LABEL_124;
      v92 = sub_14AF470(v139, 0, 0, 0);
      v97 = v139;
      if ( !v92 )
        goto LABEL_158;
      v98 = 0;
    }
    v57 = sub_1B44750(v97, a3, v93, v94, v95, v96) + v98;
LABEL_121:
    if ( 2 * dword_4FB7680 < v57 || v131 == 1 )
      goto LABEL_158;
    v131 = 1;
    v39 = 1;
LABEL_124:
    v58 = *(_QWORD *)(v40 + 32);
    if ( !v58 )
      goto LABEL_259;
    v40 = 0;
    if ( *(_BYTE *)(v58 - 8) == 77 )
      v40 = v58 - 24;
  }
  while ( v132 != v40 );
  if ( !v39 )
  {
LABEL_236:
    if ( byte_4FB73E0 == 1 && v129 )
      goto LABEL_238;
    goto LABEL_158;
  }
  if ( !v129 )
  {
    v140 = a1 + 3;
    goto LABEL_131;
  }
LABEL_238:
  v107 = sub_16498A0((__int64)a1);
  v108 = a1[6];
  v176.m128i_i64[0] = 0;
  v178 = v107;
  v109 = a1[5];
  v179 = 0;
  v180 = 0;
  v181 = 0u;
  v176.m128i_i64[1] = v109;
  v140 = a1 + 3;
  v177 = a1 + 3;
  v168.m128i_i64[0] = v108;
  if ( v108 )
  {
    sub_1623A60((__int64)&v168, v108, 2);
    if ( v176.m128i_i64[0] )
      sub_161E7C0((__int64)&v176, v176.m128i_i64[0]);
    v176.m128i_i64[0] = v168.m128i_i64[0];
    if ( v168.m128i_i64[0] )
      sub_1623210((__int64)&v168, (unsigned __int8 *)v168.m128i_i64[0], (__int64)&v176);
  }
  v110 = *(_QWORD *)(v134 - 48);
  if ( a2 != v119 )
  {
    v110 = v129;
    v129 = *(_QWORD *)(v134 - 48);
  }
  v170 = 1;
  v168.m128i_i64[0] = (__int64)"spec.store.select";
  v169 = 3;
  v111 = sub_1B47760(v176.m128i_i64, v120, v110, v129, v168.m128i_i64, (__int64)a1);
  sub_1593B40((_QWORD *)(v134 - 48), (__int64)v111);
  v112 = sub_15C70A0(v134 + 48);
  v113 = sub_15C70A0((__int64)(a1 + 6));
  sub_15AC0B0(v134, v113, v112);
  sub_17CD270(v176.m128i_i64);
LABEL_131:
  v59 = *(unsigned __int64 **)(a2 + 48);
  v60 = a2 + 40;
  if ( (unsigned __int64 *)(a2 + 40) != v59 )
  {
    do
    {
      v61 = (__int64)(v59 - 3);
      if ( !v59 )
        v61 = 0;
      sub_1624960(v61, 0, 0);
      v59 = (unsigned __int64 *)v59[1];
    }
    while ( (unsigned __int64 *)v60 != v59 );
    v59 = *(unsigned __int64 **)(a2 + 48);
  }
  v135 = *(_QWORD *)(a2 + 40);
  v62 = (unsigned __int64 *)(v135 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (unsigned __int64 *)(v135 & 0xFFFFFFFFFFFFFFF8LL) != v59 && v140 != v62 )
  {
    if ( v3 + 40 != v60 )
      sub_157EA80(v3 + 40, v60, (__int64)v59, v135 & 0xFFFFFFFFFFFFFFF8LL);
    if ( v140 != v62 && v62 != v59 )
    {
      v63 = *v62 & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)((*v59 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v62;
      *v62 = *v62 & 7 | *v59 & 0xFFFFFFFFFFFFFFF8LL;
      v64 = a1[3];
      *(_QWORD *)(v63 + 8) = v140;
      *v59 = v64 & 0xFFFFFFFFFFFFFFF8LL | *v59 & 7;
      *(_QWORD *)((v64 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v59;
      a1[3] = v63 | a1[3] & 7LL;
    }
  }
  v65 = sub_16498A0((__int64)a1);
  v176 = 0u;
  v178 = v65;
  v177 = 0;
  v179 = 0;
  v180 = 0;
  v181 = 0u;
  sub_17050D0(v176.m128i_i64, (__int64)a1);
  v161.m128i_i64[0] = sub_157F280(v118);
  v133 = v66;
  if ( v66 != v161.m128i_i64[0] )
  {
    v67 = v161.m128i_i64[0];
    do
    {
      v75 = (unsigned int)sub_1B46990(v67, v3);
      v76 = sub_1B46990(v67, a2);
      if ( (*(_BYTE *)(v67 + 23) & 0x40) != 0 )
        v68 = *(_QWORD *)(v67 - 8);
      else
        v68 = v67 - 24LL * (*(_DWORD *)(v67 + 20) & 0xFFFFFFF);
      v69 = 3LL * v76;
      v141 = 24 * v75;
      v70 = *(_QWORD *)(v68 + 24 * v75);
      v136 = 8 * v69;
      v71 = *(_QWORD *)(v68 + 8 * v69);
      if ( v71 != v70 )
      {
        if ( a2 != v119 )
        {
          v71 = *(_QWORD *)(v68 + 24 * v75);
          v70 = *(_QWORD *)(v68 + 8 * v69);
        }
        v170 = 1;
        v168.m128i_i64[0] = (__int64)"spec.select";
        v169 = 3;
        v72 = sub_1B47760(v176.m128i_i64, v120, v71, v70, v168.m128i_i64, (__int64)a1);
        v73 = sub_13CF970(v67);
        sub_1593B40((_QWORD *)(v73 + v141), (__int64)v72);
        v74 = sub_13CF970(v67);
        sub_1593B40((_QWORD *)(v74 + v136), (__int64)v72);
      }
      sub_1B42F80((__int64)&v161);
      v67 = v161.m128i_i64[0];
    }
    while ( v133 != v161.m128i_i64[0] );
  }
  v114 = (_QWORD **)v142;
  v115 = (_QWORD **)&v142[8 * (unsigned int)v143];
  if ( v142 != (_BYTE *)v115 )
  {
    do
    {
      v116 = *v114++;
      sub_15F20C0(v116);
    }
    while ( v115 != v114 );
  }
  sub_17CD270(v176.m128i_i64);
  v128 = 1;
LABEL_159:
  if ( v142 != v144 )
    _libc_free((unsigned __int64)v142);
  if ( (v165 & 1) == 0 )
    j___libc_free_0(v166);
  return v128;
}
