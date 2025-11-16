// Function: sub_3717E20
// Address: 0x3717e20
//
unsigned __int64 *__fastcall sub_3717E20(unsigned __int64 *a1, int *a2, char a3)
{
  char *v3; // r13
  int v4; // eax
  size_t v5; // r15
  char **i; // r14
  char *v7; // rax
  int v8; // eax
  unsigned int v9; // ebx
  _QWORD *v10; // r10
  __int64 v11; // rax
  __int64 *v12; // r10
  __int64 v13; // rcx
  unsigned __int64 *v14; // r15
  char *v15; // rax
  char *v16; // r13
  int v17; // r11d
  char *v18; // rsi
  char *v19; // r8
  __int64 v20; // r13
  __int64 v21; // rbx
  _QWORD *v22; // rdi
  unsigned __int64 v23; // r8
  __int64 v24; // r13
  __int64 v25; // rbx
  _QWORD *v26; // rdi
  __int64 v27; // r13
  __int64 v28; // rax
  char *v29; // rdi
  char *v30; // rsi
  _QWORD *v31; // rax
  __int64 v32; // r13
  char *v33; // rsi
  __int64 v35; // r9
  char *v36; // rcx
  __int64 v37; // rax
  __int64 v38; // r15
  __int64 v39; // r12
  char *v40; // rax
  size_t v41; // r13
  __int64 v42; // rax
  __int64 v43; // r8
  __int64 v44; // rbx
  __int64 *v45; // r14
  char *v46; // r10
  __int64 v47; // r15
  __int64 v48; // r12
  __int64 v49; // rax
  size_t v50; // r14
  __int64 v51; // rax
  __int64 v52; // r8
  __int64 v53; // r13
  char *v54; // rbx
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // r9
  __int64 v58; // r8
  _QWORD *v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // r9
  _QWORD *v64; // r13
  int v65; // esi
  __int64 v66; // rax
  __int64 v67; // rcx
  __int64 v68; // rax
  char *v69; // rdi
  __int64 v70; // rax
  char *v71; // rdi
  char *v72; // rdi
  char v73; // bl
  char *v74; // rax
  char *v75; // rdi
  char *v76; // rax
  __int64 v77; // rbx
  _QWORD *v78; // rax
  _QWORD *v79; // r13
  __int64 v80; // rax
  char *v81; // rdi
  __int64 v82; // rax
  _QWORD *v83; // rbx
  __int64 v84; // r13
  _QWORD *v85; // rax
  _QWORD *v86; // r12
  __int64 v87; // rax
  char *v88; // rdi
  char *v89; // rsi
  char *v90; // rbx
  char *v91; // r12
  _QWORD *v92; // rbx
  _QWORD *v93; // r12
  _QWORD *v94; // r12
  _QWORD *v95; // r13
  unsigned __int64 *v96; // [rsp+10h] [rbp-1290h]
  __int64 v97; // [rsp+18h] [rbp-1288h]
  __int64 v99; // [rsp+28h] [rbp-1278h]
  char *v100; // [rsp+28h] [rbp-1278h]
  __int64 v101; // [rsp+28h] [rbp-1278h]
  __int64 v102; // [rsp+38h] [rbp-1268h]
  __int64 v103; // [rsp+38h] [rbp-1268h]
  __int64 *v104; // [rsp+40h] [rbp-1260h]
  char *j; // [rsp+40h] [rbp-1260h]
  char *v106; // [rsp+40h] [rbp-1260h]
  _QWORD *v107; // [rsp+40h] [rbp-1260h]
  int v108; // [rsp+48h] [rbp-1258h]
  __int64 v109; // [rsp+48h] [rbp-1258h]
  __int64 v110; // [rsp+48h] [rbp-1258h]
  unsigned __int64 v111; // [rsp+48h] [rbp-1258h]
  unsigned __int64 v112; // [rsp+48h] [rbp-1258h]
  char v113; // [rsp+53h] [rbp-124Dh] BYREF
  __int64 v114; // [rsp+54h] [rbp-124Ch]
  int v115; // [rsp+5Ch] [rbp-1244h]
  _QWORD *v116; // [rsp+60h] [rbp-1240h] BYREF
  __int128 v117; // [rsp+68h] [rbp-1238h]
  __int64 v118; // [rsp+78h] [rbp-1228h]
  __int64 v119; // [rsp+80h] [rbp-1220h]
  char *v120; // [rsp+90h] [rbp-1210h] BYREF
  size_t v121; // [rsp+98h] [rbp-1208h]
  _QWORD v122[2]; // [rsp+A0h] [rbp-1200h] BYREF
  int v123; // [rsp+B0h] [rbp-11F0h]
  char v124; // [rsp+B4h] [rbp-11ECh]
  char v125; // [rsp+B8h] [rbp-11E8h] BYREF
  __int64 v126; // [rsp+1B8h] [rbp-10E8h]
  __int64 v127; // [rsp+1C0h] [rbp-10E0h]
  __int64 v128; // [rsp+1C8h] [rbp-10D8h]
  int v129; // [rsp+1D0h] [rbp-10D0h]
  _QWORD *v130; // [rsp+1D8h] [rbp-10C8h]
  __int64 v131; // [rsp+1E0h] [rbp-10C0h]
  __int64 v132; // [rsp+1E8h] [rbp-10B8h]
  __int64 v133; // [rsp+1F0h] [rbp-10B0h]
  int v134; // [rsp+1F8h] [rbp-10A8h]
  __int64 v135; // [rsp+200h] [rbp-10A0h]
  _QWORD v136[7]; // [rsp+208h] [rbp-1098h] BYREF
  _QWORD v137[4]; // [rsp+240h] [rbp-1060h] BYREF
  int v138; // [rsp+260h] [rbp-1040h]
  __int64 v139; // [rsp+268h] [rbp-1038h]
  char *v140; // [rsp+270h] [rbp-1030h]
  __int64 v141; // [rsp+278h] [rbp-1028h]
  int v142; // [rsp+280h] [rbp-1020h]
  char v143; // [rsp+284h] [rbp-101Ch]
  char v144; // [rsp+288h] [rbp-1018h] BYREF
  __int64 v145; // [rsp+8A0h] [rbp-A00h]
  __int64 v146; // [rsp+8A8h] [rbp-9F8h]
  __int64 v147; // [rsp+8B0h] [rbp-9F0h]
  unsigned int v148; // [rsp+8B8h] [rbp-9E8h]
  __int64 v149; // [rsp+8C0h] [rbp-9E0h]
  __int64 v150; // [rsp+8C8h] [rbp-9D8h]
  __int64 v151; // [rsp+8D0h] [rbp-9D0h]
  unsigned int v152; // [rsp+8D8h] [rbp-9C8h]
  char *v153; // [rsp+8E0h] [rbp-9C0h] BYREF
  int v154; // [rsp+8E8h] [rbp-9B8h]
  char v155; // [rsp+8F0h] [rbp-9B0h] BYREF
  __int64 v156; // [rsp+970h] [rbp-930h]
  int v157; // [rsp+978h] [rbp-928h]
  char *v158; // [rsp+980h] [rbp-920h] BYREF
  __int128 v159; // [rsp+988h] [rbp-918h] BYREF
  const char *v160; // [rsp+998h] [rbp-908h]
  __int64 v161; // [rsp+9A0h] [rbp-900h]
  int v162; // [rsp+9A8h] [rbp-8F8h]
  _BYTE v163[2016]; // [rsp+9B0h] [rbp-8F0h] BYREF
  __int64 v164; // [rsp+1190h] [rbp-110h]
  __int64 v165; // [rsp+1198h] [rbp-108h]
  __int64 v166; // [rsp+11A0h] [rbp-100h]
  unsigned int v167; // [rsp+11A8h] [rbp-F8h]
  __int64 v168; // [rsp+11B0h] [rbp-F0h]
  __int64 v169; // [rsp+11B8h] [rbp-E8h]
  __int64 v170; // [rsp+11C0h] [rbp-E0h]
  unsigned int v171; // [rsp+11C8h] [rbp-D8h]
  char *v172; // [rsp+11D0h] [rbp-D0h] BYREF
  __int64 v173; // [rsp+11D8h] [rbp-C8h]
  _BYTE v174[128]; // [rsp+11E0h] [rbp-C0h] BYREF
  __int64 v175; // [rsp+1260h] [rbp-40h]
  int v176; // [rsp+1268h] [rbp-38h]

  v3 = "__CUDA_ARCH";
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  v108 = *a2;
  DWORD2(v159) = *a2;
  v160 = "__CUDA_FTZ";
  v4 = *((unsigned __int8 *)a2 + 4);
  v158 = "__CUDA_ARCH";
  v162 = v4;
  *(_QWORD *)&v159 = 11;
  v161 = 10;
  sub_C926D0((__int64)&v116, 2, 16);
  v5 = 11;
  for ( i = &v158; ; v108 = *((_DWORD *)i + 4) )
  {
    v7 = i[2];
    v120 = v3;
    v121 = v5;
    v122[0] = v7;
    v8 = sub_C92610();
    v9 = sub_C92740((__int64)&v116, v3, v5, v8);
    v10 = &v116[v9];
    if ( !*v10 )
      break;
    if ( *v10 == -8 )
    {
      --DWORD2(v117);
      break;
    }
    i += 3;
    if ( i == (char **)v163 )
      goto LABEL_10;
LABEL_3:
    v3 = *i;
    v5 = (size_t)i[1];
  }
  v104 = &v116[v9];
  v11 = sub_C7D670(v5 + 17, 8);
  v12 = v104;
  v13 = v11;
  if ( v5 )
  {
    v99 = v11;
    memcpy((void *)(v11 + 16), v3, v5);
    v12 = v104;
    v13 = v99;
  }
  i += 3;
  *(_BYTE *)(v13 + v5 + 16) = 0;
  *(_QWORD *)v13 = v5;
  *(_DWORD *)(v13 + 8) = v108;
  *v12 = v13;
  ++DWORD1(v117);
  sub_C929D0((__int64 *)&v116, v9);
  if ( i != (char **)v163 )
    goto LABEL_3;
LABEL_10:
  v158 = 0;
  *((_QWORD *)&v159 + 1) = 0x1000000000LL;
  v14 = a1;
  *(_QWORD *)&v159 = 0;
  if ( DWORD1(v117) )
  {
    sub_C92620((__int64)&v158, v117);
    v46 = v158;
    v106 = v158;
    v103 = (__int64)v116;
    *(_QWORD *)((char *)&v159 + 4) = *(_QWORD *)((char *)&v117 + 4);
    if ( (_DWORD)v159 )
    {
      v47 = 0;
      v48 = 8LL * (unsigned int)v159 + 8;
      v110 = 8LL * (unsigned int)(v159 - 1);
      v49 = (__int64)v116;
      while ( 1 )
      {
        v53 = *(_QWORD *)(v49 + v47);
        v54 = &v46[v47];
        if ( v53 == -8 || !v53 )
        {
          *(_QWORD *)v54 = v53;
          v48 += 4;
          if ( v110 == v47 )
            goto LABEL_66;
        }
        else
        {
          v50 = *(_QWORD *)v53;
          v51 = sub_C7D670(*(_QWORD *)v53 + 17LL, 8);
          v52 = v51;
          if ( v50 )
          {
            v101 = v51;
            memcpy((void *)(v51 + 16), (const void *)(v53 + 16), v50);
            v52 = v101;
          }
          *(_BYTE *)(v52 + v50 + 16) = 0;
          *(_QWORD *)v52 = v50;
          *(_DWORD *)(v52 + 8) = *(_DWORD *)(v53 + 8);
          *(_QWORD *)v54 = v52;
          *(_DWORD *)&v106[v48] = *(_DWORD *)(v103 + v48);
          v48 += 4;
          if ( v110 == v47 )
          {
LABEL_66:
            v14 = a1;
            break;
          }
        }
        v49 = (__int64)v116;
        v46 = v158;
        v47 += 8;
      }
    }
  }
  v15 = (char *)sub_22077B0(0x20u);
  v16 = v15;
  if ( v15 )
  {
    v17 = DWORD1(v159);
    *((_QWORD *)v15 + 1) = 0;
    *((_QWORD *)v15 + 2) = 0;
    *(_QWORD *)v15 = &unk_4A30FA0;
    *((_QWORD *)v15 + 3) = 0x1000000000LL;
    if ( v17 )
    {
      sub_C92620((__int64)(v15 + 8), v159);
      v35 = *((_QWORD *)v16 + 1);
      v36 = v158;
      v37 = *((unsigned int *)v16 + 4);
      v102 = v35;
      v100 = v158;
      *(_QWORD *)(v16 + 20) = *(_QWORD *)((char *)&v159 + 4);
      if ( (_DWORD)v37 )
      {
        v96 = v14;
        v38 = 8 * v37 + 8;
        v39 = 0;
        v109 = 8LL * (unsigned int)(v37 - 1);
        v40 = v36;
        for ( j = v16; ; v35 = *((_QWORD *)j + 1) )
        {
          v44 = *(_QWORD *)&v40[v39];
          v45 = (__int64 *)(v35 + v39);
          if ( v44 == -8 || !v44 )
          {
            *v45 = v44;
            v38 += 4;
            if ( v39 == v109 )
              goto LABEL_56;
          }
          else
          {
            v41 = *(_QWORD *)v44;
            v42 = sub_C7D670(*(_QWORD *)v44 + 17LL, 8);
            v43 = v42;
            if ( v41 )
            {
              v97 = v42;
              memcpy((void *)(v42 + 16), (const void *)(v44 + 16), v41);
              v43 = v97;
            }
            *(_BYTE *)(v43 + v41 + 16) = 0;
            *(_QWORD *)v43 = v41;
            *(_DWORD *)(v43 + 8) = *(_DWORD *)(v44 + 8);
            *v45 = v43;
            *(_DWORD *)(v102 + v38) = *(_DWORD *)&v100[v38];
            v38 += 4;
            if ( v39 == v109 )
            {
LABEL_56:
              v16 = j;
              v14 = v96;
              break;
            }
          }
          v40 = v158;
          v39 += 8;
        }
      }
    }
  }
  v120 = v16;
  v18 = (char *)v14[1];
  if ( v18 == (char *)v14[2] )
  {
    sub_2275C60(v14, v18, &v120);
    v16 = v120;
  }
  else
  {
    if ( v18 )
    {
      *(_QWORD *)v18 = v16;
      v14[1] += 8LL;
      goto LABEL_16;
    }
    v14[1] = 8;
  }
  if ( v16 )
    (*(void (__fastcall **)(char *))(*(_QWORD *)v16 + 8LL))(v16);
LABEL_16:
  v19 = v158;
  if ( DWORD1(v159) && (_DWORD)v159 )
  {
    v20 = 8LL * (unsigned int)v159;
    v21 = 0;
    do
    {
      v22 = *(_QWORD **)&v19[v21];
      if ( v22 != (_QWORD *)-8LL && v22 )
      {
        sub_C7D6A0((__int64)v22, *v22 + 17LL, 8);
        v19 = v158;
      }
      v21 += 8;
    }
    while ( v21 != v20 );
  }
  _libc_free((unsigned __int64)v19);
  if ( DWORD1(v117) )
  {
    v23 = (unsigned __int64)v116;
    if ( (_DWORD)v117 )
    {
      v24 = 8LL * (unsigned int)v117;
      v25 = 0;
      do
      {
        v26 = *(_QWORD **)(v23 + v25);
        if ( v26 != (_QWORD *)-8LL && v26 )
        {
          sub_C7D6A0((__int64)v26, *v26 + 17LL, 8);
          v23 = (unsigned __int64)v116;
        }
        v25 += 8;
      }
      while ( v25 != v24 );
    }
  }
  else
  {
    v23 = (unsigned __int64)v116;
  }
  _libc_free(v23);
  v27 = sub_22077B0(0x10u);
  if ( v27 )
    *(_QWORD *)v27 = &unk_4A115B8;
  v28 = sub_22077B0(0x18u);
  v29 = (char *)v28;
  if ( v28 )
  {
    *(_BYTE *)(v28 + 16) = 0;
    *(_QWORD *)(v28 + 8) = v27;
    v27 = 0;
    *(_QWORD *)v28 = &unk_4A0C478;
  }
  v158 = (char *)v28;
  v30 = (char *)v14[1];
  if ( v30 == (char *)v14[2] )
  {
    sub_2275C60(v14, v30, &v158);
    v29 = v158;
  }
  else
  {
    if ( v30 )
    {
      *(_QWORD *)v30 = v28;
      v14[1] += 8LL;
      goto LABEL_37;
    }
    v14[1] = 8;
  }
  if ( v29 )
    (*(void (__fastcall **)(char *))(*(_QWORD *)v29 + 8LL))(v29);
LABEL_37:
  if ( v27 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v27 + 8LL))(v27);
  LOBYTE(v120) = 0;
  v122[0] = &v125;
  v130 = v136;
  v136[1] = v137;
  v140 = &v144;
  v121 = 0;
  v122[1] = 32;
  v123 = 0;
  v124 = 1;
  v126 = 0;
  v127 = 0;
  v128 = 0;
  v129 = 0;
  v131 = 1;
  v132 = 0;
  v133 = 0;
  v134 = 1065353216;
  v135 = 0;
  v136[0] = 0;
  v136[2] = 1;
  v136[3] = 0;
  v136[4] = 0;
  v136[5] = 1065353216;
  v136[6] = 0;
  memset(v137, 0, sizeof(v137));
  v138 = 0;
  v139 = 0;
  v141 = 32;
  v142 = 0;
  v143 = 1;
  sub_234B220((__int64)&v158, (__int64)&v120);
  v31 = (_QWORD *)sub_22077B0(0x300u);
  v32 = (__int64)v31;
  if ( v31 )
  {
    *v31 = &unk_4A0E7F8;
    sub_234B220((__int64)(v31 + 1), (__int64)&v158);
  }
  v116 = (_QWORD *)v32;
  v33 = (char *)v14[1];
  if ( v33 == (char *)v14[2] )
  {
    sub_2275C60(v14, v33, &v116);
    v32 = (__int64)v116;
  }
  else
  {
    if ( v33 )
    {
      *(_QWORD *)v33 = v32;
      v14[1] += 8LL;
      goto LABEL_44;
    }
    v14[1] = 8;
  }
  if ( v32 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v32 + 8LL))(v32);
LABEL_44:
  sub_233AAF0((__int64)&v158);
  sub_233AAF0((__int64)&v120);
  if ( a3 )
  {
    LOBYTE(v114) = 0;
    HIDWORD(v114) = 1;
    LOBYTE(v115) = 0;
    v116 = 0;
    v117 = 0u;
    v118 = 0;
    v119 = 0;
    sub_F10C20((__int64)&v120, v114, v115);
    v58 = (unsigned int)v121;
    v158 = (char *)&v159 + 8;
    *(_QWORD *)&v159 = 0x10000000000LL;
    if ( (_DWORD)v121 )
      sub_3717CC0((__int64)&v158, &v120, v55, v56, (unsigned int)v121, v57);
    v164 = 1;
    ++v145;
    v165 = v146;
    ++v149;
    v166 = v147;
    v146 = 0;
    v147 = 0;
    v167 = v148;
    v148 = 0;
    v169 = v150;
    v168 = 1;
    v170 = v151;
    v150 = 0;
    v151 = 0;
    v171 = v152;
    v172 = v174;
    v152 = 0;
    v173 = 0x1000000000LL;
    if ( v154 )
      sub_3717CC0((__int64)&v172, &v153, v55, v56, v58, v57);
    v175 = v156;
    v176 = v157;
    v59 = (_QWORD *)sub_22077B0(0x8F8u);
    v64 = v59;
    if ( v59 )
    {
      v65 = v159;
      *v59 = &unk_4A11978;
      v59[1] = v59 + 3;
      v59[2] = 0x10000000000LL;
      if ( v65 )
        sub_3717CC0((__int64)(v59 + 1), &v158, v60, v61, v62, v63);
      v66 = v165;
      v67 = (unsigned int)v173;
      v64[259] = 1;
      ++v164;
      v64[260] = v66;
      ++v168;
      v64[261] = v166;
      v165 = 0;
      *((_DWORD *)v64 + 524) = v167;
      v166 = 0;
      v64[264] = v169;
      v167 = 0;
      v64[265] = v170;
      LODWORD(v66) = v171;
      v64[263] = 1;
      *((_DWORD *)v64 + 532) = v66;
      v64[267] = v64 + 269;
      v169 = 0;
      v170 = 0;
      v171 = 0;
      v64[268] = 0x1000000000LL;
      if ( (_DWORD)v67 )
        sub_3717CC0((__int64)(v64 + 267), &v172, v60, v67, v62, v63);
      v64[285] = v175;
      *((_DWORD *)v64 + 572) = v176;
    }
    if ( v172 != v174 )
      _libc_free((unsigned __int64)v172);
    sub_C7D6A0(v169, 8LL * v171, 8);
    sub_C7D6A0(v165, 16LL * v167, 8);
    if ( v158 != (char *)&v159 + 8 )
      _libc_free((unsigned __int64)v158);
    v68 = sub_22077B0(0x18u);
    v69 = (char *)v68;
    if ( v68 )
    {
      *(_QWORD *)(v68 + 8) = v64;
      v64 = 0;
      *(_WORD *)(v68 + 16) = 0;
      *(_QWORD *)v68 = &unk_4A12538;
    }
    v158 = (char *)v68;
    if ( (_QWORD)v117 == *((_QWORD *)&v117 + 1) )
    {
      sub_235A6C0((unsigned __int64 *)&v116, (char *)v117, &v158);
      v69 = v158;
    }
    else
    {
      if ( (_QWORD)v117 )
      {
        *(_QWORD *)v117 = v68;
        *(_QWORD *)&v117 = v117 + 8;
        goto LABEL_86;
      }
      *(_QWORD *)&v117 = 8;
    }
    if ( v69 )
      (*(void (__fastcall **)(char *))(*(_QWORD *)v69 + 8LL))(v69);
LABEL_86:
    if ( v64 )
      (*(void (__fastcall **)(_QWORD *))(*v64 + 8LL))(v64);
    if ( v153 != &v155 )
      _libc_free((unsigned __int64)v153);
    sub_C7D6A0(v150, 8LL * v152, 8);
    sub_C7D6A0(v146, 16LL * v148, 8);
    if ( v120 != (char *)v122 )
      _libc_free((unsigned __int64)v120);
    v70 = sub_22077B0(0x28u);
    v71 = (char *)v70;
    if ( v70 )
    {
      *(_QWORD *)(v70 + 8) = 0;
      *(_BYTE *)(v70 + 16) = 0;
      *(_DWORD *)(v70 + 20) = 0;
      *(_QWORD *)v70 = &unk_4A0ECB8;
      *(_QWORD *)(v70 + 24) = 0;
      *(_QWORD *)(v70 + 32) = 0;
    }
    v158 = (char *)v70;
    if ( (_QWORD)v117 == *((_QWORD *)&v117 + 1) )
    {
      sub_235A6C0((unsigned __int64 *)&v116, (char *)v117, &v158);
      v71 = v158;
    }
    else
    {
      if ( (_QWORD)v117 )
      {
        *(_QWORD *)v117 = v70;
        *(_QWORD *)&v117 = v117 + 8;
        goto LABEL_97;
      }
      *(_QWORD *)&v117 = 8;
    }
    if ( v71 )
      (*(void (__fastcall **)(char *))(*(_QWORD *)v71 + 8LL))(v71);
LABEL_97:
    v158 = 0;
    v159 = 0u;
    v160 = 0;
    v161 = 0;
    v72 = (char *)sub_22077B0(0x10u);
    if ( v72 )
      *(_QWORD *)v72 = &unk_4A0FFF8;
    v120 = v72;
    if ( (_QWORD)v159 == *((_QWORD *)&v159 + 1) )
    {
      sub_2353750((unsigned __int64 *)&v158, (char *)v159, &v120);
      v72 = v120;
    }
    else
    {
      if ( (_QWORD)v159 )
      {
        *(_QWORD *)v159 = v72;
        *(_QWORD *)&v159 = v159 + 8;
        goto LABEL_102;
      }
      *(_QWORD *)&v159 = 8;
    }
    if ( v72 )
      (*(void (__fastcall **)(char *))(*(_QWORD *)v72 + 8LL))(v72);
LABEL_102:
    sub_291E720(&v113, 0);
    v73 = v113;
    v74 = (char *)sub_22077B0(0x10u);
    v75 = v74;
    if ( v74 )
    {
      v74[8] = v73;
      *(_QWORD *)v74 = &unk_4A11C38;
    }
    v120 = v74;
    if ( (_QWORD)v159 == *((_QWORD *)&v159 + 1) )
    {
      sub_2353750((unsigned __int64 *)&v158, (char *)v159, &v120);
      v75 = v120;
    }
    else
    {
      if ( (_QWORD)v159 )
      {
        *(_QWORD *)v159 = v74;
        *(_QWORD *)&v159 = v159 + 8;
        goto LABEL_107;
      }
      *(_QWORD *)&v159 = 8;
    }
    if ( v75 )
      (*(void (__fastcall **)(char *))(*(_QWORD *)v75 + 8LL))(v75);
LABEL_107:
    v76 = v158;
    v77 = *((_QWORD *)&v159 + 1);
    v158 = 0;
    v111 = (unsigned __int64)v76;
    v107 = (_QWORD *)v159;
    v159 = 0u;
    v78 = (_QWORD *)sub_22077B0(0x30u);
    v79 = v78;
    if ( v78 )
    {
      v78[3] = v77;
      v78[4] = 0;
      v78[2] = v107;
      *v78 = &unk_4A0C438;
      v78[5] = 0;
      v78[1] = v111;
    }
    else
    {
      if ( (_QWORD *)v111 != v107 )
      {
        v94 = (_QWORD *)v111;
        do
        {
          if ( *v94 )
            (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v94 + 8LL))(*v94);
          ++v94;
        }
        while ( v107 != v94 );
      }
      if ( v111 )
        j_j___libc_free_0(v111);
    }
    v80 = sub_22077B0(0x18u);
    v81 = (char *)v80;
    if ( v80 )
    {
      *(_QWORD *)(v80 + 8) = v79;
      v79 = 0;
      *(_QWORD *)v80 = &unk_4A12538;
      *(_WORD *)(v80 + 16) = 0;
    }
    v120 = (char *)v80;
    if ( (_QWORD)v117 == *((_QWORD *)&v117 + 1) )
    {
      sub_235A6C0((unsigned __int64 *)&v116, (char *)v117, &v120);
      v81 = v120;
    }
    else
    {
      if ( (_QWORD)v117 )
      {
        *(_QWORD *)v117 = v80;
        *(_QWORD *)&v117 = v117 + 8;
        goto LABEL_114;
      }
      *(_QWORD *)&v117 = 8;
    }
    if ( v81 )
      (*(void (__fastcall **)(char *))(*(_QWORD *)v81 + 8LL))(v81);
LABEL_114:
    if ( v79 )
      (*(void (__fastcall **)(_QWORD *))(*v79 + 8LL))(v79);
    v82 = (__int64)v116;
    v83 = (_QWORD *)v117;
    v116 = 0;
    v84 = *((_QWORD *)&v117 + 1);
    v117 = 0u;
    v112 = v82;
    v85 = (_QWORD *)sub_22077B0(0x30u);
    v86 = v85;
    if ( v85 )
    {
      v85[2] = v83;
      v85[3] = v84;
      v85[4] = 0;
      *v85 = &unk_4A0C3B8;
      v85[5] = 0;
      v85[1] = v112;
    }
    else
    {
      if ( (_QWORD *)v112 != v83 )
      {
        v95 = (_QWORD *)v112;
        do
        {
          if ( *v95 )
            (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v95 + 8LL))(*v95);
          ++v95;
        }
        while ( v83 != v95 );
      }
      if ( v112 )
        j_j___libc_free_0(v112);
    }
    v87 = sub_22077B0(0x10u);
    v88 = (char *)v87;
    if ( v87 )
    {
      *(_QWORD *)(v87 + 8) = v86;
      v86 = 0;
      *(_QWORD *)v87 = &unk_4A0C3F8;
    }
    v120 = (char *)v87;
    v89 = (char *)v14[1];
    if ( v89 == (char *)v14[2] )
    {
      sub_2275C60(v14, v89, &v120);
      v88 = v120;
    }
    else
    {
      if ( v89 )
      {
        *(_QWORD *)v89 = v87;
        v14[1] += 8LL;
LABEL_123:
        if ( v86 )
          (*(void (__fastcall **)(_QWORD *))(*v86 + 8LL))(v86);
        v90 = (char *)v159;
        v91 = v158;
        if ( (char *)v159 != v158 )
        {
          do
          {
            if ( *(_QWORD *)v91 )
              (*(void (__fastcall **)(_QWORD))(**(_QWORD **)v91 + 8LL))(*(_QWORD *)v91);
            v91 += 8;
          }
          while ( v90 != v91 );
          v91 = v158;
        }
        if ( v91 )
        {
          v89 = (char *)(*((_QWORD *)&v159 + 1) - (_QWORD)v91);
          j_j___libc_free_0((unsigned __int64)v91);
        }
        v92 = (_QWORD *)v117;
        v93 = v116;
        if ( (_QWORD *)v117 != v116 )
        {
          do
          {
            if ( *v93 )
              (*(void (__fastcall **)(_QWORD, char *))(*(_QWORD *)*v93 + 8LL))(*v93, v89);
            ++v93;
          }
          while ( v92 != v93 );
          v93 = v116;
        }
        if ( v93 )
          j_j___libc_free_0((unsigned __int64)v93);
        return v14;
      }
      v14[1] = 8;
    }
    if ( v88 )
      (*(void (__fastcall **)(char *))(*(_QWORD *)v88 + 8LL))(v88);
    goto LABEL_123;
  }
  return v14;
}
