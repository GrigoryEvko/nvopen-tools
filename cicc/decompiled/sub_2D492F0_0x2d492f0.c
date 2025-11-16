// Function: sub_2D492F0
// Address: 0x2d492f0
//
__int64 __fastcall sub_2D492F0(__int64 *a1, __int64 a2)
{
  __int64 v3; // r15
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  _QWORD *v10; // rdi
  __int64 v11; // rax
  unsigned int v12; // edx
  char v13; // cl
  __int64 v14; // r13
  __int64 v15; // r14
  __int64 v16; // r15
  __int64 v17; // r14
  __int64 v18; // r12
  __int64 v19; // r13
  __int64 v20; // r12
  __int64 v21; // rbx
  __int64 v22; // rax
  char v23; // al
  _QWORD *v24; // rax
  __int64 v25; // r14
  unsigned int *v26; // r12
  unsigned int *v27; // rbx
  __int64 v28; // rdx
  unsigned int v29; // esi
  __int64 v30; // rbx
  __int64 v31; // r12
  _QWORD *v32; // rax
  __int64 v33; // r14
  __int64 v34; // rbx
  unsigned int *v35; // rbx
  unsigned int *v36; // r12
  __int64 v37; // rdx
  unsigned int v38; // esi
  __int64 v39; // rbx
  int v40; // eax
  int v41; // eax
  unsigned int v42; // edx
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // r14
  __int64 v47; // r15
  unsigned __int16 v48; // r9
  char v49; // dl
  __int16 v50; // r13
  _QWORD *v51; // rax
  __int64 v52; // r12
  unsigned int *v53; // r14
  unsigned int *v54; // r13
  __int64 v55; // rdx
  unsigned int v56; // esi
  __int16 v57; // ax
  _QWORD *v58; // rax
  __int64 v59; // r13
  unsigned int *v60; // r14
  unsigned int *v61; // r12
  __int64 v62; // rdx
  unsigned int v63; // esi
  __int64 v64; // r12
  __int64 v65; // r14
  _QWORD *v66; // r15
  _QWORD *v67; // rax
  __int64 v68; // r13
  unsigned int *v69; // r15
  unsigned int *v70; // r12
  __int64 v71; // rdx
  unsigned int v72; // esi
  int v73; // eax
  int v74; // eax
  unsigned int v75; // edx
  __int64 v76; // rax
  __int64 v77; // rdx
  __int64 v78; // rdx
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rax
  _QWORD *v83; // rax
  __int64 v84; // r9
  __int64 v85; // r13
  unsigned int *v86; // r14
  unsigned int *v87; // r12
  __int64 v88; // rdx
  unsigned int v89; // esi
  unsigned int *v90; // r13
  unsigned int *v91; // r12
  __int64 v92; // rdx
  unsigned int v93; // esi
  unsigned int *v94; // r15
  unsigned int *v95; // r12
  __int64 v96; // rdx
  unsigned int v97; // esi
  _QWORD **v98; // rdx
  int v99; // ecx
  int v100; // eax
  __int64 *v101; // rax
  __int64 v102; // rsi
  unsigned int *v103; // r13
  unsigned int *v104; // r12
  __int64 v105; // rdx
  unsigned int v106; // esi
  unsigned int *v107; // r13
  unsigned int *v108; // r12
  __int64 v109; // rdx
  unsigned int v110; // esi
  unsigned int *v111; // r13
  unsigned int *v112; // rbx
  __int64 v113; // rdx
  unsigned int v114; // esi
  unsigned int *v115; // r14
  unsigned int *v116; // rbx
  __int64 v117; // rdx
  unsigned int v118; // esi
  unsigned int *v119; // r12
  unsigned int *v120; // rbx
  __int64 v121; // rdx
  unsigned int v122; // esi
  char v123; // [rsp+8h] [rbp-278h]
  __int16 v124; // [rsp+Ch] [rbp-274h]
  char v125; // [rsp+10h] [rbp-270h]
  __int64 v126; // [rsp+10h] [rbp-270h]
  __int64 v127; // [rsp+20h] [rbp-260h]
  __int64 v128; // [rsp+20h] [rbp-260h]
  int v129; // [rsp+20h] [rbp-260h]
  _BYTE *v130; // [rsp+20h] [rbp-260h]
  __int64 v131; // [rsp+28h] [rbp-258h]
  __int64 v132; // [rsp+30h] [rbp-250h]
  __int64 v133; // [rsp+38h] [rbp-248h]
  _QWORD *v134; // [rsp+40h] [rbp-240h]
  _BYTE *v135; // [rsp+40h] [rbp-240h]
  __int64 v137; // [rsp+68h] [rbp-218h]
  __int64 v138[4]; // [rsp+70h] [rbp-210h] BYREF
  __int16 v139; // [rsp+90h] [rbp-1F0h]
  _DWORD v140[8]; // [rsp+A0h] [rbp-1E0h] BYREF
  __int16 v141; // [rsp+C0h] [rbp-1C0h]
  _BYTE v142[32]; // [rsp+D0h] [rbp-1B0h] BYREF
  __int16 v143; // [rsp+F0h] [rbp-190h]
  __int64 v144[3]; // [rsp+100h] [rbp-180h] BYREF
  __int64 v145; // [rsp+118h] [rbp-168h]
  unsigned __int8 v146; // [rsp+120h] [rbp-160h]
  char v147; // [rsp+121h] [rbp-15Fh]
  __int64 v148; // [rsp+128h] [rbp-158h]
  __int64 v149; // [rsp+138h] [rbp-148h]
  unsigned int *v150; // [rsp+140h] [rbp-140h] BYREF
  unsigned int v151; // [rsp+148h] [rbp-138h]
  char v152; // [rsp+150h] [rbp-130h] BYREF
  _QWORD *v153; // [rsp+170h] [rbp-110h]
  _QWORD *v154; // [rsp+178h] [rbp-108h]
  __int64 v155; // [rsp+180h] [rbp-100h]
  __int64 v156; // [rsp+188h] [rbp-F8h]
  __int64 v157; // [rsp+190h] [rbp-F0h]
  __int64 v158; // [rsp+198h] [rbp-E8h]
  void *v159; // [rsp+1C0h] [rbp-C0h]
  void *v160; // [rsp+1C8h] [rbp-B8h]
  _QWORD v161[12]; // [rsp+220h] [rbp-60h] BYREF

  v3 = *(_QWORD *)(a2 - 96);
  v4 = *(_QWORD *)(a2 - 32);
  v127 = *(_QWORD *)(a2 - 64);
  v134 = *(_QWORD **)(a2 + 40);
  v5 = v134[9];
  sub_2D46B10((__int64)&v150, a2, a1[1]);
  v144[0] = (__int64)"partword.cmpxchg.end";
  v6 = v156;
  v147 = 1;
  v146 = 3;
  v7 = sub_AA8550(v134, (__int64 *)(a2 + 24), 0, (__int64)v144, 0);
  v147 = 1;
  v132 = v7;
  v144[0] = (__int64)"partword.cmpxchg.failure";
  v146 = 3;
  v8 = sub_22077B0(0x50u);
  v133 = v8;
  if ( v8 )
    sub_AA4D50(v8, v6, (__int64)v144, v5, v132);
  v147 = 1;
  v144[0] = (__int64)"partword.cmpxchg.loop";
  v146 = 3;
  v9 = sub_22077B0(0x50u);
  v131 = v9;
  if ( v9 )
    sub_AA4D50(v9, v6, (__int64)v144, v5, v133);
  v137 = v134[6];
  v10 = (_QWORD *)((v137 & 0xFFFFFFFFFFFFFFF8LL) - 24);
  if ( (v137 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    v10 = 0;
  sub_B43D60(v10);
  v11 = *a1;
  v153 = v134;
  v12 = *(_DWORD *)(v11 + 96);
  v154 = v134 + 6;
  v13 = *(_BYTE *)(a2 + 3);
  LOWORD(v155) = 0;
  _BitScanReverse64((unsigned __int64 *)&v11, 1LL << v13);
  sub_2D44EF0(
    (__int64)v144,
    (__int64)&v150,
    a2,
    *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL),
    v3,
    63 - (v11 ^ 0x3F),
    v12 >> 3);
  v141 = 257;
  v139 = 257;
  v14 = v148;
  v15 = sub_A82F30(&v150, v4, v144[0], (__int64)v138, 0);
  v16 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v157 + 32LL))(
          v157,
          25,
          v15,
          v14,
          0,
          0);
  if ( !v16 )
  {
    v143 = 257;
    v16 = sub_B504D0(25, v15, v14, (__int64)v142, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _DWORD *, _QWORD *, __int64))(*(_QWORD *)v158 + 16LL))(
      v158,
      v16,
      v140,
      v154,
      v155);
    v111 = v150;
    v112 = &v150[4 * v151];
    if ( v150 != v112 )
    {
      do
      {
        v113 = *((_QWORD *)v111 + 1);
        v114 = *v111;
        v111 += 4;
        sub_B99FD0(v16, v114, v113);
      }
      while ( v112 != v111 );
    }
  }
  v139 = 257;
  v141 = 257;
  v17 = v148;
  v18 = sub_A82F30(&v150, v127, v144[0], (__int64)v138, 0);
  v19 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v157 + 32LL))(
          v157,
          25,
          v18,
          v17,
          0,
          0);
  if ( !v19 )
  {
    v143 = 257;
    v19 = sub_B504D0(25, v18, v17, (__int64)v142, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _DWORD *, _QWORD *, __int64))(*(_QWORD *)v158 + 16LL))(
      v158,
      v19,
      v140,
      v154,
      v155);
    v119 = v150;
    v120 = &v150[4 * v151];
    if ( v150 != v120 )
    {
      do
      {
        v121 = *((_QWORD *)v119 + 1);
        v122 = *v119;
        v119 += 4;
        sub_B99FD0(v19, v122, v121);
      }
      while ( v120 != v119 );
    }
  }
  v20 = v144[0];
  v141 = 257;
  v21 = v145;
  v22 = sub_AA4E30((__int64)v153);
  v23 = sub_AE5020(v22, v20);
  v143 = 257;
  v125 = v23;
  v24 = sub_BD2C40(80, 1u);
  v25 = (__int64)v24;
  if ( v24 )
    sub_B4D190((__int64)v24, v20, v21, (__int64)v142, 0, v125, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _DWORD *, _QWORD *, __int64))(*(_QWORD *)v158 + 16LL))(
    v158,
    v25,
    v140,
    v154,
    v155);
  v26 = v150;
  v27 = &v150[4 * v151];
  if ( v150 != v27 )
  {
    do
    {
      v28 = *((_QWORD *)v26 + 1);
      v29 = *v26;
      v26 += 4;
      sub_B99FD0(v25, v29, v28);
    }
    while ( v27 != v26 );
  }
  *(_WORD *)(v25 + 2) = *(_WORD *)(a2 + 2) & 1 | *(_WORD *)(v25 + 2) & 0xFFFE;
  v141 = 257;
  v30 = v149;
  v31 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v157 + 16LL))(v157, 28, v25, v149);
  if ( !v31 )
  {
    v143 = 257;
    v31 = sub_B504D0(28, v25, v30, (__int64)v142, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _DWORD *, _QWORD *, __int64))(*(_QWORD *)v158 + 16LL))(
      v158,
      v31,
      v140,
      v154,
      v155);
    v115 = v150;
    v116 = &v150[4 * v151];
    if ( v150 != v116 )
    {
      do
      {
        v117 = *((_QWORD *)v115 + 1);
        v118 = *v115;
        v115 += 4;
        sub_B99FD0(v31, v118, v117);
      }
      while ( v116 != v115 );
    }
  }
  v143 = 257;
  v32 = sub_BD2C40(72, 1u);
  v33 = (__int64)v32;
  if ( v32 )
    sub_B4C8F0((__int64)v32, v131, 1u, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, _QWORD *, __int64))(*(_QWORD *)v158 + 16LL))(
    v158,
    v33,
    v142,
    v154,
    v155);
  v34 = 4LL * v151;
  if ( v150 != &v150[v34] )
  {
    v128 = v31;
    v35 = &v150[v34];
    v36 = v150;
    do
    {
      v37 = *((_QWORD *)v36 + 1);
      v38 = *v36;
      v36 += 4;
      sub_B99FD0(v33, v38, v37);
    }
    while ( v35 != v36 );
    v31 = v128;
  }
  LOWORD(v155) = 0;
  v153 = (_QWORD *)v131;
  v154 = (_QWORD *)(v131 + 48);
  v143 = 257;
  v39 = sub_D5C860((__int64 *)&v150, v144[0], 2, (__int64)v142);
  v40 = *(_DWORD *)(v39 + 4) & 0x7FFFFFF;
  if ( v40 == *(_DWORD *)(v39 + 72) )
  {
    sub_B48D90(v39);
    v40 = *(_DWORD *)(v39 + 4) & 0x7FFFFFF;
  }
  v41 = (v40 + 1) & 0x7FFFFFF;
  v42 = v41 | *(_DWORD *)(v39 + 4) & 0xF8000000;
  v43 = *(_QWORD *)(v39 - 8) + 32LL * (unsigned int)(v41 - 1);
  *(_DWORD *)(v39 + 4) = v42;
  if ( *(_QWORD *)v43 )
  {
    v44 = *(_QWORD *)(v43 + 8);
    **(_QWORD **)(v43 + 16) = v44;
    if ( v44 )
      *(_QWORD *)(v44 + 16) = *(_QWORD *)(v43 + 16);
  }
  *(_QWORD *)v43 = v31;
  if ( v31 )
  {
    v45 = *(_QWORD *)(v31 + 16);
    *(_QWORD *)(v43 + 8) = v45;
    if ( v45 )
      *(_QWORD *)(v45 + 16) = v43 + 8;
    *(_QWORD *)(v43 + 16) = v31 + 16;
    *(_QWORD *)(v31 + 16) = v43;
  }
  *(_QWORD *)(*(_QWORD *)(v39 - 8) + 32LL * *(unsigned int *)(v39 + 72)
                                   + 8LL * ((*(_DWORD *)(v39 + 4) & 0x7FFFFFFu) - 1)) = v134;
  v141 = 257;
  v46 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v157 + 16LL))(v157, 29, v39, v16);
  if ( !v46 )
  {
    v143 = 257;
    v46 = sub_B504D0(29, v39, v16, (__int64)v142, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _DWORD *, _QWORD *, __int64))(*(_QWORD *)v158 + 16LL))(
      v158,
      v46,
      v140,
      v154,
      v155);
    v94 = v150;
    v95 = &v150[4 * v151];
    if ( v150 != v95 )
    {
      do
      {
        v96 = *((_QWORD *)v94 + 1);
        v97 = *v94;
        v94 += 4;
        sub_B99FD0(v46, v97, v96);
      }
      while ( v95 != v94 );
    }
  }
  v141 = 257;
  v47 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v157 + 16LL))(v157, 29, v39, v19);
  if ( !v47 )
  {
    v143 = 257;
    v47 = sub_B504D0(29, v39, v19, (__int64)v142, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _DWORD *, _QWORD *, __int64))(*(_QWORD *)v158 + 16LL))(
      v158,
      v47,
      v140,
      v154,
      v155);
    v90 = v150;
    v91 = &v150[4 * v151];
    if ( v150 != v91 )
    {
      do
      {
        v92 = *((_QWORD *)v90 + 1);
        v93 = *v90;
        v90 += 4;
        sub_B99FD0(v47, v93, v92);
      }
      while ( v91 != v90 );
    }
  }
  v48 = *(_WORD *)(a2 + 2);
  v49 = *(_BYTE *)(a2 + 72);
  v126 = v145;
  v143 = 257;
  v123 = v49;
  v129 = v146;
  v124 = (v48 >> 2) & 7;
  v50 = (unsigned __int8)v48 >> 5;
  v51 = sub_BD2C40(80, unk_3F148C4);
  v52 = (__int64)v51;
  if ( v51 )
    sub_B4D5A0((__int64)v51, v126, v47, v46, v129, v124, v50, v123, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, _QWORD *, __int64))(*(_QWORD *)v158 + 16LL))(
    v158,
    v52,
    v142,
    v154,
    v155);
  v53 = v150;
  v54 = &v150[4 * v151];
  if ( v150 != v54 )
  {
    do
    {
      v55 = *((_QWORD *)v53 + 1);
      v56 = *v53;
      v53 += 4;
      sub_B99FD0(v52, v56, v55);
    }
    while ( v54 != v53 );
  }
  v57 = *(_WORD *)(a2 + 2) & 1 | *(_WORD *)(v52 + 2) & 0xFFFE;
  *(_WORD *)(v52 + 2) = v57;
  *(_WORD *)(v52 + 2) = *(_WORD *)(a2 + 2) & 2 | v57 & 0xFFFD;
  v143 = 257;
  v140[0] = 0;
  v135 = (_BYTE *)sub_94D3D0(&v150, v52, (__int64)v140, 1, (__int64)v142);
  v143 = 257;
  v140[0] = 1;
  v130 = (_BYTE *)sub_94D3D0(&v150, v52, (__int64)v140, 1, (__int64)v142);
  if ( (*(_BYTE *)(a2 + 2) & 2) != 0 )
  {
    v143 = 257;
    v58 = sub_BD2C40(72, 1u);
    v59 = (__int64)v58;
    if ( v58 )
      sub_B4C8F0((__int64)v58, v132, 1u, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _BYTE *, _QWORD *, __int64))(*(_QWORD *)v158 + 16LL))(
      v158,
      v59,
      v142,
      v154,
      v155);
    v60 = v150;
    v61 = &v150[4 * v151];
    if ( v150 != v61 )
    {
      do
      {
        v62 = *((_QWORD *)v60 + 1);
        v63 = *v60;
        v60 += 4;
        sub_B99FD0(v59, v63, v62);
      }
      while ( v61 != v60 );
    }
  }
  else
  {
    v143 = 257;
    v83 = sub_BD2C40(72, 3u);
    v85 = (__int64)v83;
    if ( v83 )
      sub_B4C9A0((__int64)v83, v132, v133, (__int64)v130, 3u, v84, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _BYTE *, _QWORD *, __int64))(*(_QWORD *)v158 + 16LL))(
      v158,
      v85,
      v142,
      v154,
      v155);
    v86 = v150;
    v87 = &v150[4 * v151];
    if ( v150 != v87 )
    {
      do
      {
        v88 = *((_QWORD *)v86 + 1);
        v89 = *v86;
        v86 += 4;
        sub_B99FD0(v85, v89, v88);
      }
      while ( v87 != v86 );
    }
  }
  v141 = 257;
  v64 = v149;
  v153 = (_QWORD *)v133;
  v154 = (_QWORD *)(v133 + 48);
  LOWORD(v155) = 0;
  v65 = (*(__int64 (__fastcall **)(__int64, __int64, _BYTE *, __int64))(*(_QWORD *)v157 + 16LL))(v157, 28, v135, v149);
  if ( !v65 )
  {
    v143 = 257;
    v65 = sub_B504D0(28, (__int64)v135, v64, (__int64)v142, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _DWORD *, _QWORD *, __int64))(*(_QWORD *)v158 + 16LL))(
      v158,
      v65,
      v140,
      v154,
      v155);
    v107 = v150;
    v108 = &v150[4 * v151];
    if ( v150 != v108 )
    {
      do
      {
        v109 = *((_QWORD *)v107 + 1);
        v110 = *v107;
        v107 += 4;
        sub_B99FD0(v65, v110, v109);
      }
      while ( v108 != v107 );
    }
  }
  v141 = 257;
  v66 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v157 + 56LL))(
                    v157,
                    33,
                    v39,
                    v65);
  if ( !v66 )
  {
    v143 = 257;
    v66 = sub_BD2C40(72, unk_3F10FD0);
    if ( v66 )
    {
      v98 = *(_QWORD ***)(v39 + 8);
      v99 = *((unsigned __int8 *)v98 + 8);
      if ( (unsigned int)(v99 - 17) > 1 )
      {
        v102 = sub_BCB2A0(*v98);
      }
      else
      {
        v100 = *((_DWORD *)v98 + 8);
        BYTE4(v138[0]) = (_BYTE)v99 == 18;
        LODWORD(v138[0]) = v100;
        v101 = (__int64 *)sub_BCB2A0(*v98);
        v102 = sub_BCE1B0(v101, v138[0]);
      }
      sub_B523C0((__int64)v66, v102, 53, 33, v39, v65, (__int64)v142, 0, 0, 0);
    }
    (*(void (__fastcall **)(__int64, _QWORD *, _DWORD *, _QWORD *, __int64))(*(_QWORD *)v158 + 16LL))(
      v158,
      v66,
      v140,
      v154,
      v155);
    v103 = v150;
    v104 = &v150[4 * v151];
    if ( v150 != v104 )
    {
      do
      {
        v105 = *((_QWORD *)v103 + 1);
        v106 = *v103;
        v103 += 4;
        sub_B99FD0((__int64)v66, v106, v105);
      }
      while ( v104 != v103 );
    }
  }
  v143 = 257;
  v67 = sub_BD2C40(72, 3u);
  v68 = (__int64)v67;
  if ( v67 )
    sub_B4C9A0((__int64)v67, v131, v132, (__int64)v66, 3u, 0, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, _QWORD *, __int64))(*(_QWORD *)v158 + 16LL))(
    v158,
    v68,
    v142,
    v154,
    v155);
  v69 = v150;
  v70 = &v150[4 * v151];
  if ( v150 != v70 )
  {
    do
    {
      v71 = *((_QWORD *)v69 + 1);
      v72 = *v69;
      v69 += 4;
      sub_B99FD0(v68, v72, v71);
    }
    while ( v70 != v69 );
  }
  v73 = *(_DWORD *)(v39 + 4) & 0x7FFFFFF;
  if ( v73 == *(_DWORD *)(v39 + 72) )
  {
    sub_B48D90(v39);
    v73 = *(_DWORD *)(v39 + 4) & 0x7FFFFFF;
  }
  v74 = (v73 + 1) & 0x7FFFFFF;
  v75 = v74 | *(_DWORD *)(v39 + 4) & 0xF8000000;
  v76 = *(_QWORD *)(v39 - 8) + 32LL * (unsigned int)(v74 - 1);
  *(_DWORD *)(v39 + 4) = v75;
  if ( *(_QWORD *)v76 )
  {
    v77 = *(_QWORD *)(v76 + 8);
    **(_QWORD **)(v76 + 16) = v77;
    if ( v77 )
      *(_QWORD *)(v77 + 16) = *(_QWORD *)(v76 + 16);
  }
  *(_QWORD *)v76 = v65;
  if ( v65 )
  {
    v78 = *(_QWORD *)(v65 + 16);
    *(_QWORD *)(v76 + 8) = v78;
    if ( v78 )
      *(_QWORD *)(v78 + 16) = v76 + 8;
    *(_QWORD *)(v76 + 16) = v65 + 16;
    *(_QWORD *)(v65 + 16) = v76;
  }
  *(_QWORD *)(*(_QWORD *)(v39 - 8) + 32LL * *(unsigned int *)(v39 + 72)
                                   + 8LL * ((*(_DWORD *)(v39 + 4) & 0x7FFFFFFu) - 1)) = v133;
  sub_D5F1F0((__int64)&v150, a2);
  if ( v144[0] != v144[1] )
    v135 = (_BYTE *)sub_2D44750((__int64 *)&v150, (__int64)v135, v144);
  v79 = sub_ACADE0(*(__int64 ***)(a2 + 8));
  v143 = 257;
  v140[0] = 0;
  v80 = sub_2466140((__int64 *)&v150, v79, v135, v140, 1, (__int64)v142);
  v143 = 257;
  v140[0] = 1;
  v81 = sub_2466140((__int64 *)&v150, v80, v130, v140, 1, (__int64)v142);
  sub_BD84D0(a2, v81);
  sub_B43D60((_QWORD *)a2);
  sub_B32BF0(v161);
  v159 = &unk_49E5698;
  v160 = &unk_49D94D0;
  nullsub_63();
  nullsub_63();
  if ( v150 != (unsigned int *)&v152 )
    _libc_free((unsigned __int64)v150);
  return 1;
}
