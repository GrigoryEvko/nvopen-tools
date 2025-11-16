// Function: sub_2C5F9B0
// Address: 0x2c5f9b0
//
bool __fastcall sub_2C5F9B0(__int64 a1, __int64 a2)
{
  unsigned __int8 *v3; // rbx
  __int64 v4; // r14
  const void *v5; // r10
  unsigned __int64 v7; // r8
  char v8; // al
  __int64 v9; // r9
  unsigned __int8 *v10; // rax
  unsigned __int8 *v11; // rax
  __int64 v12; // r15
  __int64 v13; // r11
  __int64 v14; // rax
  bool result; // al
  signed int v16; // ecx
  size_t v17; // rax
  int *v18; // rdx
  unsigned __int8 *v19; // r10
  int *v20; // rcx
  __int64 v21; // rax
  signed int v22; // eax
  unsigned __int8 *v23; // rsi
  int v24; // r8d
  char v25; // al
  unsigned __int8 **v26; // r10
  __int64 v27; // r11
  __int64 v28; // r9
  __int64 **v29; // r9
  __int64 v30; // rax
  __int64 v31; // r8
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned __int8 **v34; // rsi
  unsigned __int64 v35; // rdx
  __int64 v36; // r15
  unsigned __int64 v37; // rcx
  __int64 v38; // rax
  int v39; // edx
  unsigned __int8 **v40; // r10
  __int64 v41; // r11
  __int64 v42; // rcx
  int v43; // r15d
  unsigned __int64 v44; // rax
  bool v45; // of
  unsigned __int64 v46; // rax
  int *v47; // rdi
  int *v48; // rax
  __int64 v49; // rsi
  __int64 *v50; // rdi
  __int64 v51; // r9
  __int64 v52; // rax
  int v53; // edx
  __int64 v54; // rsi
  __int64 v55; // rcx
  __int64 v56; // rcx
  unsigned int **v57; // rdi
  __int64 v58; // r13
  __int64 v59; // r14
  __int64 i; // rbx
  int *v61; // rdi
  __int64 v62; // rax
  int v63; // edx
  __int64 v64; // rdx
  unsigned __int8 *v65; // rax
  unsigned __int8 *v66; // rdx
  __int64 v67; // rdx
  __int64 v68; // r8
  __int64 v69; // r15
  int v70; // esi
  unsigned __int8 **v71; // r9
  unsigned __int64 v72; // rdx
  unsigned __int8 **v73; // rcx
  __int64 v74; // rax
  int v75; // edx
  __int64 v76; // r15
  __int64 v77; // rdx
  __int64 v78; // rax
  signed __int64 v79; // rdx
  __int64 v80; // [rsp-138h] [rbp-138h]
  char v81; // [rsp-130h] [rbp-130h]
  __int64 v82; // [rsp-130h] [rbp-130h]
  const void *v83; // [rsp-128h] [rbp-128h]
  __int64 v84; // [rsp-128h] [rbp-128h]
  __int64 v85; // [rsp-120h] [rbp-120h]
  int v86; // [rsp-120h] [rbp-120h]
  unsigned __int8 **v87; // [rsp-120h] [rbp-120h]
  unsigned __int8 *v88; // [rsp-120h] [rbp-120h]
  __int64 v89; // [rsp-118h] [rbp-118h]
  int v90; // [rsp-118h] [rbp-118h]
  __int64 v91; // [rsp-118h] [rbp-118h]
  char v92; // [rsp-118h] [rbp-118h]
  __int64 **v93; // [rsp-118h] [rbp-118h]
  unsigned __int8 *v94; // [rsp-110h] [rbp-110h]
  int v95; // [rsp-110h] [rbp-110h]
  __int64 v96; // [rsp-110h] [rbp-110h]
  int v97; // [rsp-108h] [rbp-108h]
  unsigned __int8 **v98; // [rsp-108h] [rbp-108h]
  __int64 v99; // [rsp-108h] [rbp-108h]
  __int64 v100; // [rsp-108h] [rbp-108h]
  __int64 v101; // [rsp-108h] [rbp-108h]
  __int64 v102; // [rsp-108h] [rbp-108h]
  unsigned __int8 **v103; // [rsp-108h] [rbp-108h]
  signed int v104; // [rsp-F4h] [rbp-F4h]
  unsigned __int8 *v105; // [rsp-F0h] [rbp-F0h]
  __int64 v106; // [rsp-E8h] [rbp-E8h]
  int v107; // [rsp-E8h] [rbp-E8h]
  unsigned __int8 **v108; // [rsp-E8h] [rbp-E8h]
  __int64 v109; // [rsp-E8h] [rbp-E8h]
  __int64 v110; // [rsp-E0h] [rbp-E0h]
  int v111; // [rsp-E0h] [rbp-E0h]
  int v112; // [rsp-E0h] [rbp-E0h]
  char v113; // [rsp-D8h] [rbp-D8h]
  int v114; // [rsp-D8h] [rbp-D8h]
  __int64 v115; // [rsp-D8h] [rbp-D8h]
  signed __int64 v116; // [rsp-D8h] [rbp-D8h]
  unsigned __int8 **v117; // [rsp-D8h] [rbp-D8h]
  unsigned __int8 *v118; // [rsp-D0h] [rbp-D0h]
  unsigned __int8 v119; // [rsp-D0h] [rbp-D0h]
  __int64 v120; // [rsp-D0h] [rbp-D0h]
  __int64 **v121; // [rsp-D0h] [rbp-D0h]
  char v122; // [rsp-D0h] [rbp-D0h]
  unsigned __int8 *v123; // [rsp-C8h] [rbp-C8h]
  __int64 v124; // [rsp-C8h] [rbp-C8h]
  unsigned __int8 **v125; // [rsp-C8h] [rbp-C8h]
  unsigned __int8 **v126; // [rsp-C8h] [rbp-C8h]
  unsigned __int8 *v127; // [rsp-C8h] [rbp-C8h]
  unsigned __int8 *v128; // [rsp-C0h] [rbp-C0h]
  bool v129; // [rsp-C0h] [rbp-C0h]
  __int64 v130; // [rsp-C0h] [rbp-C0h]
  __int64 v131; // [rsp-C0h] [rbp-C0h]
  __int64 v132; // [rsp-C0h] [rbp-C0h]
  int v133; // [rsp-C0h] [rbp-C0h]
  bool v134; // [rsp-C0h] [rbp-C0h]
  __int64 v135; // [rsp-C0h] [rbp-C0h]
  unsigned __int8 **v136; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v137; // [rsp-B0h] [rbp-B0h]
  _WORD v138[16]; // [rsp-A8h] [rbp-A8h] BYREF
  int *v139; // [rsp-88h] [rbp-88h] BYREF
  __int64 v140; // [rsp-80h] [rbp-80h]
  _BYTE v141[120]; // [rsp-78h] [rbp-78h] BYREF

  if ( *(_BYTE *)a2 != 92 )
    return 0;
  v3 = *(unsigned __int8 **)(a2 - 64);
  if ( !v3 )
    return 0;
  v4 = *(_QWORD *)(a2 - 32);
  if ( !v4 )
    return 0;
  v5 = *(const void **)(a2 + 72);
  v7 = *(unsigned int *)(a2 + 80);
  v8 = *(_BYTE *)v4;
  if ( *v3 == 92 && *((_QWORD *)v3 - 8) )
  {
    v123 = (unsigned __int8 *)*((_QWORD *)v3 - 4);
    if ( v123 )
    {
      v110 = *((_QWORD *)v3 + 9);
      if ( v8 != 92 )
      {
        v128 = (unsigned __int8 *)*((_QWORD *)v3 - 8);
        goto LABEL_15;
      }
      v128 = (unsigned __int8 *)*((_QWORD *)v3 - 8);
      v9 = 1;
      goto LABEL_7;
    }
    v128 = (unsigned __int8 *)*((_QWORD *)v3 - 8);
  }
  if ( v8 != 92 )
    return 0;
  v110 = 0;
  v9 = 0;
LABEL_7:
  v118 = *(unsigned __int8 **)(v4 - 64);
  if ( !v118 || (v105 = *(unsigned __int8 **)(v4 - 32)) == 0 )
  {
    if ( (_BYTE)v9 )
    {
LABEL_15:
      v118 = *(unsigned __int8 **)(a2 - 32);
      v9 = 1;
      v105 = v118;
      v113 = 0;
      v106 = 0;
      goto LABEL_16;
    }
    return 0;
  }
  v113 = 1;
  v106 = *(_QWORD *)(v4 + 72);
  v10 = v128;
  if ( !(_BYTE)v9 )
    v10 = *(unsigned __int8 **)(a2 - 64);
  v128 = v10;
  v11 = v123;
  if ( !(_BYTE)v9 )
    v11 = *(unsigned __int8 **)(a2 - 64);
  v123 = v11;
LABEL_16:
  v12 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)(v12 + 8) != 17 )
    v12 = 0;
  v13 = *((_QWORD *)v128 + 1);
  if ( *(_BYTE *)(v13 + 8) != 17 )
    return 0;
  v14 = *((_QWORD *)v3 + 1);
  if ( *(_BYTE *)(v14 + 8) != 17 || !v12 || *((_QWORD *)v118 + 1) != v13 )
    return 0;
  v16 = *(_DWORD *)(v13 + 32);
  v97 = *(_DWORD *)(v14 + 32);
  v139 = (int *)v141;
  v140 = 0x1000000000LL;
  v17 = 4 * v7;
  v104 = v16;
  if ( v7 > 0x10 )
  {
    v81 = v9;
    v83 = v5;
    v85 = 4 * v7;
    v91 = v13;
    v95 = v7;
    sub_C8D5F0((__int64)&v139, v141, v7, 4u, v7, v9);
    LODWORD(v7) = v95;
    v13 = v91;
    v17 = v85;
    v5 = v83;
    v61 = &v139[(unsigned int)v140];
    LOBYTE(v9) = v81;
  }
  else
  {
    v18 = (int *)v141;
    if ( !v17 )
      goto LABEL_27;
    v61 = (int *)v141;
  }
  v86 = v7;
  v92 = v9;
  v96 = v13;
  memcpy(v61, v5, v17);
  v18 = v139;
  LODWORD(v17) = v140;
  LODWORD(v7) = v86;
  LOBYTE(v9) = v92;
  v13 = v96;
LABEL_27:
  v19 = 0;
  v94 = 0;
  LODWORD(v140) = v7 + v17;
  v20 = &v18[(unsigned int)(v7 + v17)];
  if ( v20 == v18 )
  {
LABEL_114:
    result = sub_ACADE0((__int64 **)v12) != 0;
    goto LABEL_42;
  }
  do
  {
    v21 = *v18;
    if ( (int)v21 < 0 )
    {
      if ( (int)v21 < v97 )
        goto LABEL_45;
LABEL_94:
      LODWORD(v21) = v21 - v97;
      v23 = (unsigned __int8 *)v4;
      *v18 = v21;
      if ( v113 )
      {
        v22 = *(_DWORD *)(v106 + 4LL * (int)v21);
        if ( v22 >= v104 )
        {
          v23 = v105;
          v22 -= v104;
        }
        else
        {
          v23 = v118;
        }
LABEL_33:
        *v18 = v22;
        LODWORD(v21) = *v18;
      }
      if ( (_DWORD)v21 == -1 )
        goto LABEL_45;
      goto LABEL_35;
    }
    if ( (int)v21 >= v97 )
      goto LABEL_94;
    if ( (_BYTE)v9 )
    {
      v22 = *(_DWORD *)(v110 + 4 * v21);
      if ( v22 >= v104 )
      {
        v23 = v123;
        v22 -= v104;
      }
      else
      {
        v23 = v128;
      }
      goto LABEL_33;
    }
    v23 = v3;
LABEL_35:
    v24 = *v23;
    if ( v24 == 12 )
      goto LABEL_41;
    if ( v24 == 13 )
    {
      *v18 = -1;
    }
    else if ( v19 == v23 || !v19 )
    {
      v19 = v23;
    }
    else
    {
      if ( v94 && v23 != v94 )
        goto LABEL_41;
      v94 = v23;
      *v18 = v104 + v21;
    }
LABEL_45:
    ++v18;
  }
  while ( v20 != v18 );
  if ( !v19 )
    goto LABEL_114;
  if ( !v94 )
  {
    v122 = v9;
    v127 = v19;
    v135 = v13;
    v78 = sub_ACADE0((__int64 **)v13);
    LOBYTE(v9) = v122;
    v19 = v127;
    v94 = (unsigned __int8 *)v78;
    v13 = v135;
  }
  v119 = v9;
  v124 = v13;
  v130 = (__int64)v19;
  v25 = sub_B4ED80(v139, (unsigned int)v140, v104);
  v26 = (unsigned __int8 **)v130;
  v27 = v124;
  v28 = v119;
  if ( v25 )
  {
    v77 = v130;
    v134 = v25;
    sub_2C535E0(a1, (unsigned __int8 *)a2, v77);
    result = v134;
    goto LABEL_42;
  }
  if ( v119 )
  {
    v64 = 32LL * (*((_DWORD *)v3 + 1) & 0x7FFFFFF);
    if ( (v3[7] & 0x40) != 0 )
    {
      v65 = (unsigned __int8 *)*((_QWORD *)v3 - 1);
      v66 = &v65[v64];
    }
    else
    {
      v65 = &v3[-v64];
      v66 = v3;
    }
    v67 = v66 - v65;
    v132 = v67;
    v121 = *(__int64 ***)(a1 + 152);
    v68 = v67 >> 5;
    v69 = v67 >> 5;
    v112 = *(_DWORD *)(a1 + 192);
    v136 = (unsigned __int8 **)v138;
    v137 = 0x400000000LL;
    if ( (unsigned __int64)v67 > 0x80 )
    {
      v88 = v65;
      v103 = v26;
      v109 = v67 >> 5;
      sub_C8D5F0((__int64)&v136, v138, v67 >> 5, 8u, v68, v28);
      v65 = v88;
      v27 = v124;
      v26 = v103;
      LODWORD(v68) = v109;
    }
    v70 = v137;
    v71 = v136;
    v72 = 0;
    v73 = &v136[(unsigned int)v137];
    if ( v132 > 0 )
    {
      do
      {
        v73[v72 / 8] = *(unsigned __int8 **)&v65[4 * v72];
        v72 += 8LL;
        --v69;
      }
      while ( v69 );
      v70 = v137;
      v71 = v136;
    }
    LODWORD(v137) = v68 + v70;
    v101 = v27;
    v108 = v26;
    v74 = sub_DFCEF0(v121, v3, v71, (unsigned int)(v68 + v70), v112);
    v133 = v75;
    v26 = v108;
    v76 = v74;
    v27 = v101;
    if ( v136 != (unsigned __int8 **)v138 )
    {
      _libc_free((unsigned __int64)v136);
      v27 = v101;
      v26 = v108;
    }
    v120 = v76;
    v107 = v133;
  }
  else
  {
    v107 = 0;
    v120 = 0;
  }
  v111 = 0;
  v131 = 0;
  if ( v113 )
  {
    v100 = v27;
    v117 = v26;
    v62 = sub_F946A0(*(_QWORD *)(a1 + 152), v4, *(_DWORD *)(a1 + 192));
    v27 = v100;
    v26 = v117;
    v131 = v62;
    v111 = v63;
  }
  v29 = *(__int64 ***)(a1 + 152);
  v114 = *(_DWORD *)(a1 + 192);
  v30 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v31 = *(_QWORD *)(a2 - 8);
    v32 = v31 + v30;
  }
  else
  {
    v31 = a2 - v30;
    v32 = a2;
  }
  v33 = v32 - v31;
  v34 = (unsigned __int8 **)v138;
  v137 = 0x400000000LL;
  v35 = v33 >> 5;
  v136 = (unsigned __int8 **)v138;
  v36 = v33 >> 5;
  if ( (unsigned __int64)v33 > 0x80 )
  {
    v80 = v33;
    v82 = v31;
    v84 = v27;
    v87 = v26;
    v93 = v29;
    v102 = v33 >> 5;
    sub_C8D5F0((__int64)&v136, v138, v35, 8u, v31, (__int64)v29);
    v33 = v80;
    v31 = v82;
    v27 = v84;
    v26 = v87;
    v34 = &v136[(unsigned int)v137];
    v29 = v93;
    LODWORD(v35) = v102;
  }
  v37 = 0;
  if ( v33 > 0 )
  {
    do
    {
      v34[v37 / 8] = *(unsigned __int8 **)(v31 + 4 * v37);
      v37 += 8LL;
      --v36;
    }
    while ( v36 );
  }
  v89 = v27;
  v98 = v26;
  LODWORD(v137) = v137 + v35;
  v38 = sub_DFCEF0(v29, (unsigned __int8 *)a2, v136, (unsigned int)v137, v114);
  v40 = v98;
  v41 = v89;
  v42 = v38;
  if ( v136 != (unsigned __int8 **)v138 )
  {
    v90 = v39;
    v115 = v41;
    v125 = v98;
    v99 = v38;
    _libc_free((unsigned __int64)v136);
    v39 = v90;
    v42 = v99;
    v41 = v115;
    v40 = v125;
  }
  v43 = 1;
  if ( v111 != 1 )
    v43 = v107;
  v44 = v131 + v120;
  if ( __OFADD__(v131, v120) )
  {
    v44 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v131 <= 0 )
      v44 = 0x8000000000000000LL;
  }
  if ( v39 == 1 )
    v43 = 1;
  v45 = __OFADD__(v42, v44);
  v46 = v42 + v44;
  if ( v45 )
  {
    v46 = 0x8000000000000000LL;
    if ( v42 > 0 )
      v46 = 0x7FFFFFFFFFFFFFFFLL;
  }
  v116 = v46;
  v47 = &v139[(unsigned int)v140];
  if ( !((4LL * (unsigned int)v140) >> 4) )
  {
    v48 = v139;
LABEL_131:
    v79 = (char *)v47 - (char *)v48;
    if ( (char *)v47 - (char *)v48 != 8 )
    {
      if ( v79 != 12 )
      {
        v49 = 7;
        if ( v79 != 4 )
          goto LABEL_76;
        goto LABEL_134;
      }
      if ( v104 <= *v48 )
        goto LABEL_75;
      ++v48;
    }
    if ( v104 <= *v48 )
      goto LABEL_75;
    ++v48;
LABEL_134:
    v49 = 7;
    if ( v104 <= *v48 )
      goto LABEL_75;
    goto LABEL_76;
  }
  v48 = v139;
  while ( v104 > *v48 )
  {
    if ( v104 <= v48[1] )
    {
      ++v48;
      break;
    }
    if ( v104 <= v48[2] )
    {
      v48 += 2;
      break;
    }
    if ( v104 <= v48[3] )
    {
      v48 += 3;
      break;
    }
    v48 += 4;
    if ( &v139[4 * ((4LL * (unsigned int)v140) >> 4)] == v48 )
      goto LABEL_131;
  }
LABEL_75:
  v49 = (unsigned int)(v47 == v48) + 6;
LABEL_76:
  v50 = *(__int64 **)(a1 + 152);
  v137 = (__int64)v94;
  v51 = *(unsigned int *)(a1 + 192);
  v136 = v40;
  v126 = v40;
  v52 = sub_DFBC30(v50, v49, v41, (__int64)v139, (unsigned int)v140, v51, 0, 0, (__int64)&v136, 2, 0);
  v54 = *((_QWORD *)v3 + 2);
  v55 = v52;
  if ( !v54 || *(_QWORD *)(v54 + 8) )
  {
    if ( v107 == 1 )
      v53 = 1;
    v52 += v120;
    if ( __OFADD__(v120, v55) )
    {
      v52 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v120 <= 0 )
        v52 = 0x8000000000000000LL;
    }
  }
  v56 = *(_QWORD *)(v4 + 16);
  if ( v56 && !*(_QWORD *)(v56 + 8) )
    goto LABEL_84;
  if ( v111 == 1 )
    v53 = 1;
  v45 = __OFADD__(v131, v52);
  v52 += v131;
  if ( v45 )
  {
    if ( v131 > 0 )
    {
      v52 = 0x7FFFFFFFFFFFFFFFLL;
      goto LABEL_84;
    }
    if ( v53 != v43 )
      goto LABEL_112;
  }
  else
  {
LABEL_84:
    if ( v53 == v43 )
    {
      if ( v116 >= v52 )
        goto LABEL_86;
LABEL_41:
      result = 0;
      goto LABEL_42;
    }
LABEL_112:
    if ( v43 < v53 )
      goto LABEL_41;
  }
LABEL_86:
  v57 = (unsigned int **)(a1 + 8);
  v58 = a1 + 200;
  v138[8] = 257;
  v59 = sub_A83CB0(v57, v126, v94, (__int64)v139, (unsigned int)v140, (__int64)&v136);
  sub_BD84D0(a2, v59);
  if ( *(_BYTE *)v59 > 0x1Cu )
  {
    sub_BD6B90((unsigned __int8 *)v59, (unsigned __int8 *)a2);
    for ( i = *(_QWORD *)(v59 + 16); i; i = *(_QWORD *)(i + 8) )
      sub_F15FC0(v58, *(_QWORD *)(i + 24));
    if ( *(_BYTE *)v59 > 0x1Cu )
      sub_F15FC0(v58, v59);
  }
  result = 1;
  if ( *(_BYTE *)a2 > 0x1Cu )
  {
    sub_F15FC0(v58, a2);
    result = 1;
  }
LABEL_42:
  if ( v139 != (int *)v141 )
  {
    v129 = result;
    _libc_free((unsigned __int64)v139);
    return v129;
  }
  return result;
}
