// Function: sub_2D7EEE0
// Address: 0x2d7eee0
//
__int64 __fastcall sub_2D7EEE0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // r8
  __int64 v7; // r9
  char v8; // r14
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 *v12; // r15
  __int64 *v13; // r14
  __int64 v14; // rax
  _BYTE *v15; // rdi
  __int64 (*v16)(); // rax
  _QWORD *v17; // r14
  unsigned __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rax
  unsigned __int64 v21; // r14
  __int64 v22; // r13
  unsigned __int64 v23; // r15
  unsigned __int64 v24; // r8
  unsigned __int64 v25; // rdi
  __int64 *v26; // r13
  __int64 *v27; // r14
  __int64 v28; // r12
  __int64 v29; // r12
  __int64 v30; // r15
  __int64 v31; // r14
  __int16 *v32; // rdx
  _QWORD *v33; // rax
  __int64 v34; // r15
  __int64 v35; // r14
  unsigned __int64 v36; // r14
  unsigned __int64 v37; // rbx
  __int64 v38; // rdx
  unsigned int v39; // esi
  __int64 v40; // rax
  unsigned __int64 v41; // rax
  const char *v42; // r15
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  const char *v46; // rax
  __int64 *v47; // r14
  __int64 v48; // rax
  unsigned __int64 v49; // rax
  __int64 v50; // rsi
  __int64 v51; // r12
  _QWORD **v52; // r14
  _QWORD **v53; // r15
  __int64 v54; // rbx
  _QWORD *v55; // rdi
  _QWORD **v56; // r15
  _QWORD **v57; // r14
  const char *v58; // r12
  __int64 v59; // rbx
  __int64 v60; // r12
  _QWORD *v61; // rdi
  unsigned __int8 *v62; // rax
  __int64 *v63; // r12
  __int64 *v64; // r14
  __int16 *v65; // rdx
  const char *v66; // rax
  __int64 v67; // r14
  __int64 v68; // rbx
  __int64 v69; // rax
  __int64 v70; // r12
  unsigned __int8 *v71; // r15
  __int64 v72; // rbx
  __int64 v73; // rax
  unsigned __int8 *v74; // rdi
  _BYTE *v75; // r15
  __int16 *v76; // rax
  __int16 *v77; // rdx
  __int64 v78; // rsi
  __int64 v79; // r15
  _BYTE *v80; // rbx
  __int16 *v81; // rax
  __int16 *v82; // rdx
  __int64 v83; // r9
  const char *v84; // rsi
  const char **v85; // r8
  char v86; // dl
  __int64 v87; // rsi
  unsigned __int8 *v88; // rsi
  __int64 v89; // rdi
  unsigned __int8 *v90; // rax
  unsigned int v91; // r15d
  __int64 v92; // [rsp-10h] [rbp-280h]
  __int64 v93; // [rsp-8h] [rbp-278h]
  __int64 v94; // [rsp+18h] [rbp-258h]
  bool v95; // [rsp+27h] [rbp-249h]
  const char **v96; // [rsp+28h] [rbp-248h]
  __int64 v97; // [rsp+30h] [rbp-240h]
  unsigned __int64 v98; // [rsp+30h] [rbp-240h]
  __int64 v99; // [rsp+38h] [rbp-238h]
  __int64 *v100; // [rsp+38h] [rbp-238h]
  __int64 v101; // [rsp+40h] [rbp-230h]
  __int64 v102; // [rsp+40h] [rbp-230h]
  unsigned __int64 v103; // [rsp+48h] [rbp-228h]
  unsigned __int8 *v104; // [rsp+50h] [rbp-220h]
  __int64 v105; // [rsp+58h] [rbp-218h]
  unsigned __int8 *v106; // [rsp+58h] [rbp-218h]
  __int64 v107; // [rsp+78h] [rbp-1F8h]
  unsigned __int64 v108; // [rsp+80h] [rbp-1F0h]
  __int64 v109; // [rsp+88h] [rbp-1E8h]
  __int64 *v110; // [rsp+88h] [rbp-1E8h]
  const char *v111; // [rsp+88h] [rbp-1E8h]
  __int64 v112; // [rsp+88h] [rbp-1E8h]
  __int64 *v113; // [rsp+88h] [rbp-1E8h]
  __int64 *v114; // [rsp+B0h] [rbp-1C0h] BYREF
  __int64 v115; // [rsp+B8h] [rbp-1B8h]
  _QWORD v116[2]; // [rsp+C0h] [rbp-1B0h] BYREF
  char *v117; // [rsp+D0h] [rbp-1A0h] BYREF
  __int16 *v118; // [rsp+D8h] [rbp-198h]
  const char *v119; // [rsp+E0h] [rbp-190h]
  __int16 v120; // [rsp+F0h] [rbp-180h]
  const char *v121; // [rsp+100h] [rbp-170h] BYREF
  __int16 *v122; // [rsp+108h] [rbp-168h]
  __int64 v123; // [rsp+110h] [rbp-160h]
  int v124; // [rsp+118h] [rbp-158h]
  char v125; // [rsp+11Ch] [rbp-154h]
  __int16 v126; // [rsp+120h] [rbp-150h] BYREF
  _BYTE *v127; // [rsp+130h] [rbp-140h] BYREF
  __int64 v128; // [rsp+138h] [rbp-138h]
  _BYTE v129[48]; // [rsp+140h] [rbp-130h] BYREF
  _BYTE *v130; // [rsp+170h] [rbp-100h] BYREF
  __int64 v131; // [rsp+178h] [rbp-F8h]
  _BYTE v132[48]; // [rsp+180h] [rbp-F0h] BYREF
  unsigned __int64 v133; // [rsp+1B0h] [rbp-C0h] BYREF
  unsigned int v134; // [rsp+1B8h] [rbp-B8h]
  char v135; // [rsp+1C3h] [rbp-ADh]
  __int64 *v136; // [rsp+1E0h] [rbp-90h]
  __int64 v137; // [rsp+1E8h] [rbp-88h]
  __int64 v138; // [rsp+1F0h] [rbp-80h] BYREF
  __int64 *v139; // [rsp+200h] [rbp-70h]
  __int64 v140; // [rsp+208h] [rbp-68h]
  __int64 v141; // [rsp+210h] [rbp-60h] BYREF

  v2 = (unsigned __int8)byte_50181A8;
  if ( byte_50181A8 )
    return 0;
  v4 = a1;
  v5 = a2;
  sub_2FEFA40(&v133);
  v8 = v135;
  if ( v139 != &v141 )
    j_j___libc_free_0((unsigned __int64)v139);
  if ( v136 != &v138 )
    j_j___libc_free_0((unsigned __int64)v136);
  if ( !v8 )
    return 0;
  v116[0] = a2;
  v114 = v116;
  v115 = 0x200000001LL;
  if ( !a2 )
    BUG();
  v9 = *(_QWORD *)(a2 + 32);
  if ( v9 == *(_QWORD *)(a2 + 40) + 48LL )
  {
    v109 = a2;
    *(_QWORD *)(a1 + 88) = v9;
    *(_WORD *)(a1 + 96) = 0;
  }
  else
  {
    do
    {
      if ( !v9 )
        BUG();
      v10 = (unsigned int)v115;
      if ( *(_BYTE *)(v9 - 24) != 86 || *(_QWORD *)(a2 - 96) != *(_QWORD *)(v9 - 120) )
        break;
      if ( (unsigned __int64)(unsigned int)v115 + 1 > HIDWORD(v115) )
      {
        sub_C8D5F0((__int64)&v114, v116, (unsigned int)v115 + 1LL, 8u, v6, v7);
        v10 = (unsigned int)v115;
      }
      v114[v10] = v9 - 24;
      v11 = *(_QWORD *)(a2 + 40);
      v10 = (unsigned int)(v115 + 1);
      LODWORD(v115) = v115 + 1;
      v9 = *(_QWORD *)(v9 + 8);
    }
    while ( v9 != v11 + 48 );
    v12 = v114 + 1;
    v13 = &v114[v10];
    v109 = *(v13 - 1);
    *(_QWORD *)(a1 + 88) = *(_QWORD *)(v109 + 32);
    *(_WORD *)(a1 + 96) = 0;
    while ( v13 != v12 )
    {
      v14 = *v12++;
      sub_2D7E760(a1, *(_QWORD *)(v14 + 64));
    }
  }
  v95 = sub_BCAC40(*(_QWORD *)(*(_QWORD *)(a2 - 96) + 8LL), 1);
  if ( !v95 || (*(_BYTE *)(a2 + 7) & 0x20) != 0 && sub_B91C10(a2, 15) )
    goto LABEL_22;
  v15 = *(_BYTE **)(a1 + 16);
  v16 = *(__int64 (**)())(*(_QWORD *)v15 + 104LL);
  if ( v16 == sub_2D56590 )
    goto LABEL_25;
  if ( ((unsigned __int8 (__fastcall *)(_BYTE *, bool))v16)(
         v15,
         (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17 <= 1) )
  {
    v15 = *(_BYTE **)(v4 + 16);
LABEL_25:
    if ( !v15[537004] )
      goto LABEL_22;
    v17 = *(_QWORD **)(v4 + 32);
    if ( !(unsigned __int8)sub_BC8C50(a2, &v130, &v133) )
      goto LABEL_153;
    v18 = v133;
    if ( !&v130[v133] )
      goto LABEL_153;
    if ( (unsigned __int64)v130 >= v133 )
      v18 = (unsigned __int64)v130;
    v91 = sub_F02DD0(v18, (unsigned __int64)&v130[v133]);
    if ( (unsigned int)sub_DF95A0(v17) >= v91 )
    {
LABEL_153:
      v19 = *(_QWORD *)(a2 - 96);
      if ( (unsigned __int8)(*(_BYTE *)v19 - 82) > 1u )
        goto LABEL_22;
      v20 = *(_QWORD *)(v19 + 16);
      if ( !v20
        || *(_QWORD *)(v20 + 8)
        || !(unsigned __int8)sub_2D56C30(v17, *(_QWORD *)(a2 - 64))
        && !(unsigned __int8)sub_2D56C30(v17, *(_QWORD *)(a2 - 32)) )
      {
        goto LABEL_22;
      }
    }
    if ( sub_11F3070(*(_QWORD *)(a2 + 40), *(_QWORD *)(v4 + 80), *(__int64 **)(v4 + 64)) )
      goto LABEL_22;
  }
  v21 = *(_QWORD *)(v4 + 824);
  *(_QWORD *)(v4 + 824) = 0;
  if ( v21 )
  {
    v22 = *(_QWORD *)(v21 + 24);
    v23 = v22 + 8LL * *(unsigned int *)(v21 + 32);
    if ( v22 != v23 )
    {
      do
      {
        v24 = *(_QWORD *)(v23 - 8);
        v23 -= 8LL;
        if ( v24 )
        {
          v25 = *(_QWORD *)(v24 + 24);
          if ( v25 != v24 + 40 )
          {
            v108 = v24;
            _libc_free(v25);
            v24 = v108;
          }
          j_j___libc_free_0(v24);
        }
      }
      while ( v22 != v23 );
      v23 = *(_QWORD *)(v21 + 24);
    }
    if ( v23 != v21 + 40 )
      _libc_free(v23);
    if ( *(_QWORD *)v21 != v21 + 16 )
      _libc_free(*(_QWORD *)v21);
    j_j___libc_free_0(v21);
  }
  v26 = v114;
  v127 = v129;
  v128 = 0x600000000LL;
  v131 = 0x600000000LL;
  v130 = v132;
  if ( &v114[(unsigned int)v115] != v114 )
  {
    v27 = &v114[(unsigned int)v115];
    do
    {
      v29 = *v26;
      v30 = *(_QWORD *)(*v26 - 64);
      if ( (unsigned __int8)sub_2D56C30(*(_QWORD **)(v4 + 32), v30) )
        sub_9C95B0((__int64)&v127, v30);
      v28 = *(_QWORD *)(v29 - 32);
      if ( (unsigned __int8)sub_2D56C30(*(_QWORD **)(v4 + 32), v28) )
        sub_9C95B0((__int64)&v130, v28);
      ++v26;
    }
    while ( v27 != v26 );
    v5 = a2;
  }
  v94 = *(_QWORD *)(v5 + 40);
  v110 = *(__int64 **)(v109 + 32);
  sub_23D0AB0((__int64)&v133, v5, 0, 0, 0);
  v117 = (char *)sub_BD5D20(v5);
  v31 = *(_QWORD *)(v5 - 96);
  v119 = ".frozen";
  v120 = 773;
  v118 = v32;
  v126 = 257;
  v33 = sub_BD2C40(72, 1u);
  v34 = (__int64)v33;
  if ( v33 )
    sub_B549F0((__int64)v33, v31, (__int64)&v121, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, char **, __int64, __int64))(*(_QWORD *)v140 + 16LL))(
    v140,
    v34,
    &v117,
    v137,
    v138);
  v35 = 16LL * v134;
  if ( v133 != v133 + v35 )
  {
    v105 = v4;
    v36 = v133 + v35;
    v37 = v133;
    do
    {
      v38 = *(_QWORD *)(v37 + 8);
      v39 = *(_DWORD *)v37;
      v37 += 16LL;
      sub_B99FD0(v34, v39, v38);
    }
    while ( v36 != v37 );
    v4 = v105;
  }
  v40 = *(_QWORD *)(v4 + 56);
  if ( (_DWORD)v128 )
  {
    if ( (_DWORD)v131 )
    {
      v93 = *(_QWORD *)(v4 + 56);
      v117 = 0;
      v121 = 0;
      sub_F38330(v34, v110, 1, (unsigned __int64 *)&v117, (unsigned __int64 *)&v121, 0, 0, v93);
      v42 = v117;
      v111 = v121;
      v104 = (unsigned __int8 *)*((_QWORD *)v121 + 5);
      v106 = (unsigned __int8 *)*((_QWORD *)v117 + 5);
      v103 = *(_QWORD *)&v117[-32 * (*((_DWORD *)v117 + 1) & 0x7FFFFFF)];
    }
    else
    {
      v41 = sub_F38250(v34, v110, 1, 0, 0, 0, v40, 0);
      v111 = 0;
      v104 = 0;
      v42 = (const char *)v41;
      v106 = *(unsigned __int8 **)(v41 + 40);
      v103 = *(_QWORD *)(v41 - 32LL * (*(_DWORD *)(v41 + 4) & 0x7FFFFFF));
    }
    v121 = "select.end";
    v126 = 259;
    sub_BD6B50((unsigned __int8 *)v103, &v121);
    if ( v106 )
    {
      v121 = "select.true.sink";
      v126 = 259;
      sub_BD6B50(v106, &v121);
    }
  }
  else
  {
    v89 = v34;
    v42 = 0;
    v111 = (const char *)sub_F382C0(v89, v110, 1, 0, 0, 0, v40, 0);
    v104 = (unsigned __int8 *)*((_QWORD *)v111 + 5);
    v90 = *(unsigned __int8 **)&v111[-32 * (*((_DWORD *)v111 + 1) & 0x7FFFFFF)];
    v121 = "select.end";
    v103 = (unsigned __int64)v90;
    v126 = 259;
    sub_BD6B50(v90, &v121);
    v43 = v92;
    v106 = 0;
  }
  if ( v104 )
  {
    v46 = "select.false";
    v126 = 259;
    if ( (_DWORD)v131 )
      v46 = "select.false.sink";
    v121 = v46;
    sub_BD6B50(v104, &v121);
  }
  if ( *(_BYTE *)(v4 + 832) )
  {
    if ( v106 )
      sub_D695C0((__int64)&v121, v4 + 840, (__int64 *)v106, v43, v44, v45);
    if ( v104 )
      sub_D695C0((__int64)&v121, v4 + 840, (__int64 *)v104, v43, v44, v45);
    sub_D695C0((__int64)&v121, v4 + 840, (__int64 *)v103, v43, v44, v45);
  }
  v47 = *(__int64 **)(v4 + 64);
  v48 = sub_FDD860(v47, v94);
  sub_FE1040(v47, v103, v48);
  v49 = sub_986580(v94);
  v50 = v5;
  v51 = (__int64)(v42 + 24);
  sub_B47C00(v49, v50, dword_444DF90, 4);
  v52 = (_QWORD **)&v127[8 * (unsigned int)v128];
  v53 = (_QWORD **)v127;
  if ( v52 != (_QWORD **)v127 )
  {
    v101 = v4;
    v54 = v97;
    do
    {
      v55 = *v53;
      LOWORD(v54) = 0;
      v50 = v51;
      ++v53;
      sub_B444E0(v55, v51, v54);
    }
    while ( v52 != v53 );
    v4 = v101;
  }
  v56 = (_QWORD **)v130;
  v57 = (_QWORD **)&v130[8 * (unsigned int)v131];
  if ( v57 != (_QWORD **)v130 )
  {
    v58 = v111;
    v112 = v4;
    v59 = v99;
    v60 = (__int64)(v58 + 24);
    do
    {
      v61 = *v56;
      LOWORD(v59) = 0;
      v50 = v60;
      ++v56;
      sub_B444E0(v61, v60, v59);
    }
    while ( v57 != v56 );
    v4 = v112;
  }
  if ( v106 )
  {
    v62 = v104;
    if ( !v104 )
      v62 = (unsigned __int8 *)v94;
    v104 = v62;
  }
  else
  {
    v106 = (unsigned __int8 *)v94;
  }
  v63 = v114;
  v121 = 0;
  v122 = &v126;
  v125 = 1;
  v123 = 2;
  v64 = &v114[(unsigned int)v115];
  v124 = 0;
  if ( v114 == v64 )
  {
    *(_WORD *)(v4 + 96) = 0;
    *(_QWORD *)(v4 + 88) = v94 + 48;
    goto LABEL_123;
  }
  do
  {
    v50 = *v63;
    v66 = (const char *)sub_AE6EC0((__int64)&v121, *v63);
    if ( v125 )
      v65 = &v122[4 * HIDWORD(v123)];
    else
      v65 = &v122[4 * (unsigned int)v123];
    ++v63;
    v117 = (char *)v66;
    v118 = v65;
    sub_254BBF0((__int64)&v117);
  }
  while ( v64 != v63 );
  v98 = (unsigned __int64)v114;
  v113 = &v114[(unsigned int)v115];
  if ( v114 == v113 )
    goto LABEL_121;
  v102 = v4;
  v100 = (__int64 *)(v4 + 840);
  do
  {
    v67 = *(v113 - 1);
    v120 = 257;
    v68 = *(_QWORD *)(v67 + 8);
    v69 = sub_BD2DA0(80);
    v70 = v69;
    if ( v69 )
    {
      v71 = (unsigned __int8 *)v69;
      sub_B44260(v69, v68, 55, 0x8000000u, 0, 0);
      *(_DWORD *)(v70 + 72) = 2;
      sub_BD6B50((unsigned __int8 *)v70, (const char **)&v117);
      sub_BD2A10(v70, *(_DWORD *)(v70 + 72), 1);
    }
    else
    {
      v71 = 0;
    }
    v72 = v67;
    v73 = v107;
    LOWORD(v73) = 1;
    v107 = v73;
    sub_B44220(v71, *(_QWORD *)(v103 + 56), v73);
    v74 = v71;
    v75 = 0;
    sub_BD6B90(v74, (unsigned __int8 *)v67);
    while ( !v125 )
    {
      if ( !sub_C8CA60((__int64)&v121, v72) )
        goto LABEL_108;
      v75 = *(_BYTE **)(v72 - 64);
      if ( *v75 != 86 )
        goto LABEL_108;
LABEL_133:
      v72 = (__int64)v75;
    }
    v76 = v122;
    v77 = &v122[4 * HIDWORD(v123)];
    if ( v122 != v77 )
    {
      while ( *(_QWORD *)v76 != v72 )
      {
        v76 += 4;
        if ( v77 == v76 )
          goto LABEL_108;
      }
      v75 = *(_BYTE **)(v72 - 64);
      if ( *v75 == 86 )
        goto LABEL_133;
    }
LABEL_108:
    v78 = (__int64)v75;
    v79 = v67;
    v80 = 0;
    sub_F0A850(v70, v78, (__int64)v106);
    while ( !v125 )
    {
      if ( !sub_C8CA60((__int64)&v121, v79) )
        goto LABEL_115;
      v80 = *(_BYTE **)(v79 - 32);
      if ( *v80 != 86 )
        goto LABEL_115;
LABEL_130:
      v79 = (__int64)v80;
    }
    v81 = v122;
    v82 = &v122[4 * HIDWORD(v123)];
    if ( v122 != v82 )
    {
      while ( *(_QWORD *)v81 != v79 )
      {
        v81 += 4;
        if ( v82 == v81 )
          goto LABEL_115;
      }
      v80 = *(_BYTE **)(v79 - 32);
      if ( *v80 == 86 )
        goto LABEL_130;
    }
LABEL_115:
    sub_F0A850(v70, (__int64)v80, (__int64)v104);
    v84 = *(const char **)(v67 + 48);
    v85 = (const char **)(v70 + 48);
    v117 = (char *)v84;
    if ( !v84 )
    {
      if ( v85 == (const char **)&v117 )
        goto LABEL_119;
      v87 = *(_QWORD *)(v70 + 48);
      if ( !v87 )
        goto LABEL_119;
LABEL_136:
      v96 = v85;
      sub_B91220((__int64)v85, v87);
      v85 = v96;
      goto LABEL_137;
    }
    sub_B96E90((__int64)&v117, (__int64)v84, 1);
    v85 = (const char **)(v70 + 48);
    if ( (char **)(v70 + 48) == &v117 )
    {
      if ( v117 )
        sub_B91220((__int64)&v117, (__int64)v117);
      goto LABEL_119;
    }
    v87 = *(_QWORD *)(v70 + 48);
    if ( v87 )
      goto LABEL_136;
LABEL_137:
    v88 = (unsigned __int8 *)v117;
    *(_QWORD *)(v70 + 48) = v117;
    if ( v88 )
      sub_B976B0((__int64)&v117, v88, (__int64)v85);
LABEL_119:
    sub_2D594F0(v67, v70, v100, *(unsigned __int8 *)(v102 + 832), (__int64)v85, v83);
    sub_B43D60((_QWORD *)v67);
    v50 = v67;
    sub_25DDDB0((__int64)&v121, v67);
    --v113;
  }
  while ( (__int64 *)v98 != v113 );
  v4 = v102;
LABEL_121:
  v86 = v125;
  *(_WORD *)(v4 + 96) = 0;
  *(_QWORD *)(v4 + 88) = v94 + 48;
  if ( !v86 )
    _libc_free((unsigned __int64)v122);
LABEL_123:
  sub_F94A20(&v133, v50);
  if ( v130 != v132 )
    _libc_free((unsigned __int64)v130);
  if ( v127 != v129 )
    _libc_free((unsigned __int64)v127);
  v2 = v95;
LABEL_22:
  if ( v114 != v116 )
    _libc_free((unsigned __int64)v114);
  return v2;
}
