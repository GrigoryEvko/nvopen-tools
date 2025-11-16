// Function: sub_2C5A430
// Address: 0x2c5a430
//
__int64 __fastcall sub_2C5A430(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 result; // rax
  __int64 v6; // rbx
  __int64 *v7; // rdx
  __int64 v8; // r14
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // r13
  __int64 v13; // r15
  __int64 v14; // r10
  __int64 *v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rdx
  char v18; // al
  int v19; // edx
  int v20; // eax
  __int64 *v21; // rdi
  __int64 v22; // r11
  __int64 v23; // rax
  __int64 v24; // rax
  int v25; // edx
  __int64 v26; // rax
  __int64 v27; // r11
  int v28; // ecx
  int v29; // edx
  unsigned __int64 v30; // rdx
  __int64 *v31; // rdi
  __int64 *v32; // rax
  __int64 v33; // rax
  int v34; // edx
  bool v35; // zf
  int v36; // edx
  __int64 *v37; // rdi
  __int64 v38; // rax
  __int64 *v39; // rdi
  __int64 v40; // r13
  int v41; // edx
  __int64 v42; // rcx
  int v43; // eax
  int v44; // edx
  bool v45; // of
  unsigned __int64 v46; // r13
  __int64 *v47; // rdi
  __int64 v48; // rax
  int v49; // edx
  int v50; // edx
  unsigned __int64 v51; // r13
  _QWORD *v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  int v55; // edx
  int v56; // edx
  __int64 v57; // r13
  __int64 v58; // rdi
  __int64 (__fastcall *v59)(__int64, __int64, __int64, _DWORD *, __int64); // rax
  __int64 v60; // rdi
  __int64 (__fastcall *v61)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  _BYTE *v62; // r14
  __int64 v63; // rdi
  __int64 (__fastcall *v64)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v65; // r10
  unsigned __int8 v66; // al
  int v67; // eax
  __int64 v68; // rax
  _BYTE *v69; // r10
  __int64 v70; // r15
  __int64 v71; // r13
  __int64 i; // rbx
  _QWORD *v73; // rax
  __int64 v74; // r14
  __int64 v75; // r13
  __int64 v76; // rbx
  __int64 v77; // rdx
  unsigned int v78; // esi
  _QWORD *v79; // rax
  _QWORD *v80; // r10
  void *v81; // rcx
  __int64 v82; // rax
  __int64 v83; // r15
  __int64 v84; // rbx
  __int64 v85; // r13
  __int64 v86; // rdx
  unsigned int v87; // esi
  _QWORD *v88; // rax
  __int64 v89; // rax
  __int64 v90; // rcx
  __int64 v91; // rbx
  __int64 v92; // r13
  __int64 v93; // rdx
  unsigned int v94; // esi
  __int64 v95; // rax
  bool v96; // cc
  unsigned __int64 v97; // rax
  signed __int64 v98; // [rsp-100h] [rbp-100h]
  _BYTE *v99; // [rsp-F8h] [rbp-F8h]
  _BYTE *v100; // [rsp-F0h] [rbp-F0h]
  _BYTE *v101; // [rsp-E8h] [rbp-E8h]
  __int64 v102; // [rsp-E0h] [rbp-E0h]
  int v103; // [rsp-E0h] [rbp-E0h]
  __int64 v104; // [rsp-D8h] [rbp-D8h]
  __int64 v105; // [rsp-D0h] [rbp-D0h]
  __int64 v106; // [rsp-D0h] [rbp-D0h]
  int v107; // [rsp-D0h] [rbp-D0h]
  int v108; // [rsp-D0h] [rbp-D0h]
  char v109; // [rsp-C8h] [rbp-C8h]
  int v110; // [rsp-C8h] [rbp-C8h]
  __int64 v111; // [rsp-C0h] [rbp-C0h]
  __int64 v112; // [rsp-C0h] [rbp-C0h]
  __int64 v113; // [rsp-B8h] [rbp-B8h]
  unsigned __int64 v114; // [rsp-B8h] [rbp-B8h]
  int v115; // [rsp-B8h] [rbp-B8h]
  int v116; // [rsp-B8h] [rbp-B8h]
  int v117; // [rsp-B8h] [rbp-B8h]
  _BYTE *v118; // [rsp-B8h] [rbp-B8h]
  __int64 v119; // [rsp-B0h] [rbp-B0h]
  __int64 v120; // [rsp-A8h] [rbp-A8h]
  __int64 v121; // [rsp-A8h] [rbp-A8h]
  __int64 v122; // [rsp-A8h] [rbp-A8h]
  _DWORD *v123; // [rsp-A0h] [rbp-A0h]
  _BYTE *v124; // [rsp-A0h] [rbp-A0h]
  _BYTE *v125; // [rsp-A0h] [rbp-A0h]
  _BYTE *v126; // [rsp-A0h] [rbp-A0h]
  _QWORD *v127; // [rsp-A0h] [rbp-A0h]
  __int64 v128; // [rsp-A0h] [rbp-A0h]
  __int64 v129; // [rsp-A0h] [rbp-A0h]
  _BYTE *v130; // [rsp-A0h] [rbp-A0h]
  __int64 v131[4]; // [rsp-98h] [rbp-98h] BYREF
  __int16 v132; // [rsp-78h] [rbp-78h]
  _BYTE *v133; // [rsp-68h] [rbp-68h] BYREF
  _BYTE *v134; // [rsp-60h] [rbp-60h]
  __int16 v135; // [rsp-48h] [rbp-48h]

  if ( *(_BYTE *)a2 != 92 )
    return 0;
  v3 = *(_QWORD *)(a2 - 64);
  v4 = *(_QWORD *)(v3 + 16);
  if ( !v4 || *(_QWORD *)(v4 + 8) || *(_BYTE *)v3 != 86 )
    return 0;
  v6 = a1;
  if ( (*(_BYTE *)(v3 + 7) & 0x40) != 0 )
  {
    v7 = *(__int64 **)(v3 - 8);
    v8 = *v7;
    if ( !*v7 )
      return 0;
  }
  else
  {
    v7 = (__int64 *)(v3 - 32LL * (*(_DWORD *)(v3 + 4) & 0x7FFFFFF));
    v8 = *v7;
    if ( !*v7 )
      return 0;
  }
  v104 = v7[4];
  if ( !v104 )
    return 0;
  v100 = (_BYTE *)v7[8];
  if ( !v100 )
    return 0;
  v9 = *(_QWORD *)(a2 - 32);
  v10 = *(_QWORD *)(v9 + 16);
  if ( !v10 || *(_QWORD *)(v10 + 8) || *(_BYTE *)v9 != 86 )
    return 0;
  if ( (*(_BYTE *)(v9 + 7) & 0x40) != 0 )
  {
    v11 = *(__int64 **)(v9 - 8);
    v120 = *v11;
    if ( !*v11 )
      return 0;
  }
  else
  {
    v11 = (__int64 *)(v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF));
    v120 = *v11;
    if ( !*v11 )
      return 0;
  }
  v101 = (_BYTE *)v11[4];
  if ( !v101 )
    return 0;
  v99 = (_BYTE *)v11[8];
  if ( !v99 )
    return 0;
  v12 = *(_QWORD *)(v8 + 8);
  v13 = *(unsigned int *)(a2 + 80);
  v123 = *(_DWORD **)(a2 + 72);
  if ( *(_BYTE *)(v12 + 8) != 17 )
    return 0;
  v14 = *(_QWORD *)(v120 + 8);
  if ( *(_BYTE *)(v14 + 8) != 17 || v12 != v14 )
    return 0;
  v15 = (*(_BYTE *)(a2 + 7) & 0x40) != 0
      ? *(__int64 **)(a2 - 8)
      : (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v119 = *v15;
  if ( (unsigned __int8)sub_920620(*v15) )
  {
    v109 = v119 == 0;
  }
  else
  {
    v119 = 0;
    v109 = 1;
  }
  v16 = (*(_BYTE *)(a2 + 7) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v105 = *(_QWORD *)(v16 + 32);
  if ( (unsigned __int8)sub_920620(v105) )
  {
    v17 = v105;
    v18 = v105 == 0;
  }
  else
  {
    v17 = 0;
    v18 = 1;
  }
  if ( v109 != v18 )
    return 0;
  if ( v119 )
  {
    v19 = *(_BYTE *)(v17 + 1) >> 1;
    v20 = *(_BYTE *)(v119 + 1) >> 1;
    if ( (unsigned __int8)v19 == 127 )
    {
      if ( v20 != 127 )
        return 0;
    }
    else if ( v20 == 127 || v19 != v20 )
    {
      return 0;
    }
  }
  v21 = *(__int64 **)(a1 + 152);
  v22 = *(_QWORD *)(v104 + 8);
  if ( *(_BYTE *)(v22 + 8) != 17 )
    v22 = 0;
  v23 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)(v23 + 8) != 17 )
    v23 = 0;
  v106 = v22;
  v111 = v23;
  v24 = sub_DFD2D0(v21, 57, v22);
  v110 = v25;
  v113 = v24;
  v26 = sub_DFD2D0(*(__int64 **)(v6 + 152), 57, v106);
  v27 = v106;
  v28 = 1;
  if ( v29 != 1 )
    v28 = v110;
  v30 = v26 + v113;
  v107 = v28;
  if ( __OFADD__(v26, v113) )
  {
    v30 = 0x8000000000000000LL;
    if ( v26 > 0 )
      v30 = 0x7FFFFFFFFFFFFFFFLL;
  }
  v114 = v30;
  v31 = *(__int64 **)(v6 + 152);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v32 = *(__int64 **)(a2 - 8);
  else
    v32 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v133 = (_BYTE *)*v32;
  v134 = (_BYTE *)v32[4];
  v102 = v27;
  v33 = sub_DFBC30(v31, 6, v27, (__int64)v123, v13, *(unsigned int *)(v6 + 192), 0, 0, (__int64)&v133, 2, a2);
  v35 = v34 == 1;
  v36 = 1;
  if ( !v35 )
    v36 = v107;
  v108 = v36;
  if ( __OFADD__(v33, v114) )
  {
    v96 = v33 <= 0;
    v97 = 0x8000000000000000LL;
    if ( !v96 )
      v97 = 0x7FFFFFFFFFFFFFFFLL;
    v98 = v97;
  }
  else
  {
    v98 = v33 + v114;
  }
  v133 = (_BYTE *)v8;
  v37 = *(__int64 **)(v6 + 152);
  v134 = (_BYTE *)v120;
  v38 = sub_DFBC30(v37, 6, v12, (__int64)v123, v13, *(unsigned int *)(v6 + 192), 0, 0, (__int64)&v133, 2, 0);
  v39 = *(__int64 **)(v6 + 152);
  v40 = v38;
  v115 = v41;
  v133 = (_BYTE *)v104;
  v134 = v101;
  v42 = sub_DFBC30(v39, 6, v102, (__int64)v123, v13, *(unsigned int *)(v6 + 192), 0, 0, (__int64)&v133, 2, 0);
  v43 = 1;
  if ( v44 != 1 )
    v43 = v115;
  v45 = __OFADD__(v42, v40);
  v46 = v42 + v40;
  v116 = v43;
  if ( v45 )
  {
    v46 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v42 <= 0 )
      v46 = 0x8000000000000000LL;
  }
  v47 = *(__int64 **)(v6 + 152);
  v133 = v100;
  v134 = v99;
  v48 = sub_DFBC30(v47, 6, v102, (__int64)v123, v13, *(unsigned int *)(v6 + 192), 0, 0, (__int64)&v133, 2, 0);
  v35 = v49 == 1;
  v50 = 1;
  if ( !v35 )
    v50 = v116;
  v45 = __OFADD__(v48, v46);
  v51 = v48 + v46;
  v117 = v50;
  if ( v45 )
  {
    v51 = 0x8000000000000000LL;
    if ( v48 > 0 )
      v51 = 0x7FFFFFFFFFFFFFFFLL;
  }
  v103 = *(_DWORD *)(v111 + 32);
  v52 = (_QWORD *)sub_BD5C60(a2);
  v53 = sub_BCB2A0(v52);
  BYTE4(v133) = 0;
  LODWORD(v133) = v103;
  if ( ((*(_BYTE *)(v53 + 8) - 7) & 0xFD) != 0 && v103 != 1 )
    sub_BCE1B0((__int64 *)v53, (__int64)v133);
  v54 = sub_DFD2D0(*(__int64 **)(v6 + 152), 57, v111);
  v35 = v55 == 1;
  v56 = 1;
  if ( !v35 )
    v56 = v117;
  v45 = __OFADD__(v54, v51);
  v57 = v54 + v51;
  if ( v45 )
  {
    if ( v54 <= 0 )
    {
      if ( v56 == v108 )
        goto LABEL_62;
      goto LABEL_99;
    }
    v57 = 0x7FFFFFFFFFFFFFFFLL;
  }
  if ( v56 == v108 )
  {
    if ( v57 <= v98 )
      goto LABEL_62;
    return 0;
  }
LABEL_99:
  if ( v108 < v56 )
    return 0;
LABEL_62:
  v58 = *(_QWORD *)(v6 + 88);
  v132 = 257;
  v112 = v6 + 8;
  v59 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _DWORD *, __int64))(*(_QWORD *)v58 + 112LL);
  if ( (char *)v59 != (char *)sub_9B6630 )
  {
    v118 = (_BYTE *)v59(v58, v8, v120, v123, v13);
LABEL_66:
    if ( v118 )
      goto LABEL_67;
    goto LABEL_106;
  }
  if ( *(_BYTE *)v8 <= 0x15u && *(_BYTE *)v120 <= 0x15u )
  {
    v118 = (_BYTE *)sub_AD5CE0(v8, v120, v123, v13, 0);
    goto LABEL_66;
  }
LABEL_106:
  v135 = 257;
  v73 = sub_BD2C40(112, unk_3F1FE60);
  v118 = v73;
  if ( v73 )
    sub_B4E9E0((__int64)v73, v8, v120, v123, v13, (__int64)&v133, 0, 0);
  (*(void (__fastcall **)(_QWORD, _BYTE *, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v6 + 96) + 16LL))(
    *(_QWORD *)(v6 + 96),
    v118,
    v131,
    *(_QWORD *)(v6 + 64),
    *(_QWORD *)(v6 + 72));
  v74 = *(_QWORD *)(v6 + 8);
  v75 = v74 + 16LL * *(unsigned int *)(v6 + 16);
  if ( v74 != v75 )
  {
    v121 = v6;
    v76 = *(_QWORD *)(v6 + 8);
    do
    {
      v77 = *(_QWORD *)(v76 + 8);
      v78 = *(_DWORD *)v76;
      v76 += 16;
      sub_B99FD0((__int64)v118, v78, v77);
    }
    while ( v75 != v76 );
    v6 = v121;
  }
LABEL_67:
  v60 = *(_QWORD *)(v6 + 88);
  v132 = 257;
  v61 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v60 + 112LL);
  if ( v61 != sub_9B6630 )
  {
    v62 = (_BYTE *)((__int64 (__fastcall *)(__int64, __int64, _BYTE *, _DWORD *, __int64))v61)(
                     v60,
                     v104,
                     v101,
                     v123,
                     v13);
LABEL_71:
    if ( v62 )
      goto LABEL_72;
    goto LABEL_118;
  }
  if ( *(_BYTE *)v104 <= 0x15u && *v101 <= 0x15u )
  {
    v62 = (_BYTE *)sub_AD5CE0(v104, (__int64)v101, v123, v13, 0);
    goto LABEL_71;
  }
LABEL_118:
  v135 = 257;
  v88 = sub_BD2C40(112, unk_3F1FE60);
  v62 = v88;
  if ( v88 )
    sub_B4E9E0((__int64)v88, v104, (__int64)v101, v123, v13, (__int64)&v133, 0, 0);
  (*(void (__fastcall **)(_QWORD, _BYTE *, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v6 + 96) + 16LL))(
    *(_QWORD *)(v6 + 96),
    v62,
    v131,
    *(_QWORD *)(v112 + 56),
    *(_QWORD *)(v112 + 64));
  v89 = *(_QWORD *)(v6 + 8);
  v90 = v89 + 16LL * *(unsigned int *)(v6 + 16);
  if ( v89 != v90 )
  {
    v122 = v6;
    v91 = *(_QWORD *)(v6 + 8);
    v92 = v90;
    do
    {
      v93 = *(_QWORD *)(v91 + 8);
      v94 = *(_DWORD *)v91;
      v91 += 16;
      sub_B99FD0((__int64)v62, v94, v93);
    }
    while ( v92 != v91 );
    v6 = v122;
  }
LABEL_72:
  v63 = *(_QWORD *)(v6 + 88);
  v132 = 257;
  v64 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v63 + 112LL);
  if ( v64 == sub_9B6630 )
  {
    if ( *v100 > 0x15u || *v99 > 0x15u )
      goto LABEL_112;
    v65 = sub_AD5CE0((__int64)v100, (__int64)v99, v123, v13, 0);
  }
  else
  {
    v65 = ((__int64 (__fastcall *)(__int64, _BYTE *, _BYTE *, _DWORD *, __int64))v64)(v63, v100, v99, v123, v13);
  }
  if ( !v65 )
  {
LABEL_112:
    v135 = 257;
    v79 = sub_BD2C40(112, unk_3F1FE60);
    v80 = v79;
    if ( v79 )
    {
      v81 = v123;
      v127 = v79;
      sub_B4E9E0((__int64)v79, (__int64)v100, (__int64)v99, v81, v13, (__int64)&v133, 0, 0);
      v80 = v127;
    }
    v128 = (__int64)v80;
    (*(void (__fastcall **)(_QWORD, _QWORD *, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v6 + 96) + 16LL))(
      *(_QWORD *)(v6 + 96),
      v80,
      v131,
      *(_QWORD *)(v112 + 56),
      *(_QWORD *)(v112 + 64));
    v82 = *(_QWORD *)(v6 + 8);
    v65 = v128;
    if ( v82 != v82 + 16LL * *(unsigned int *)(v6 + 16) )
    {
      v129 = v6;
      v83 = v82 + 16LL * *(unsigned int *)(v6 + 16);
      v84 = *(_QWORD *)(v6 + 8);
      v85 = v65;
      do
      {
        v86 = *(_QWORD *)(v84 + 8);
        v87 = *(_DWORD *)v84;
        v84 += 16;
        sub_B99FD0(v85, v87, v86);
      }
      while ( v83 != v84 );
      v6 = v129;
      v65 = v85;
    }
  }
  v135 = 257;
  if ( v119 )
  {
    v66 = *(_BYTE *)(v119 + 1);
    BYTE4(v131[0]) = 1;
    v124 = (_BYTE *)v65;
    v67 = v66 >> 1;
    if ( v67 == 127 )
      v67 = -1;
    LODWORD(v131[0]) = v67;
    v68 = sub_B36280((unsigned int **)v112, (__int64)v118, (__int64)v62, v65, v131[0], (__int64)&v133, 0);
    v69 = v124;
    v70 = v68;
  }
  else
  {
    v130 = (_BYTE *)v65;
    v95 = sub_B36550((unsigned int **)v112, (__int64)v118, (__int64)v62, v65, (__int64)&v133, 0);
    v69 = v130;
    v70 = v95;
  }
  v71 = v6 + 200;
  if ( *v118 > 0x1Cu )
  {
    v125 = v69;
    sub_F15FC0(v6 + 200, (__int64)v118);
    v69 = v125;
  }
  if ( *v62 > 0x1Cu )
  {
    v126 = v69;
    sub_F15FC0(v6 + 200, (__int64)v62);
    v69 = v126;
  }
  if ( *v69 > 0x1Cu )
    sub_F15FC0(v6 + 200, (__int64)v69);
  sub_BD84D0(a2, v70);
  if ( *(_BYTE *)v70 > 0x1Cu )
  {
    sub_BD6B90((unsigned __int8 *)v70, (unsigned __int8 *)a2);
    for ( i = *(_QWORD *)(v70 + 16); i; i = *(_QWORD *)(i + 8) )
      sub_F15FC0(v71, *(_QWORD *)(i + 24));
    if ( *(_BYTE *)v70 > 0x1Cu )
      sub_F15FC0(v71, v70);
  }
  result = 1;
  if ( *(_BYTE *)a2 > 0x1Cu )
  {
    sub_F15FC0(v71, a2);
    return 1;
  }
  return result;
}
