// Function: sub_23A95C0
// Address: 0x23a95c0
//
__int64 __fastcall sub_23A95C0(__int64 a1, unsigned __int64 a2, unsigned __int64 *a3, char a4)
{
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // r9
  _QWORD *v10; // rax
  char v11; // r15
  char v12; // r15
  __int64 v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rbx
  __int64 v25; // rax
  _QWORD *v26; // rax
  _QWORD *v27; // rax
  _QWORD *v28; // rax
  _QWORD *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  _QWORD *v35; // rax
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // r15
  __int64 v41; // rdx
  __int64 v42; // rdx
  int v43; // esi
  __int64 v44; // rdx
  __int64 v45; // rcx
  _BYTE *v46; // r15
  _BYTE *v47; // rbx
  _BYTE *v48; // r15
  unsigned __int64 v49; // rdi
  __int64 *v50; // r15
  __int64 *v51; // rbx
  __int64 *v52; // r15
  unsigned __int64 v53; // rdi
  _QWORD *v54; // rax
  _QWORD *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // r9
  __int64 v60; // r9
  char v61; // al
  char v62; // r15
  char v63; // r13
  __int64 v64; // rax
  _QWORD *v65; // rax
  _QWORD *v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // r8
  __int64 v70; // r9
  __int64 v71; // rax
  __int64 v72; // rdx
  __int64 v73; // rcx
  __int64 v74; // r8
  __int64 v75; // r9
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // r9
  __int64 v79; // rdx
  __int64 v80; // rcx
  __int64 v81; // r8
  __int64 v82; // r9
  __int64 v83; // rax
  _QWORD *v84; // rax
  __int64 v85; // rdx
  unsigned __int64 v86; // rdx
  __int64 v87; // rdx
  unsigned __int64 v88; // [rsp+8h] [rbp-AB8h]
  __int64 v89; // [rsp+10h] [rbp-AB0h]
  __int64 v90; // [rsp+18h] [rbp-AA8h]
  __int64 v91; // [rsp+20h] [rbp-AA0h]
  __int64 v92; // [rsp+28h] [rbp-A98h]
  __int64 v93; // [rsp+30h] [rbp-A90h]
  __int64 v95; // [rsp+38h] [rbp-A88h]
  __int64 v96; // [rsp+38h] [rbp-A88h]
  __int64 v97; // [rsp+40h] [rbp-A80h]
  char v98; // [rsp+40h] [rbp-A80h]
  __int64 v99; // [rsp+48h] [rbp-A78h]
  unsigned int v100; // [rsp+50h] [rbp-A70h]
  __int16 v101; // [rsp+58h] [rbp-A68h]
  __int64 v103; // [rsp+68h] [rbp-A58h] BYREF
  _QWORD *v104; // [rsp+70h] [rbp-A50h] BYREF
  __int64 v105; // [rsp+7Ch] [rbp-A44h]
  int v106; // [rsp+84h] [rbp-A3Ch]
  __int64 v107; // [rsp+88h] [rbp-A38h]
  int v108; // [rsp+90h] [rbp-A30h]
  __int64 v109; // [rsp+94h] [rbp-A2Ch]
  int v110; // [rsp+9Ch] [rbp-A24h]
  __int64 v111; // [rsp+A0h] [rbp-A20h] BYREF
  __int64 v112; // [rsp+A8h] [rbp-A18h]
  __int64 v113; // [rsp+B0h] [rbp-A10h]
  __int64 v114; // [rsp+C0h] [rbp-A00h] BYREF
  unsigned __int64 v115; // [rsp+C8h] [rbp-9F8h]
  __int64 v116; // [rsp+D0h] [rbp-9F0h]
  __int64 v117; // [rsp+D8h] [rbp-9E8h]
  __int64 v118; // [rsp+E0h] [rbp-9E0h]
  unsigned __int64 v119[21]; // [rsp+F0h] [rbp-9D0h] BYREF
  char v120; // [rsp+198h] [rbp-928h] BYREF
  _QWORD *v121; // [rsp+1A0h] [rbp-920h] BYREF
  unsigned __int64 v122; // [rsp+1A8h] [rbp-918h] BYREF
  __int64 v123; // [rsp+1B0h] [rbp-910h]
  __int64 v124; // [rsp+1B8h] [rbp-908h]
  __int64 v125; // [rsp+1C0h] [rbp-900h]
  __int64 v126; // [rsp+1C8h] [rbp-8F8h]
  __int64 v127; // [rsp+1D0h] [rbp-8F0h]
  __int64 v128; // [rsp+1D8h] [rbp-8E8h]
  __int64 v129; // [rsp+1E0h] [rbp-8E0h]
  __int64 v130; // [rsp+1E8h] [rbp-8D8h]
  __int64 v131; // [rsp+1F0h] [rbp-8D0h]
  __int64 v132; // [rsp+1F8h] [rbp-8C8h]
  __int64 v133; // [rsp+200h] [rbp-8C0h]
  __int64 *v134; // [rsp+208h] [rbp-8B8h] BYREF
  __int64 v135; // [rsp+210h] [rbp-8B0h]
  __int64 v136; // [rsp+218h] [rbp-8A8h] BYREF
  __int64 v137; // [rsp+220h] [rbp-8A0h]
  __int64 v138; // [rsp+228h] [rbp-898h]
  __int64 v139; // [rsp+230h] [rbp-890h]
  _BYTE *v140; // [rsp+238h] [rbp-888h] BYREF
  __int64 v141; // [rsp+240h] [rbp-880h]
  _BYTE v142[2168]; // [rsp+248h] [rbp-878h] BYREF

  v5 = a1;
  v100 = a2;
  sub_2AB7C70(&v121, *(_WORD *)(a1 + 8) ^ 0x101u);
  v101 = (__int16)v121;
  v89 = v123;
  v88 = v122;
  v99 = v125;
  v90 = v126;
  v6 = v124;
  v91 = v127;
  v7 = v131;
  v92 = v128;
  v93 = v129;
  v97 = v130;
  v95 = v132;
  v8 = sub_22077B0(0x68u);
  if ( v8 )
  {
    *(_QWORD *)(v8 + 32) = v6;
    *(_QWORD *)(v8 + 88) = v7;
    *(_WORD *)(v8 + 8) = v101;
    *(_QWORD *)(v8 + 16) = v88;
    *(_QWORD *)(v8 + 24) = v89;
    *(_QWORD *)v8 = &unk_4A119F8;
    *(_QWORD *)(v8 + 40) = v99;
    *(_QWORD *)(v8 + 48) = v90;
    *(_QWORD *)(v8 + 56) = v91;
    *(_QWORD *)(v8 + 64) = v92;
    *(_QWORD *)(v8 + 72) = v93;
    *(_QWORD *)(v8 + 80) = v97;
    *(_QWORD *)(v8 + 96) = v95;
  }
  v119[0] = v8;
  sub_23A1F40(a3, v119);
  sub_233EFE0((__int64 *)v119);
  if ( LOBYTE(qword_4F90408[8]) )
  {
    v54 = (_QWORD *)sub_22077B0(0x10u);
    if ( v54 )
      *v54 = &unk_4A0F8B8;
    v121 = v54;
    sub_23A1F40(a3, (unsigned __int64 *)&v121);
    sub_233EFE0((__int64 *)&v121);
  }
  if ( a4 )
  {
    v11 = *(_BYTE *)(a1 + 11);
    if ( byte_4FDD688 && v11 )
    {
      LODWORD(v119[0]) = a2;
      sub_235D930((__int64)&v121, (int *)v119, 0, 0, 0, v9);
      sub_2353940(a3, (__int64 *)&v121);
      sub_233F7F0((__int64)&v122);
      sub_233F7D0((__int64 *)&v121);
      v11 = *(_BYTE *)(a1 + 11);
    }
    v12 = v11 ^ 1;
    v98 = *(_BYTE *)(a1 + 13);
    v13 = sub_22077B0(0x28u);
    if ( v13 )
    {
      *(_BYTE *)(v13 + 9) = 0;
      *(_BYTE *)(v13 + 11) = 0;
      *(_BYTE *)(v13 + 13) = 0;
      *(_QWORD *)v13 = &unk_4A119B8;
      *(_BYTE *)(v13 + 15) = 0;
      *(_BYTE *)(v13 + 17) = 0;
      *(_BYTE *)(v13 + 24) = 0;
      *(_DWORD *)(v13 + 28) = a2;
      *(_BYTE *)(v13 + 32) = v12;
      *(_BYTE *)(v13 + 33) = v98;
    }
    v121 = (_QWORD *)v13;
    sub_23A1F40(a3, (unsigned __int64 *)&v121);
    sub_233EFE0((__int64 *)&v121);
    v14 = (_QWORD *)sub_22077B0(0x10u);
    if ( v14 )
      *v14 = &unk_4A10FB8;
    v121 = v14;
    sub_23A1F40(a3, (unsigned __int64 *)&v121);
    sub_233EFE0((__int64 *)&v121);
    sub_291E720(&v121, 1);
    sub_23A2000(a3, (char *)&v121);
  }
  else
  {
    v10 = (_QWORD *)sub_22077B0(0x10u);
    if ( v10 )
      *v10 = &unk_4A0FD78;
    v121 = v10;
    sub_23A1F40(a3, (unsigned __int64 *)&v121);
    sub_233EFE0((__int64 *)&v121);
  }
  LOBYTE(v105) = 0;
  HIDWORD(v105) = 1;
  LOBYTE(v106) = 0;
  sub_F10C20((__int64)&v121, v105, v106);
  sub_2353C90(a3, (__int64)&v121, v15, v16, v17, v18);
  sub_233BCC0((__int64)&v121);
  if ( (unsigned int)a2 > 1 && byte_4FDD928 )
  {
    v114 = 0;
    v115 = 0;
    v116 = 0;
    v117 = 0;
    v118 = 0;
    LOBYTE(v121) = 0;
    sub_23A2060((unsigned __int64 *)&v114, (char *)&v121);
    v66 = (_QWORD *)sub_22077B0(0x10u);
    if ( v66 )
      *v66 = &unk_4A0F1B8;
    v121 = v66;
    sub_23A1F40((unsigned __int64 *)&v114, (unsigned __int64 *)&v121);
    sub_233EFE0((__int64 *)&v121);
    LOBYTE(v107) = 0;
    HIDWORD(v107) = 1;
    LOBYTE(v108) = 0;
    sub_F10C20((__int64)&v121, v107, v108);
    sub_2353C90((unsigned __int64 *)&v114, (__int64)&v121, v67, v68, v69, v70);
    sub_233BCC0((__int64)&v121);
    v119[0] = (unsigned __int64)&v119[2];
    v119[1] = 0x600000000LL;
    v71 = *(_QWORD *)(a1 + 16);
    LOWORD(v122) = 1;
    v121 = (_QWORD *)v71;
    LODWORD(v119[8]) = 0;
    memset(&v119[9], 0, 48);
    sub_23A47E0(v119, (__int64 *)&v121, v72, v73, v74, v75);
    LOBYTE(v121) = __PAIR64__(dword_5033EF0[1], dword_5033EF0[0]) == a2;
    BYTE1(v121) = 1;
    sub_23A4730(v119, (__int16 *)&v121, (__int64)dword_5033EF0, HIDWORD(a2), v76, v77);
    sub_23A20C0((__int64)&v121, (__int64)v119, 1, 1, 0, v78);
    sub_2353940((unsigned __int64 *)&v114, (__int64 *)&v121);
    sub_233F7F0((__int64)&v122);
    sub_233F7D0((__int64 *)&v121);
    v111 = 0x100010000000001LL;
    v112 = 0x1000101000000LL;
    v113 = 0;
    sub_29744D0(&v121, &v111);
    sub_23A1F80((unsigned __int64 *)&v114, (__int64 *)&v121);
    LOBYTE(v109) = 0;
    HIDWORD(v109) = 1;
    LOBYTE(v110) = 0;
    sub_F10C20((__int64)&v121, v109, v110);
    sub_2353C90((unsigned __int64 *)&v114, (__int64)&v121, v79, v80, v81, v82);
    sub_233BCC0((__int64)&v121);
    v83 = v114;
    v114 = 0;
    v124 = 0;
    v121 = (_QWORD *)v83;
    v125 = 0;
    v122 = v115;
    v115 = 0;
    v123 = v116;
    v116 = 0;
    v84 = (_QWORD *)sub_22077B0(0x30u);
    if ( v84 )
    {
      v84[4] = 0;
      v84[5] = 0;
      *v84 = &unk_4A0F5F8;
      v85 = (__int64)v121;
      v121 = 0;
      v84[1] = v85;
      v86 = v122;
      v122 = 0;
      v84[2] = v86;
      v87 = v123;
      v123 = 0;
      v84[3] = v87;
    }
    v111 = (__int64)v84;
    sub_23A1F40(a3, (unsigned __int64 *)&v111);
    sub_233EFE0(&v111);
    sub_233F7F0((__int64)&v121);
    sub_2337B30(v119);
    sub_233F7F0((__int64)&v114);
  }
  v119[2] = 0;
  v119[0] = 0x1010100000001LL;
  v119[1] = 0x1000101010001LL;
  sub_29744D0(&v121, v119);
  sub_23A1F80(a3, (__int64 *)&v121);
  if ( !a4 )
  {
    if ( !*(_BYTE *)(a1 + 10) )
      goto LABEL_20;
    goto LABEL_38;
  }
  v30 = (_QWORD *)sub_22077B0(0x10u);
  if ( v30 )
    *v30 = &unk_4A10D38;
  v121 = v30;
  sub_23A1F40(a3, (unsigned __int64 *)&v121);
  sub_233EFE0((__int64 *)&v121);
  LOBYTE(v111) = 0;
  HIDWORD(v111) = 1;
  LOBYTE(v112) = 0;
  sub_F10C20((__int64)&v121, v111, v112);
  sub_2353C90(a3, (__int64)&v121, v31, v32, v33, v34);
  sub_233BCC0((__int64)&v121);
  v35 = (_QWORD *)sub_22077B0(0x10u);
  if ( v35 )
    *v35 = &unk_4A0EF38;
  v121 = v35;
  sub_23A1F40(a3, (unsigned __int64 *)&v121);
  sub_233EFE0((__int64 *)&v121);
  if ( *(_BYTE *)(a1 + 10) )
  {
LABEL_38:
    v121 = 0;
    memset(v119, 0, 0x98u);
    v119[20] = 0;
    v119[13] = (unsigned __int64)&v119[15];
    v119[19] = (unsigned __int64)&v120;
    v134 = &v136;
    v119[9] = 1;
    v122 = 0;
    v123 = 0;
    v124 = 0;
    v125 = 0;
    v126 = 0;
    v127 = 0;
    v128 = 0;
    v129 = 0;
    v130 = 1;
    v131 = 0;
    v132 = 0;
    v133 = 0;
    v119[15] = 1;
    v135 = 0;
    v136 = 1;
    v137 = 0;
    v138 = 0;
    v139 = 0;
    v140 = v142;
    v141 = 0;
    v36 = sub_22077B0(0xB0u);
    v40 = v36;
    if ( v36 )
    {
      ++v130;
      *(_QWORD *)(v36 + 80) = 1;
      *(_QWORD *)v36 = &unk_4A10E78;
      *(_QWORD *)(v36 + 8) = v121;
      *(_QWORD *)(v36 + 16) = v122;
      *(_QWORD *)(v36 + 24) = v123;
      *(_QWORD *)(v36 + 32) = v124;
      *(_QWORD *)(v36 + 40) = v125;
      *(_QWORD *)(v36 + 48) = v126;
      *(_QWORD *)(v36 + 56) = v127;
      *(_QWORD *)(v36 + 64) = v128;
      *(_QWORD *)(v36 + 72) = v129;
      v41 = v131;
      v131 = 0;
      *(_QWORD *)(v36 + 88) = v41;
      v42 = v132;
      v132 = 0;
      *(_QWORD *)(v36 + 96) = v42;
      LODWORD(v42) = v133;
      LODWORD(v133) = 0;
      *(_DWORD *)(v36 + 104) = v42;
      *(_QWORD *)(v36 + 112) = v36 + 128;
      v43 = v135;
      *(_QWORD *)(v36 + 120) = 0;
      if ( v43 )
        sub_23A8E80(v36 + 112, (__int64)&v134, v36 + 128, v37, v38, v39);
      v44 = v137;
      v45 = (unsigned int)v141;
      *(_QWORD *)(v40 + 128) = 1;
      ++v136;
      *(_QWORD *)(v40 + 136) = v44;
      v137 = 0;
      *(_QWORD *)(v40 + 144) = v138;
      v138 = 0;
      *(_DWORD *)(v40 + 152) = v139;
      LODWORD(v139) = 0;
      *(_QWORD *)(v40 + 160) = v40 + 176;
      *(_QWORD *)(v40 + 168) = 0;
      if ( (_DWORD)v45 )
        sub_23A9220(v40 + 160, (__int64)&v140, v40 + 176, v45, v38, v39);
    }
    v114 = v40;
    sub_23A1F40(a3, (unsigned __int64 *)&v114);
    sub_233EFE0(&v114);
    v46 = &v140[88 * (unsigned int)v141];
    if ( v140 != v46 )
    {
      v47 = &v140[88 * (unsigned int)v141];
      v48 = v140;
      do
      {
        v47 -= 88;
        v49 = *((_QWORD *)v47 + 1);
        if ( (_BYTE *)v49 != v47 + 24 )
          _libc_free(v49);
      }
      while ( v48 != v47 );
      v5 = a1;
      v46 = v140;
    }
    if ( v46 != v142 )
      _libc_free((unsigned __int64)v46);
    sub_C7D6A0(v137, 16LL * (unsigned int)v139, 8);
    v50 = &v134[11 * (unsigned int)v135];
    if ( v134 != v50 )
    {
      v96 = v5;
      v51 = &v134[11 * (unsigned int)v135];
      v52 = v134;
      do
      {
        v51 -= 11;
        v53 = v51[1];
        if ( (__int64 *)v53 != v51 + 3 )
          _libc_free(v53);
      }
      while ( v52 != v51 );
      v5 = v96;
      v50 = v134;
    }
    if ( v50 != &v136 )
      _libc_free((unsigned __int64)v50);
    sub_C7D6A0(v131, 16LL * (unsigned int)v133, 8);
    sub_C7D6A0(0, 0, 8);
    sub_C7D6A0(0, 0, 8);
    if ( v100 > 1 && byte_4FDD928 )
    {
      LOBYTE(v121) = 0;
      sub_23A2060(a3, (char *)&v121);
    }
  }
LABEL_20:
  v19 = sub_22077B0(0x10u);
  if ( v19 )
  {
    *(_BYTE *)(v19 + 8) = 0;
    *(_QWORD *)v19 = &unk_4A11078;
  }
  v121 = (_QWORD *)v19;
  sub_23A1F40(a3, (unsigned __int64 *)&v121);
  sub_233EFE0((__int64 *)&v121);
  if ( !a4 )
  {
    LOBYTE(v114) = 0;
    HIDWORD(v114) = 1;
    LOBYTE(v115) = 0;
    sub_F10C20((__int64)&v121, v114, v115);
    sub_2353C90(a3, (__int64)&v121, v56, v57, v58, v59);
    sub_233BCC0((__int64)&v121);
    v61 = *(_BYTE *)(v5 + 11);
    if ( byte_4FDD688 && v61 )
    {
      LODWORD(v119[0]) = v100;
      sub_235D930((__int64)&v121, (int *)v119, 0, 0, 0, v60);
      sub_2353940(a3, (__int64 *)&v121);
      sub_233F7F0((__int64)&v122);
      sub_233F7D0((__int64 *)&v121);
      v61 = *(_BYTE *)(v5 + 11);
    }
    v62 = *(_BYTE *)(v5 + 13);
    v63 = v61 ^ 1;
    v64 = sub_22077B0(0x28u);
    if ( v64 )
    {
      *(_BYTE *)(v64 + 9) = 0;
      *(_BYTE *)(v64 + 11) = 0;
      *(_BYTE *)(v64 + 13) = 0;
      *(_QWORD *)v64 = &unk_4A119B8;
      *(_BYTE *)(v64 + 15) = 0;
      *(_BYTE *)(v64 + 17) = 0;
      *(_BYTE *)(v64 + 24) = 0;
      *(_DWORD *)(v64 + 28) = v100;
      *(_BYTE *)(v64 + 32) = v63;
      *(_BYTE *)(v64 + 33) = v62;
    }
    v121 = (_QWORD *)v64;
    sub_23A1F40(a3, (unsigned __int64 *)&v121);
    if ( v121 )
      (*(void (__fastcall **)(_QWORD *))(*v121 + 8LL))(v121);
    v65 = (_QWORD *)sub_22077B0(0x10u);
    if ( v65 )
      *v65 = &unk_4A10FB8;
    v121 = v65;
    sub_23A1F40(a3, (unsigned __int64 *)&v121);
    sub_233EFE0((__int64 *)&v121);
    sub_291E720(&v121, 1);
    sub_23A2000(a3, (char *)&v121);
  }
  if ( LOBYTE(qword_4F90408[8]) )
  {
    v55 = (_QWORD *)sub_22077B0(0x10u);
    if ( v55 )
      *v55 = &unk_4A0F8B8;
    v121 = v55;
    sub_23A1F40(a3, (unsigned __int64 *)&v121);
    sub_233EFE0((__int64 *)&v121);
  }
  LOBYTE(v119[0]) = 0;
  HIDWORD(v119[0]) = 1;
  LOBYTE(v119[1]) = 0;
  sub_F10C20((__int64)&v121, v119[0], v119[1]);
  sub_2353C90(a3, (__int64)&v121, v20, v21, v22, v23);
  sub_233BCC0((__int64)&v121);
  v24 = *(_QWORD *)(v5 + 16);
  v25 = sub_22077B0(0x18u);
  if ( v25 )
  {
    *(_QWORD *)(v25 + 8) = v24;
    *(_QWORD *)v25 = &unk_4A12478;
    *(_WORD *)(v25 + 16) = 1;
  }
  v121 = (_QWORD *)v25;
  v103 = 0;
  v122 = 0;
  v123 = 0;
  v124 = 0;
  v125 = 0;
  v126 = 0;
  LODWORD(v127) = 1;
  v26 = (_QWORD *)sub_22077B0(0x10u);
  if ( v26 )
    *v26 = &unk_4A0B640;
  v104 = v26;
  sub_23A1F40(&v122, (unsigned __int64 *)&v104);
  sub_233EFE0((__int64 *)&v104);
  v27 = (_QWORD *)sub_22077B0(0x10u);
  if ( v27 )
    *v27 = &unk_4A0B680;
  v104 = v27;
  sub_23A1F40(&v122, (unsigned __int64 *)&v104);
  sub_233EFE0((__int64 *)&v104);
  sub_233F7D0(&v103);
  sub_2353940(a3, (__int64 *)&v121);
  sub_233F7F0((__int64)&v122);
  sub_233F7D0((__int64 *)&v121);
  v28 = (_QWORD *)sub_22077B0(0x18u);
  if ( v28 )
  {
    v28[1] = 0;
    v28[2] = 0;
    *v28 = &unk_4A0EDF8;
  }
  v121 = v28;
  sub_23A1F40(a3, (unsigned __int64 *)&v121);
  return sub_233EFE0((__int64 *)&v121);
}
