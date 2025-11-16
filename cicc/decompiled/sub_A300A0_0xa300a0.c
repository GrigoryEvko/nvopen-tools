// Function: sub_A300A0
// Address: 0xa300a0
//
__int64 __fastcall sub_A300A0(__int64 *a1, __int64 a2, __int64 a3, unsigned int *a4)
{
  __int64 v4; // r9
  _BYTE *v7; // rsi
  __int64 v8; // r15
  __int64 v9; // r12
  __int64 v10; // r12
  int v11; // eax
  __int64 v12; // rax
  volatile signed __int32 *v13; // rax
  unsigned int v14; // eax
  unsigned int v15; // r8d
  unsigned __int8 *v16; // r12
  unsigned __int8 *v17; // r9
  __int64 v18; // rax
  int v19; // r15d
  _QWORD *v20; // rax
  __int64 v21; // r12
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  int v26; // eax
  __int64 v27; // rdx
  int v28; // edx
  int v29; // eax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdx
  int v36; // r8d
  _QWORD *v37; // r12
  _QWORD *v38; // r15
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  int v42; // eax
  __int64 v43; // rdx
  int v44; // edx
  int v45; // eax
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdx
  int v52; // r8d
  _QWORD *v53; // r12
  _QWORD *v54; // r15
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rax
  int v58; // eax
  __int64 v59; // rdx
  int v60; // edx
  int v61; // eax
  __int64 v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rdx
  int v68; // r8d
  _QWORD *v69; // r12
  __int64 v70; // rsi
  _QWORD *v71; // r15
  __int64 v72; // rax
  __int64 v73; // rdx
  __int64 v74; // rax
  int v75; // eax
  __int64 v76; // rdx
  int v77; // edx
  int v78; // eax
  __int64 v79; // rdx
  __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 v83; // rax
  __int64 v84; // rdx
  int v85; // r8d
  __int64 v86; // r12
  unsigned int *v87; // rbx
  unsigned int *v88; // r13
  unsigned int v89; // esi
  __int64 v90; // rbx
  __int64 v91; // rdi
  __int64 v93; // [rsp+0h] [rbp-420h]
  _QWORD *v94; // [rsp+0h] [rbp-420h]
  _QWORD *v95; // [rsp+0h] [rbp-420h]
  _QWORD *v96; // [rsp+0h] [rbp-420h]
  __int64 v97; // [rsp+8h] [rbp-418h]
  __int64 v98; // [rsp+8h] [rbp-418h]
  __int64 v99; // [rsp+8h] [rbp-418h]
  __int64 v100; // [rsp+8h] [rbp-418h]
  __int64 v101; // [rsp+10h] [rbp-410h]
  __int64 v102; // [rsp+10h] [rbp-410h]
  __int64 v103; // [rsp+10h] [rbp-410h]
  __int64 v104; // [rsp+10h] [rbp-410h]
  unsigned __int8 *v105; // [rsp+10h] [rbp-410h]
  unsigned int v106; // [rsp+18h] [rbp-408h]
  unsigned int v107; // [rsp+18h] [rbp-408h]
  unsigned int v108; // [rsp+18h] [rbp-408h]
  unsigned int v109; // [rsp+18h] [rbp-408h]
  unsigned int v110; // [rsp+18h] [rbp-408h]
  int v111; // [rsp+18h] [rbp-408h]
  int v112; // [rsp+18h] [rbp-408h]
  int v113; // [rsp+18h] [rbp-408h]
  int v114; // [rsp+18h] [rbp-408h]
  int v115; // [rsp+18h] [rbp-408h]
  int v116; // [rsp+18h] [rbp-408h]
  int v117; // [rsp+18h] [rbp-408h]
  int v118; // [rsp+18h] [rbp-408h]
  int v119; // [rsp+18h] [rbp-408h]
  int v120; // [rsp+18h] [rbp-408h]
  int v121; // [rsp+18h] [rbp-408h]
  unsigned int v122; // [rsp+18h] [rbp-408h]
  int v123; // [rsp+18h] [rbp-408h]
  __int64 v125; // [rsp+20h] [rbp-400h] BYREF
  volatile signed __int32 *v126; // [rsp+28h] [rbp-3F8h]
  __int64 v127; // [rsp+30h] [rbp-3F0h] BYREF
  volatile signed __int32 *v128; // [rsp+38h] [rbp-3E8h]
  _BYTE *v129; // [rsp+40h] [rbp-3E0h] BYREF
  __int64 v130; // [rsp+48h] [rbp-3D8h]
  _BYTE v131[256]; // [rsp+50h] [rbp-3D0h] BYREF
  __int64 v132; // [rsp+150h] [rbp-2D0h] BYREF
  __int64 v133; // [rsp+158h] [rbp-2C8h]
  __int64 v134; // [rsp+160h] [rbp-2C0h]
  __int64 v135[73]; // [rsp+168h] [rbp-2B8h] BYREF
  __int64 v136; // [rsp+3B0h] [rbp-70h]
  unsigned int *v137; // [rsp+3E0h] [rbp-40h]

  v4 = a3;
  v132 = a2;
  v7 = (_BYTE *)a1[21];
  if ( v7 == (_BYTE *)a1[22] )
  {
    sub_A28060((__int64)(a1 + 20), v7, &v132);
    v4 = a3;
  }
  else
  {
    if ( v7 )
    {
      *(_QWORD *)v7 = a2;
      v7 = (_BYTE *)a1[21];
    }
    a1[21] = (__int64)(v7 + 8);
  }
  v8 = 2;
  sub_A28440((__int64)&v132, a2, (__int64)(a1 + 1), *a1, 0, v4);
  v137 = a4;
  sub_A19830(v132, 8u, 3u);
  v9 = v132;
  sub_A17B10(v132, 3u, *(_DWORD *)(v132 + 56));
  sub_A17CC0(v9, 1u, 6);
  sub_A17CC0(v9, 1u, 6);
  sub_A17CC0(v9, 2u, 6);
  v129 = v131;
  v10 = 8;
  v130 = 0x4000000000LL;
  v11 = sub_A16070(*(char **)(v134 + 200), *(_QWORD *)(v134 + 208));
  if ( v11 )
  {
    if ( v11 == 1 )
      v10 = 7;
  }
  else
  {
    v10 = 0;
    v8 = 8;
  }
  sub_A23770(&v125);
  sub_A186C0(v125, 16, 1);
  sub_A186C0(v125, 0, 6);
  sub_A186C0(v125, v10, v8);
  v12 = v125;
  v125 = 0;
  v127 = v12;
  v13 = v126;
  v126 = 0;
  v128 = v13;
  v14 = sub_A1AB30(v132, &v127);
  v15 = v14;
  if ( v128 )
  {
    v106 = v14;
    sub_A191D0(v128);
    v15 = v106;
  }
  v16 = *(unsigned __int8 **)(v134 + 200);
  v17 = &v16[*(_QWORD *)(v134 + 208)];
  if ( v16 != v17 )
  {
    v18 = (unsigned int)v130;
    do
    {
      v19 = *v16;
      if ( v18 + 1 > (unsigned __int64)HIDWORD(v130) )
      {
        v105 = v17;
        v122 = v15;
        sub_C8D5F0(&v129, v131, v18 + 1, 4);
        v18 = (unsigned int)v130;
        v17 = v105;
        v15 = v122;
      }
      ++v16;
      *(_DWORD *)&v129[4 * v18] = v19;
      v18 = (unsigned int)(v130 + 1);
      LODWORD(v130) = v130 + 1;
    }
    while ( v17 != v16 );
  }
  sub_A214F0(v132, 0x10u, (__int64)&v129, v15);
  LODWORD(v130) = 0;
  if ( v126 )
    sub_A191D0(v126);
  v20 = (_QWORD *)v134;
  v21 = *(_QWORD *)(v134 + 16);
  v93 = v134 + 8;
  if ( v134 + 8 != v21 )
  {
    do
    {
      v22 = 0;
      if ( v21 )
        v22 = v21 - 56;
      v97 = v133;
      v23 = sub_BD5D20(v22);
      v107 = v24;
      v101 = v23;
      v25 = sub_C94890(v23, v24);
      v26 = sub_C0CA60(v97, v101, (v25 << 32) | v107);
      v27 = (unsigned int)v130;
      if ( (unsigned __int64)(unsigned int)v130 + 1 > HIDWORD(v130) )
      {
        v121 = v26;
        sub_C8D5F0(&v129, v131, (unsigned int)v130 + 1LL, 4);
        v27 = (unsigned int)v130;
        v26 = v121;
      }
      *(_DWORD *)&v129[4 * v27] = v26;
      LODWORD(v130) = v130 + 1;
      sub_BD5D20(v22);
      v29 = v28;
      v30 = (unsigned int)v130;
      if ( (unsigned __int64)(unsigned int)v130 + 1 > HIDWORD(v130) )
      {
        v123 = v29;
        sub_C8D5F0(&v129, v131, (unsigned int)v130 + 1LL, 4);
        v30 = (unsigned int)v130;
        v29 = v123;
      }
      *(_DWORD *)&v129[4 * v30] = v29;
      LODWORD(v130) = v130 + 1;
      v31 = (unsigned int)v130;
      if ( (unsigned __int64)(unsigned int)v130 + 1 > HIDWORD(v130) )
      {
        sub_C8D5F0(&v129, v131, (unsigned int)v130 + 1LL, 4);
        v31 = (unsigned int)v130;
      }
      *(_DWORD *)&v129[4 * v31] = 0;
      LODWORD(v130) = v130 + 1;
      v32 = (unsigned int)v130;
      if ( (unsigned __int64)(unsigned int)v130 + 1 > HIDWORD(v130) )
      {
        sub_C8D5F0(&v129, v131, (unsigned int)v130 + 1LL, 4);
        v32 = (unsigned int)v130;
      }
      *(_DWORD *)&v129[4 * v32] = 0;
      LODWORD(v130) = v130 + 1;
      v33 = (unsigned int)v130;
      if ( (unsigned __int64)(unsigned int)v130 + 1 > HIDWORD(v130) )
      {
        sub_C8D5F0(&v129, v131, (unsigned int)v130 + 1LL, 4);
        v33 = (unsigned int)v130;
      }
      *(_DWORD *)&v129[4 * v33] = 0;
      v34 = (unsigned int)(v130 + 1);
      v35 = *(_BYTE *)(v22 + 32) & 0xF;
      LODWORD(v130) = v130 + 1;
      if ( (unsigned int)v35 > 0xA )
        goto LABEL_99;
      v36 = dword_3F23000[v35];
      if ( v34 + 1 > (unsigned __int64)HIDWORD(v130) )
      {
        v120 = dword_3F23000[v35];
        sub_C8D5F0(&v129, v131, v34 + 1, 4);
        v34 = (unsigned int)v130;
        v36 = v120;
      }
      *(_DWORD *)&v129[4 * v34] = v36;
      LODWORD(v130) = v130 + 1;
      sub_A214F0(v132, 7u, (__int64)&v129, 0);
      LODWORD(v130) = 0;
      v21 = *(_QWORD *)(v21 + 8);
    }
    while ( v93 != v21 );
    v20 = (_QWORD *)v134;
  }
  v37 = (_QWORD *)v20[4];
  v94 = v20 + 3;
  if ( v37 != v20 + 3 )
  {
    do
    {
      v38 = 0;
      if ( v37 )
        v38 = v37 - 7;
      v98 = v133;
      v39 = sub_BD5D20(v38);
      v108 = v40;
      v102 = v39;
      v41 = sub_C94890(v39, v40);
      v42 = sub_C0CA60(v98, v102, (v41 << 32) | v108);
      v43 = (unsigned int)v130;
      if ( (unsigned __int64)(unsigned int)v130 + 1 > HIDWORD(v130) )
      {
        v118 = v42;
        sub_C8D5F0(&v129, v131, (unsigned int)v130 + 1LL, 4);
        v43 = (unsigned int)v130;
        v42 = v118;
      }
      *(_DWORD *)&v129[4 * v43] = v42;
      LODWORD(v130) = v130 + 1;
      sub_BD5D20(v38);
      v45 = v44;
      v46 = (unsigned int)v130;
      if ( (unsigned __int64)(unsigned int)v130 + 1 > HIDWORD(v130) )
      {
        v119 = v45;
        sub_C8D5F0(&v129, v131, (unsigned int)v130 + 1LL, 4);
        v46 = (unsigned int)v130;
        v45 = v119;
      }
      *(_DWORD *)&v129[4 * v46] = v45;
      LODWORD(v130) = v130 + 1;
      v47 = (unsigned int)v130;
      if ( (unsigned __int64)(unsigned int)v130 + 1 > HIDWORD(v130) )
      {
        sub_C8D5F0(&v129, v131, (unsigned int)v130 + 1LL, 4);
        v47 = (unsigned int)v130;
      }
      *(_DWORD *)&v129[4 * v47] = 0;
      LODWORD(v130) = v130 + 1;
      v48 = (unsigned int)v130;
      if ( (unsigned __int64)(unsigned int)v130 + 1 > HIDWORD(v130) )
      {
        sub_C8D5F0(&v129, v131, (unsigned int)v130 + 1LL, 4);
        v48 = (unsigned int)v130;
      }
      *(_DWORD *)&v129[4 * v48] = 0;
      LODWORD(v130) = v130 + 1;
      v49 = (unsigned int)v130;
      if ( (unsigned __int64)(unsigned int)v130 + 1 > HIDWORD(v130) )
      {
        sub_C8D5F0(&v129, v131, (unsigned int)v130 + 1LL, 4);
        v49 = (unsigned int)v130;
      }
      *(_DWORD *)&v129[4 * v49] = 0;
      v50 = (unsigned int)(v130 + 1);
      v51 = v38[4] & 0xF;
      LODWORD(v130) = v130 + 1;
      if ( (unsigned int)v51 > 0xA )
        goto LABEL_99;
      v52 = dword_3F23000[v51];
      if ( v50 + 1 > (unsigned __int64)HIDWORD(v130) )
      {
        v117 = dword_3F23000[v51];
        sub_C8D5F0(&v129, v131, v50 + 1, 4);
        v50 = (unsigned int)v130;
        v52 = v117;
      }
      *(_DWORD *)&v129[4 * v50] = v52;
      LODWORD(v130) = v130 + 1;
      sub_A214F0(v132, 8u, (__int64)&v129, 0);
      LODWORD(v130) = 0;
      v37 = (_QWORD *)v37[1];
    }
    while ( v94 != v37 );
    v20 = (_QWORD *)v134;
  }
  v53 = (_QWORD *)v20[6];
  v95 = v20 + 5;
  if ( v20 + 5 != v53 )
  {
    do
    {
      v54 = 0;
      if ( v53 )
        v54 = v53 - 6;
      v99 = v133;
      v55 = sub_BD5D20(v54);
      v109 = v56;
      v103 = v55;
      v57 = sub_C94890(v55, v56);
      v58 = sub_C0CA60(v99, v103, (v57 << 32) | v109);
      v59 = (unsigned int)v130;
      if ( (unsigned __int64)(unsigned int)v130 + 1 > HIDWORD(v130) )
      {
        v116 = v58;
        sub_C8D5F0(&v129, v131, (unsigned int)v130 + 1LL, 4);
        v59 = (unsigned int)v130;
        v58 = v116;
      }
      *(_DWORD *)&v129[4 * v59] = v58;
      LODWORD(v130) = v130 + 1;
      sub_BD5D20(v54);
      v61 = v60;
      v62 = (unsigned int)v130;
      if ( (unsigned __int64)(unsigned int)v130 + 1 > HIDWORD(v130) )
      {
        v112 = v61;
        sub_C8D5F0(&v129, v131, (unsigned int)v130 + 1LL, 4);
        v62 = (unsigned int)v130;
        v61 = v112;
      }
      *(_DWORD *)&v129[4 * v62] = v61;
      LODWORD(v130) = v130 + 1;
      v63 = (unsigned int)v130;
      if ( (unsigned __int64)(unsigned int)v130 + 1 > HIDWORD(v130) )
      {
        sub_C8D5F0(&v129, v131, (unsigned int)v130 + 1LL, 4);
        v63 = (unsigned int)v130;
      }
      *(_DWORD *)&v129[4 * v63] = 0;
      LODWORD(v130) = v130 + 1;
      v64 = (unsigned int)v130;
      if ( (unsigned __int64)(unsigned int)v130 + 1 > HIDWORD(v130) )
      {
        sub_C8D5F0(&v129, v131, (unsigned int)v130 + 1LL, 4);
        v64 = (unsigned int)v130;
      }
      *(_DWORD *)&v129[4 * v64] = 0;
      LODWORD(v130) = v130 + 1;
      v65 = (unsigned int)v130;
      if ( (unsigned __int64)(unsigned int)v130 + 1 > HIDWORD(v130) )
      {
        sub_C8D5F0(&v129, v131, (unsigned int)v130 + 1LL, 4);
        v65 = (unsigned int)v130;
      }
      *(_DWORD *)&v129[4 * v65] = 0;
      v66 = (unsigned int)(v130 + 1);
      v67 = v54[4] & 0xF;
      LODWORD(v130) = v130 + 1;
      if ( (unsigned int)v67 > 0xA )
        goto LABEL_99;
      v68 = dword_3F23000[v67];
      if ( v66 + 1 > (unsigned __int64)HIDWORD(v130) )
      {
        v115 = dword_3F23000[v67];
        sub_C8D5F0(&v129, v131, v66 + 1, 4);
        v66 = (unsigned int)v130;
        v68 = v115;
      }
      *(_DWORD *)&v129[4 * v66] = v68;
      LODWORD(v130) = v130 + 1;
      sub_A214F0(v132, 0xEu, (__int64)&v129, 0);
      LODWORD(v130) = 0;
      v53 = (_QWORD *)v53[1];
    }
    while ( v95 != v53 );
    v20 = (_QWORD *)v134;
  }
  v69 = (_QWORD *)v20[8];
  v70 = (__int64)(v20 + 7);
  v96 = v20 + 7;
  if ( v20 + 7 != v69 )
  {
    while ( 1 )
    {
      v71 = 0;
      if ( v69 )
        v71 = v69 - 7;
      v100 = v133;
      v72 = sub_BD5D20(v71);
      v110 = v73;
      v104 = v72;
      v74 = sub_C94890(v72, v73);
      v75 = sub_C0CA60(v100, v104, (v74 << 32) | v110);
      v76 = (unsigned int)v130;
      if ( (unsigned __int64)(unsigned int)v130 + 1 > HIDWORD(v130) )
      {
        v114 = v75;
        sub_C8D5F0(&v129, v131, (unsigned int)v130 + 1LL, 4);
        v76 = (unsigned int)v130;
        v75 = v114;
      }
      *(_DWORD *)&v129[4 * v76] = v75;
      LODWORD(v130) = v130 + 1;
      sub_BD5D20(v71);
      v78 = v77;
      v79 = (unsigned int)v130;
      if ( (unsigned __int64)(unsigned int)v130 + 1 > HIDWORD(v130) )
      {
        v113 = v78;
        sub_C8D5F0(&v129, v131, (unsigned int)v130 + 1LL, 4);
        v79 = (unsigned int)v130;
        v78 = v113;
      }
      *(_DWORD *)&v129[4 * v79] = v78;
      LODWORD(v130) = v130 + 1;
      v80 = (unsigned int)v130;
      if ( (unsigned __int64)(unsigned int)v130 + 1 > HIDWORD(v130) )
      {
        sub_C8D5F0(&v129, v131, (unsigned int)v130 + 1LL, 4);
        v80 = (unsigned int)v130;
      }
      *(_DWORD *)&v129[4 * v80] = 0;
      LODWORD(v130) = v130 + 1;
      v81 = (unsigned int)v130;
      if ( (unsigned __int64)(unsigned int)v130 + 1 > HIDWORD(v130) )
      {
        sub_C8D5F0(&v129, v131, (unsigned int)v130 + 1LL, 4);
        v81 = (unsigned int)v130;
      }
      *(_DWORD *)&v129[4 * v81] = 0;
      LODWORD(v130) = v130 + 1;
      v82 = (unsigned int)v130;
      if ( (unsigned __int64)(unsigned int)v130 + 1 > HIDWORD(v130) )
      {
        sub_C8D5F0(&v129, v131, (unsigned int)v130 + 1LL, 4);
        v82 = (unsigned int)v130;
      }
      *(_DWORD *)&v129[4 * v82] = 0;
      v83 = (unsigned int)(v130 + 1);
      v84 = v71[4] & 0xF;
      LODWORD(v130) = v130 + 1;
      if ( (unsigned int)v84 > 0xA )
        break;
      v85 = dword_3F23000[v84];
      if ( v83 + 1 > (unsigned __int64)HIDWORD(v130) )
      {
        v111 = dword_3F23000[v84];
        sub_C8D5F0(&v129, v131, v83 + 1, 4);
        v83 = (unsigned int)v130;
        v85 = v111;
      }
      v70 = 18;
      *(_DWORD *)&v129[4 * v83] = v85;
      LODWORD(v130) = v130 + 1;
      sub_A214F0(v132, 0x12u, (__int64)&v129, 0);
      LODWORD(v130) = 0;
      v69 = (_QWORD *)v69[1];
      if ( v96 == v69 )
        goto LABEL_87;
    }
LABEL_99:
    BUG();
  }
LABEL_87:
  if ( v129 != v131 )
    _libc_free(v129, v70);
  sub_A2D2B0(&v132);
  v86 = v132;
  v87 = v137;
  v88 = v137 + 5;
  sub_A17B10(v132, 3u, *(_DWORD *)(v132 + 56));
  sub_A17CC0(v86, 0x11u, 6);
  sub_A17CC0(v86, 5u, 6);
  do
  {
    v89 = *v87++;
    sub_A17CC0(v86, v89, 6);
  }
  while ( v88 != v87 );
  sub_A192A0(v132);
  v90 = v136;
  while ( v90 )
  {
    sub_A167C0(*(_QWORD *)(v90 + 24));
    v91 = v90;
    v90 = *(_QWORD *)(v90 + 16);
    j_j___libc_free_0(v91, 48);
  }
  return sub_A17F40(v135);
}
