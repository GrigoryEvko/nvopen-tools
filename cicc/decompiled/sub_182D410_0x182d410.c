// Function: sub_182D410
// Address: 0x182d410
//
unsigned __int64 __fastcall sub_182D410(
        __int64 a1,
        __int64 *a2,
        unsigned __int8 a3,
        int a4,
        __int64 a5,
        double a6,
        double a7,
        double a8)
{
  _QWORD *v10; // rax
  unsigned __int8 *v11; // rsi
  __int64 v12; // rax
  __int64 **v13; // r13
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 **v18; // rdx
  _QWORD *v19; // r10
  __int64 *v20; // r13
  __int64 v21; // rcx
  __int64 v22; // rax
  _QWORD *v23; // r10
  __int64 v24; // rsi
  __int64 v25; // r13
  unsigned __int8 *v26; // rsi
  _QWORD *v27; // r13
  unsigned __int64 v28; // rsi
  _QWORD *v29; // rax
  _DWORD *v30; // rdi
  __int64 v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // rax
  _DWORD *v34; // r8
  _DWORD *v35; // rdi
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 v38; // rsi
  __int64 v39; // rax
  __int64 v40; // r10
  __int64 v41; // rax
  __int64 v42; // rax
  unsigned __int8 *v43; // rsi
  unsigned __int8 *v44; // rsi
  unsigned int v45; // eax
  int v46; // ecx
  unsigned __int64 v47; // rcx
  __int8 *v48; // r9
  unsigned __int64 v49; // rax
  __m128i *v50; // rax
  __int64 v51; // rax
  size_t v52; // r12
  __int64 *v53; // rax
  __int64 **v54; // rax
  __int64 v55; // r12
  unsigned __int64 result; // rax
  _QWORD *v57; // rax
  __int64 v58; // r9
  __int64 v59; // r10
  __int64 **v60; // rax
  __int64 *v61; // rax
  __int64 v62; // rax
  __int64 v63; // r9
  __int64 v64; // r10
  __int64 *v65; // r12
  __int64 v66; // rcx
  __int64 v67; // rax
  __int64 v68; // rsi
  unsigned __int8 *v69; // rsi
  unsigned __int64 v70; // rcx
  __int8 *v71; // r9
  unsigned __int64 v72; // rax
  __int64 v73; // rax
  size_t v74; // r12
  __int64 *v75; // rax
  __int64 **v76; // rax
  _QWORD *v77; // rax
  __int64 v78; // r10
  __int64 **v79; // rax
  __int64 *v80; // rax
  __int64 v81; // rax
  __int64 v82; // r10
  unsigned __int64 v83; // rsi
  __int64 v84; // rax
  __int64 v85; // rsi
  __int64 v86; // rdx
  unsigned __int8 *v87; // rsi
  __int64 v88; // rsi
  __int64 v89; // rax
  __int64 v90; // rsi
  __int64 v91; // rdx
  unsigned __int8 *v92; // rsi
  __int64 v93; // rsi
  __int64 v94; // rax
  __int64 v95; // rsi
  __int64 v96; // rdx
  unsigned __int8 *v97; // rsi
  __int64 *v98; // r13
  __int64 v99; // rax
  __int64 v100; // rcx
  __int64 v101; // rsi
  unsigned __int8 *v102; // rsi
  __int64 v103; // [rsp+8h] [rbp-138h]
  __int64 v104; // [rsp+10h] [rbp-130h]
  __int64 v105; // [rsp+10h] [rbp-130h]
  __int64 v106; // [rsp+10h] [rbp-130h]
  __int64 *v107; // [rsp+18h] [rbp-128h]
  __int64 v108; // [rsp+18h] [rbp-128h]
  __int64 v109; // [rsp+18h] [rbp-128h]
  __int64 v110; // [rsp+18h] [rbp-128h]
  __int64 *v111; // [rsp+18h] [rbp-128h]
  unsigned __int64 *v112; // [rsp+18h] [rbp-128h]
  __int64 v113; // [rsp+18h] [rbp-128h]
  __int64 v114; // [rsp+18h] [rbp-128h]
  _QWORD *v115; // [rsp+20h] [rbp-120h]
  _QWORD *v116; // [rsp+20h] [rbp-120h]
  _QWORD *v117; // [rsp+20h] [rbp-120h]
  _QWORD *v118; // [rsp+20h] [rbp-120h]
  __int64 v119; // [rsp+20h] [rbp-120h]
  __int64 v120; // [rsp+20h] [rbp-120h]
  __int64 v121; // [rsp+20h] [rbp-120h]
  _QWORD *v122; // [rsp+20h] [rbp-120h]
  __int64 v123; // [rsp+20h] [rbp-120h]
  __int64 *v124; // [rsp+20h] [rbp-120h]
  unsigned __int8 *v127; // [rsp+30h] [rbp-110h]
  unsigned __int8 *v129; // [rsp+38h] [rbp-108h]
  __int64 *v130; // [rsp+40h] [rbp-100h]
  __int64 *v131; // [rsp+48h] [rbp-F8h] BYREF
  unsigned __int8 *v132; // [rsp+58h] [rbp-E8h] BYREF
  __int64 v133[2]; // [rsp+60h] [rbp-E0h] BYREF
  _QWORD v134[2]; // [rsp+70h] [rbp-D0h] BYREF
  __m128i *v135; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v136; // [rsp+88h] [rbp-B8h]
  __m128i v137; // [rsp+90h] [rbp-B0h] BYREF
  __m128i *v138; // [rsp+A0h] [rbp-A0h] BYREF
  size_t v139; // [rsp+A8h] [rbp-98h]
  __m128i v140; // [rsp+B0h] [rbp-90h] BYREF
  unsigned __int8 *v141; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v142; // [rsp+C8h] [rbp-78h]
  __int64 *v143; // [rsp+D0h] [rbp-70h]
  _QWORD *v144; // [rsp+D8h] [rbp-68h]
  __int64 v145; // [rsp+E0h] [rbp-60h]
  int v146; // [rsp+E8h] [rbp-58h]
  __int64 v147; // [rsp+F0h] [rbp-50h]
  __int64 v148; // [rsp+F8h] [rbp-48h]

  v131 = a2;
  v10 = (_QWORD *)sub_16498A0(a5);
  v11 = *(unsigned __int8 **)(a5 + 48);
  v141 = 0;
  v144 = v10;
  v12 = *(_QWORD *)(a5 + 40);
  v145 = 0;
  v142 = v12;
  v146 = 0;
  v147 = 0;
  v148 = 0;
  v143 = (__int64 *)(a5 + 24);
  v138 = (__m128i *)v11;
  if ( v11 )
  {
    sub_1623A60((__int64)&v138, (__int64)v11, 2);
    if ( v141 )
      sub_161E7C0((__int64)&v141, (__int64)v141);
    v141 = (unsigned __int8 *)v138;
    if ( v138 )
      sub_1623210((__int64)&v138, (unsigned __int8 *)v138, (__int64)&v141);
  }
  v137.m128i_i16[0] = 257;
  LOWORD(v134[0]) = 257;
  v13 = (__int64 **)sub_1643330(v144);
  v14 = sub_15A0680(*v131, 56, 0);
  if ( *((_BYTE *)v131 + 16) > 0x10u || *(_BYTE *)(v14 + 16) > 0x10u )
  {
    v140.m128i_i16[0] = 257;
    v15 = sub_15FB440(24, v131, v14, (__int64)&v138, 0);
    if ( v142 )
    {
      v130 = v143;
      sub_157E9D0(v142 + 40, v15);
      v88 = *v130;
      v89 = *(_QWORD *)(v15 + 24) & 7LL;
      *(_QWORD *)(v15 + 32) = v130;
      v88 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v15 + 24) = v88 | v89;
      *(_QWORD *)(v88 + 8) = v15 + 24;
      *v130 = *v130 & 7 | (v15 + 24);
    }
    sub_164B780(v15, v133);
    if ( v141 )
    {
      v132 = v141;
      sub_1623A60((__int64)&v132, (__int64)v141, 2);
      v90 = *(_QWORD *)(v15 + 48);
      v91 = v15 + 48;
      if ( v90 )
      {
        sub_161E7C0(v15 + 48, v90);
        v91 = v15 + 48;
      }
      v92 = v132;
      *(_QWORD *)(v15 + 48) = v132;
      if ( v92 )
        sub_1623210((__int64)&v132, v92, v91);
    }
  }
  else
  {
    v15 = sub_15A2D80(v131, v14, 0, a6, a7, a8);
  }
  if ( v13 != *(__int64 ***)v15 )
  {
    if ( *(_BYTE *)(v15 + 16) > 0x10u )
    {
      v140.m128i_i16[0] = 257;
      v15 = sub_15FDBD0(36, v15, (__int64)v13, (__int64)&v138, 0);
      if ( v142 )
      {
        v98 = v143;
        sub_157E9D0(v142 + 40, v15);
        v99 = *(_QWORD *)(v15 + 24);
        v100 = *v98;
        *(_QWORD *)(v15 + 32) = v98;
        v100 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v15 + 24) = v100 | v99 & 7;
        *(_QWORD *)(v100 + 8) = v15 + 24;
        *v98 = *v98 & 7 | (v15 + 24);
      }
      sub_164B780(v15, (__int64 *)&v135);
      if ( v141 )
      {
        v132 = v141;
        sub_1623A60((__int64)&v132, (__int64)v141, 2);
        v101 = *(_QWORD *)(v15 + 48);
        if ( v101 )
          sub_161E7C0(v15 + 48, v101);
        v102 = v132;
        *(_QWORD *)(v15 + 48) = v132;
        if ( v102 )
          sub_1623210((__int64)&v132, v102, v15 + 48);
      }
    }
    else
    {
      v15 = sub_15A46C0(36, (__int64 ***)v15, v13, 0);
    }
  }
  v16 = sub_182D270(*(_BYTE *)(a1 + 264), (__int64 *)&v141, (__int64)v131, a6, a7, a8);
  v17 = sub_182C3C0(a1, v16, *v131, (__int64 *)&v141, a6, a7, a8);
  v137.m128i_i16[0] = 257;
  LOWORD(v134[0]) = 257;
  v18 = (__int64 **)sub_16471D0(v144, 0);
  if ( v18 != *(__int64 ***)v17 )
  {
    if ( *(_BYTE *)(v17 + 16) > 0x10u )
    {
      v140.m128i_i16[0] = 257;
      v17 = sub_15FDBD0(46, v17, (__int64)v18, (__int64)&v138, 0);
      if ( v142 )
      {
        v124 = v143;
        sub_157E9D0(v142 + 40, v17);
        v93 = *v124;
        v94 = *(_QWORD *)(v17 + 24) & 7LL;
        *(_QWORD *)(v17 + 32) = v124;
        v93 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v17 + 24) = v93 | v94;
        *(_QWORD *)(v93 + 8) = v17 + 24;
        *v124 = *v124 & 7 | (v17 + 24);
      }
      sub_164B780(v17, v133);
      if ( v141 )
      {
        v132 = v141;
        sub_1623A60((__int64)&v132, (__int64)v141, 2);
        v95 = *(_QWORD *)(v17 + 48);
        v96 = v17 + 48;
        if ( v95 )
        {
          sub_161E7C0(v17 + 48, v95);
          v96 = v17 + 48;
        }
        v97 = v132;
        *(_QWORD *)(v17 + 48) = v132;
        if ( v97 )
          sub_1623210((__int64)&v132, v97, v96);
      }
    }
    else
    {
      v17 = sub_15A46C0(46, (__int64 ***)v17, v18, 0);
    }
  }
  v19 = sub_1648A60(64, 1u);
  if ( v19 )
  {
    v115 = v19;
    sub_15F9210((__int64)v19, *(_QWORD *)(*(_QWORD *)v17 + 24LL), v17, 0, 0, 0);
    v19 = v115;
  }
  if ( v142 )
  {
    v20 = v143;
    v116 = v19;
    sub_157E9D0(v142 + 40, (__int64)v19);
    v19 = v116;
    v21 = *v20;
    v22 = v116[3];
    v116[4] = v20;
    v21 &= 0xFFFFFFFFFFFFFFF8LL;
    v116[3] = v21 | v22 & 7;
    *(_QWORD *)(v21 + 8) = v116 + 3;
    *v20 = *v20 & 7 | (unsigned __int64)(v116 + 3);
  }
  v117 = v19;
  sub_164B780((__int64)v19, (__int64 *)&v135);
  v23 = v117;
  if ( v141 )
  {
    v138 = (__m128i *)v141;
    sub_1623A60((__int64)&v138, (__int64)v141, 2);
    v23 = v117;
    v24 = v117[6];
    v25 = (__int64)(v117 + 6);
    if ( v24 )
    {
      sub_161E7C0((__int64)(v117 + 6), v24);
      v23 = v117;
    }
    v26 = (unsigned __int8 *)v138;
    v23[6] = v138;
    if ( v26 )
    {
      v118 = v23;
      sub_1623210((__int64)&v138, v26, v25);
      v23 = v118;
    }
  }
  v137.m128i_i16[0] = 257;
  if ( *(_BYTE *)(v15 + 16) > 0x10u || *((_BYTE *)v23 + 16) > 0x10u )
  {
    v122 = v23;
    v140.m128i_i16[0] = 257;
    v77 = sub_1648A60(56, 2u);
    v78 = (__int64)v122;
    v27 = v77;
    if ( v77 )
    {
      v123 = (__int64)v77;
      v79 = *(__int64 ***)v15;
      if ( *(_BYTE *)(*(_QWORD *)v15 + 8LL) == 16 )
      {
        v105 = v78;
        v111 = v79[4];
        v80 = (__int64 *)sub_1643320(*v79);
        v81 = (__int64)sub_16463B0(v80, (unsigned int)v111);
        v82 = v105;
      }
      else
      {
        v113 = v78;
        v81 = sub_1643320(*v79);
        v82 = v113;
      }
      sub_15FEC10((__int64)v27, v81, 51, 33, v15, v82, (__int64)&v138, 0);
    }
    else
    {
      v123 = 0;
    }
    if ( v142 )
    {
      v112 = (unsigned __int64 *)v143;
      sub_157E9D0(v142 + 40, (__int64)v27);
      v83 = *v112;
      v84 = v27[3] & 7LL;
      v27[4] = v112;
      v83 &= 0xFFFFFFFFFFFFFFF8LL;
      v27[3] = v83 | v84;
      *(_QWORD *)(v83 + 8) = v27 + 3;
      *v112 = *v112 & 7 | (unsigned __int64)(v27 + 3);
    }
    sub_164B780(v123, (__int64 *)&v135);
    if ( v141 )
    {
      v133[0] = (__int64)v141;
      sub_1623A60((__int64)v133, (__int64)v141, 2);
      v85 = v27[6];
      v86 = (__int64)(v27 + 6);
      if ( v85 )
      {
        sub_161E7C0((__int64)(v27 + 6), v85);
        v86 = (__int64)(v27 + 6);
      }
      v87 = (unsigned __int8 *)v133[0];
      v27[6] = v133[0];
      if ( v87 )
        sub_1623210((__int64)v133, v87, v86);
    }
  }
  else
  {
    v27 = (_QWORD *)sub_15A37B0(0x21u, (_QWORD *)v15, v23, 0);
  }
  v28 = sub_16D5D50();
  v29 = *(_QWORD **)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    goto LABEL_41;
  v30 = dword_4FA0208;
  do
  {
    while ( 1 )
    {
      v31 = v29[2];
      v32 = v29[3];
      if ( v28 <= v29[4] )
        break;
      v29 = (_QWORD *)v29[3];
      if ( !v32 )
        goto LABEL_32;
    }
    v30 = v29;
    v29 = (_QWORD *)v29[2];
  }
  while ( v31 );
LABEL_32:
  if ( v30 == dword_4FA0208 )
    goto LABEL_41;
  if ( v28 < *((_QWORD *)v30 + 4) )
    goto LABEL_41;
  v33 = *((_QWORD *)v30 + 7);
  v34 = v30 + 12;
  if ( !v33 )
    goto LABEL_41;
  v35 = v30 + 12;
  do
  {
    while ( 1 )
    {
      v36 = *(_QWORD *)(v33 + 16);
      v37 = *(_QWORD *)(v33 + 24);
      if ( *(_DWORD *)(v33 + 32) >= dword_4FA9BA8 )
        break;
      v33 = *(_QWORD *)(v33 + 24);
      if ( !v37 )
        goto LABEL_39;
    }
    v35 = (_DWORD *)v33;
    v33 = *(_QWORD *)(v33 + 16);
  }
  while ( v36 );
LABEL_39:
  if ( v34 == v35 || dword_4FA9BA8 < v35[8] || (int)v35[9] <= 0 )
  {
LABEL_41:
    if ( !*(_BYTE *)(a1 + 264) )
      goto LABEL_47;
    v38 = 255;
  }
  else
  {
    v38 = dword_4FA9C40;
    if ( dword_4FA9C40 == -1 )
      goto LABEL_47;
  }
  v137.m128i_i16[0] = 257;
  v39 = sub_15A0680(*(_QWORD *)v15, v38, 0);
  if ( *(_BYTE *)(v15 + 16) > 0x10u || *(_BYTE *)(v39 + 16) > 0x10u )
  {
    v119 = v39;
    v140.m128i_i16[0] = 257;
    v57 = sub_1648A60(56, 2u);
    v58 = v119;
    v59 = (__int64)v57;
    if ( v57 )
    {
      v120 = (__int64)v57;
      v60 = *(__int64 ***)v15;
      if ( *(_BYTE *)(*(_QWORD *)v15 + 8LL) == 16 )
      {
        v103 = v59;
        v104 = v58;
        v107 = v60[4];
        v61 = (__int64 *)sub_1643320(*v60);
        v62 = (__int64)sub_16463B0(v61, (unsigned int)v107);
        v63 = v104;
        v64 = v103;
      }
      else
      {
        v106 = v59;
        v114 = v58;
        v62 = sub_1643320(*v60);
        v64 = v106;
        v63 = v114;
      }
      v108 = v64;
      sub_15FEC10(v64, v62, 51, 33, v15, v63, (__int64)&v138, 0);
      v59 = v108;
    }
    else
    {
      v120 = 0;
    }
    if ( v142 )
    {
      v65 = v143;
      v109 = v59;
      sub_157E9D0(v142 + 40, v59);
      v59 = v109;
      v66 = *v65;
      v67 = *(_QWORD *)(v109 + 24);
      *(_QWORD *)(v109 + 32) = v65;
      v66 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v109 + 24) = v66 | v67 & 7;
      *(_QWORD *)(v66 + 8) = v109 + 24;
      *v65 = *v65 & 7 | (v109 + 24);
    }
    v110 = v59;
    sub_164B780(v120, (__int64 *)&v135);
    v40 = v110;
    if ( v141 )
    {
      v133[0] = (__int64)v141;
      sub_1623A60((__int64)v133, (__int64)v141, 2);
      v40 = v110;
      v68 = *(_QWORD *)(v110 + 48);
      if ( v68 )
      {
        sub_161E7C0(v110 + 48, v68);
        v40 = v110;
      }
      v69 = (unsigned __int8 *)v133[0];
      *(_QWORD *)(v40 + 48) = v133[0];
      if ( v69 )
      {
        v121 = v40;
        sub_1623210((__int64)v133, v69, v110 + 48);
        v40 = v121;
      }
    }
  }
  else
  {
    v40 = sub_15A37B0(0x21u, (_QWORD *)v15, (_QWORD *)v39, 0);
  }
  v140.m128i_i16[0] = 257;
  v27 = (_QWORD *)sub_1281C00((__int64 *)&v141, (__int64)v27, v40, (__int64)&v138);
LABEL_47:
  v138 = *(__m128i **)(a1 + 160);
  v41 = sub_161BE60(&v138, 1u, 0x186A0u);
  v42 = sub_1AA92B0(v27, a5, *(_BYTE *)(a1 + 265) ^ 1u, v41, 0, 0);
  v43 = *(unsigned __int8 **)(v42 + 48);
  v142 = *(_QWORD *)(v42 + 40);
  v143 = (__int64 *)(v42 + 24);
  v138 = (__m128i *)v43;
  if ( v43 )
  {
    sub_1623A60((__int64)&v138, (__int64)v43, 2);
    v44 = v141;
    if ( !v141 )
      goto LABEL_50;
  }
  else
  {
    v44 = v141;
    if ( !v141 )
      goto LABEL_52;
  }
  sub_161E7C0((__int64)&v141, (__int64)v44);
LABEL_50:
  v141 = (unsigned __int8 *)v138;
  if ( v138 )
    sub_1623210((__int64)&v138, (unsigned __int8 *)v138, (__int64)&v141);
LABEL_52:
  v45 = *(_DWORD *)(a1 + 200);
  v46 = a4 + 16 * (a3 + 2 * *(unsigned __int8 *)(a1 + 265));
  if ( v45 > 4 )
  {
    if ( v45 != 32 )
LABEL_54:
      sub_16BD130("unsupported architecture", 1u);
    v47 = (unsigned int)(v46 + 64);
    v48 = &v140.m128i_i8[5];
    do
    {
      *--v48 = v47 % 0xA + 48;
      v49 = v47;
      v47 /= 0xAu;
    }
    while ( v49 > 9 );
    v133[0] = (__int64)v134;
    sub_182BB80(v133, v48, (__int64)v140.m128i_i64 + 5);
    v50 = (__m128i *)sub_2241130(v133, 0, 0, "int3\nnopl ", 10);
    v135 = &v137;
    if ( (__m128i *)v50->m128i_i64[0] == &v50[1] )
    {
      v137 = _mm_loadu_si128(v50 + 1);
    }
    else
    {
      v135 = (__m128i *)v50->m128i_i64[0];
      v137.m128i_i64[0] = v50[1].m128i_i64[0];
    }
    v136 = v50->m128i_i64[1];
    v50->m128i_i64[0] = (__int64)v50[1].m128i_i64;
    v50->m128i_i64[1] = 0;
    v50[1].m128i_i8[0] = 0;
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v136) <= 5 )
      sub_4262D8((__int64)"basic_string::append");
    v51 = sub_2241490(&v135, "(%rax)", 6);
    v138 = &v140;
    if ( *(_QWORD *)v51 == v51 + 16 )
    {
      v140 = _mm_loadu_si128((const __m128i *)(v51 + 16));
    }
    else
    {
      v138 = *(__m128i **)v51;
      v140.m128i_i64[0] = *(_QWORD *)(v51 + 16);
    }
    v139 = *(_QWORD *)(v51 + 8);
    *(_QWORD *)v51 = v51 + 16;
    *(_QWORD *)(v51 + 8) = 0;
    *(_BYTE *)(v51 + 16) = 0;
    v52 = v139;
    v127 = (unsigned __int8 *)v138;
    v132 = (unsigned __int8 *)*v131;
    v53 = (__int64 *)sub_1643270(v144);
    v54 = (__int64 **)sub_1644EA0(v53, &v132, 1, 0);
    v55 = sub_15EE570(v54, v127, v52, "{rdi}", 5u, 1, 0, 0);
    if ( v138 != &v140 )
      j_j___libc_free_0(v138, v140.m128i_i64[0] + 1);
    if ( v135 != &v137 )
      j_j___libc_free_0(v135, v137.m128i_i64[0] + 1);
    if ( (_QWORD *)v133[0] != v134 )
      j_j___libc_free_0(v133[0], v134[0] + 1LL);
  }
  else
  {
    if ( v45 <= 2 )
      goto LABEL_54;
    v70 = (unsigned int)(v46 + 2304);
    v71 = &v140.m128i_i8[5];
    do
    {
      *--v71 = v70 % 0xA + 48;
      v72 = v70;
      v70 /= 0xAu;
    }
    while ( v72 > 9 );
    v135 = &v137;
    sub_182BB80((__int64 *)&v135, v71, (__int64)v140.m128i_i64 + 5);
    v73 = sub_2241130(&v135, 0, 0, "brk #", 5);
    v138 = &v140;
    if ( *(_QWORD *)v73 == v73 + 16 )
    {
      v140 = _mm_loadu_si128((const __m128i *)(v73 + 16));
    }
    else
    {
      v138 = *(__m128i **)v73;
      v140.m128i_i64[0] = *(_QWORD *)(v73 + 16);
    }
    v139 = *(_QWORD *)(v73 + 8);
    *(_QWORD *)v73 = v73 + 16;
    *(_QWORD *)(v73 + 8) = 0;
    *(_BYTE *)(v73 + 16) = 0;
    v74 = v139;
    v129 = (unsigned __int8 *)v138;
    v133[0] = *v131;
    v75 = (__int64 *)sub_1643270(v144);
    v76 = (__int64 **)sub_1644EA0(v75, v133, 1, 0);
    v55 = sub_15EE570(v76, v129, v74, "{x0}", 4u, 1, 0, 0);
    if ( v138 != &v140 )
      j_j___libc_free_0(v138, v140.m128i_i64[0] + 1);
    if ( v135 != &v137 )
      j_j___libc_free_0(v135, v137.m128i_i64[0] + 1);
  }
  v140.m128i_i16[0] = 257;
  result = sub_1285290((__int64 *)&v141, *(_QWORD *)(*(_QWORD *)v55 + 24LL), v55, (int)&v131, 1, (__int64)&v138, 0);
  if ( v141 )
    return sub_161E7C0((__int64)&v141, (__int64)v141);
  return result;
}
