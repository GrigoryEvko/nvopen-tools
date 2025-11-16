// Function: sub_1B55A30
// Address: 0x1b55a30
//
__int64 __fastcall sub_1B55A30(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        _QWORD *a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 v13; // r12
  __int64 v15; // r13
  __int64 v16; // rdi
  unsigned int v17; // r15d
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rax
  _BOOL8 v23; // r14
  _QWORD *v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // r15
  int v30; // r14d
  __int64 v31; // r12
  __int64 v32; // rdx
  _QWORD *v33; // rcx
  __int64 v34; // rdi
  __int64 v35; // rax
  char v36; // di
  __int64 v37; // rsi
  __int64 v38; // rax
  __int64 v39; // r8
  __int64 v40; // rdi
  __int64 i; // rcx
  __int64 v42; // rcx
  __int64 v43; // rax
  __int64 v44; // r12
  __int64 v45; // r14
  __int64 v46; // rax
  __int64 v47; // rsi
  __int64 v48; // rax
  _QWORD *v49; // r8
  _QWORD *v50; // rax
  unsigned __int8 v51; // al
  __int64 v52; // rsi
  __int64 v53; // rdi
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // rax
  __int64 v57; // r8
  __int64 v58; // r8
  __int64 v59; // r14
  __int64 v60; // rdx
  __int64 v61; // r15
  __int64 v62; // rdx
  __int64 v63; // rax
  __int64 v64; // r10
  const char *v65; // rax
  __int64 v66; // rdx
  _QWORD *v67; // r8
  __int64 v68; // r10
  _QWORD *v69; // r10
  __int64 v70; // rdx
  unsigned __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rsi
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // rax
  __int64 v77; // r8
  __int64 v78; // r12
  unsigned int v79; // eax
  __int64 v80; // rdx
  __int64 v81; // rdx
  __int64 v82; // rcx
  __int64 v83; // r15
  _QWORD *v84; // rax
  _QWORD *v85; // rdi
  __int64 v86; // rsi
  __int64 v87; // rdi
  __int64 v88; // rbx
  __int64 v89; // rax
  __int64 v90; // rdx
  __int64 v91; // rax
  int v92; // ebx
  _QWORD *v93; // rax
  __int64 v94; // rax
  __int64 v95; // r14
  __int64 v96; // rdi
  __int64 v97; // r15
  unsigned __int64 v98; // rax
  __int64 v99; // rcx
  __int64 v100; // r8
  __int64 v101; // r9
  __int64 v102; // rsi
  _QWORD *v103; // rax
  __int64 v104; // rax
  char v105; // [rsp+Fh] [rbp-141h]
  _QWORD *v106; // [rsp+18h] [rbp-138h]
  __int64 v107; // [rsp+20h] [rbp-130h]
  __int64 v108; // [rsp+28h] [rbp-128h]
  unsigned __int8 v109; // [rsp+28h] [rbp-128h]
  __int64 v110; // [rsp+28h] [rbp-128h]
  __int64 v111; // [rsp+28h] [rbp-128h]
  __int64 v112; // [rsp+28h] [rbp-128h]
  __int64 v113; // [rsp+30h] [rbp-120h]
  __int64 v114; // [rsp+30h] [rbp-120h]
  __int64 v115; // [rsp+30h] [rbp-120h]
  __int64 v116; // [rsp+30h] [rbp-120h]
  int v117; // [rsp+38h] [rbp-118h]
  __int64 v118; // [rsp+38h] [rbp-118h]
  __int64 v119; // [rsp+40h] [rbp-110h]
  int v121; // [rsp+48h] [rbp-108h]
  int v122; // [rsp+48h] [rbp-108h]
  unsigned __int8 v123; // [rsp+48h] [rbp-108h]
  __int64 v124; // [rsp+58h] [rbp-F8h] BYREF
  __int64 v125; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v126; // [rsp+68h] [rbp-E8h] BYREF
  __int64 v127; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v128; // [rsp+78h] [rbp-D8h] BYREF
  const char *v129; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v130; // [rsp+88h] [rbp-C8h]
  __m128i v131; // [rsp+90h] [rbp-C0h] BYREF
  _WORD v132[16]; // [rsp+A0h] [rbp-B0h] BYREF
  __m128i v133; // [rsp+C0h] [rbp-90h] BYREF
  _QWORD v134[3]; // [rsp+D0h] [rbp-80h] BYREF
  int v135; // [rsp+E8h] [rbp-68h]
  __int64 v136; // [rsp+F0h] [rbp-60h]
  __int64 v137; // [rsp+F8h] [rbp-58h]
  _BYTE v138[80]; // [rsp+100h] [rbp-50h] BYREF

  v13 = a2;
  v15 = *(_QWORD *)(a2 + 40);
  v16 = *(_QWORD *)(a2 - 72);
  if ( *(_QWORD *)(a1 - 72) == v16 && *(_QWORD *)(a1 - 48) != *(_QWORD *)(a1 - 24) )
  {
    if ( sub_157F0B0(v15) )
    {
      v17 = 1;
      v23 = *(_QWORD *)(a1 - 24) == v15;
      v24 = (_QWORD *)sub_157E9C0(v15);
      v25 = sub_1643320(v24);
      v26 = sub_159C470(v25, v23, 0);
      sub_1593B40((_QWORD *)(a2 - 72), v26);
      return v17;
    }
    a2 = *(_QWORD *)(a2 - 72);
    if ( (unsigned int)sub_1C105D0(a4[1], a2, v15) )
    {
      if ( !byte_4FB6EA0 )
        return 0;
      for ( i = *(_QWORD *)(v15 + 48); i != v15 + 40; i = *(_QWORD *)(v42 + 8) )
      {
        if ( !sub_1B44350(i) )
        {
          if ( !v42 )
            BUG();
          if ( (unsigned int)*(unsigned __int8 *)(v42 - 8) - 25 > 9 )
            return 0;
        }
      }
      if ( !(unsigned __int8)sub_1B553C0(v15, *(_QWORD *)(a1 + 40)) )
        return 0;
      v43 = *(_QWORD *)(a1 - 24);
      if ( v43 == v15 && v43 )
      {
        a2 = v15;
        if ( !(unsigned __int8)sub_1B553C0(*(_QWORD *)(v13 - 24), v15) )
          return 0;
      }
      else
      {
        a2 = v15;
        if ( !(unsigned __int8)sub_1B553C0(*(_QWORD *)(v13 - 48), v15) )
          return 0;
      }
      if ( (unsigned int)sub_1648EF0(v15) != 2 )
        return 0;
    }
    if ( !byte_4FB6F80 )
    {
      v17 = sub_1B45E10((_BYTE *)v15);
      if ( (_BYTE)v17 )
      {
        v133.m128i_i64[0] = *(_QWORD *)(v15 + 8);
        sub_15CDD40(v133.m128i_i64);
        v86 = *(_QWORD *)(v15 + 48);
        v87 = *(_QWORD *)(v13 - 72);
        v88 = v133.m128i_i64[0];
        v89 = v86 - 24;
        if ( !v86 )
          v89 = 0;
        v118 = v89;
        v129 = (const char *)v133.m128i_i64[0];
        v131.m128i_i64[0] = (__int64)sub_1649960(v87);
        LOWORD(v134[0]) = 773;
        v133.m128i_i64[0] = (__int64)&v131;
        v131.m128i_i64[1] = v90;
        v133.m128i_i64[1] = (__int64)".pr";
        if ( v88 )
        {
          v115 = v88;
          v91 = v88;
          v92 = 0;
          do
          {
            ++v92;
            v129 = *(const char **)(v91 + 8);
            sub_15CDD40((__int64 *)&v129);
            v91 = (__int64)v129;
          }
          while ( v129 );
          v122 = v92;
          v88 = v115;
        }
        else
        {
          v122 = 0;
        }
        v93 = (_QWORD *)sub_157E9C0(v15);
        v116 = sub_1643320(v93);
        v94 = sub_1648B60(64);
        v95 = v94;
        if ( v94 )
        {
          sub_15F1EA0(v94, v116, 53, 0, 0, v118);
          *(_DWORD *)(v95 + 56) = v122;
          sub_164B780(v95, v133.m128i_i64);
          sub_1648880(v95, *(_DWORD *)(v95 + 56), 1);
        }
        v133.m128i_i64[0] = v88;
        if ( v88 )
        {
          v96 = v88;
          v123 = v17;
          do
          {
            v97 = sub_1648700(v96)[5];
            v98 = sub_157EBA0(v97);
            v102 = *(_QWORD *)(v13 - 72);
            if ( *(_BYTE *)(v98 + 16) == 26
              && v13 != v98
              && (*(_DWORD *)(v98 + 20) & 0xFFFFFFF) == 3
              && *(_QWORD *)(v98 - 72) == v102
              && *(_QWORD *)(v98 - 48) != *(_QWORD *)(v98 - 24) )
            {
              v119 = *(_QWORD *)(v98 - 24);
              v103 = (_QWORD *)sub_157E9C0(v15);
              v104 = sub_1643320(v103);
              v102 = sub_159C470(v104, v15 == v119, 0);
            }
            sub_1704F80(v95, v102, v97, v99, v100, v101);
            v133.m128i_i64[0] = *(_QWORD *)(v133.m128i_i64[0] + 8);
            sub_15CDD40(v133.m128i_i64);
            v96 = v133.m128i_i64[0];
          }
          while ( v133.m128i_i64[0] );
          v17 = v123;
        }
        sub_1593B40((_QWORD *)(v13 - 72), v95);
        return v17;
      }
    }
    v16 = *(_QWORD *)(v13 - 72);
  }
  if ( *(_BYTE *)(v16 + 16) == 5 && (unsigned __int8)sub_1593DF0(v16, a2, a3, a4) )
    return 0;
  if ( byte_4FB7300 )
  {
    v17 = sub_1B4BCE0(a1, v13, a3, a5, a6, a7, a8, a9, a10, a11, a12);
    if ( (_BYTE)v17 )
      return v17;
  }
  sub_1580910(&v133);
  v131 = v133;
  sub_1974F30((__int64)v132, (__int64)v134);
  if ( !v131.m128i_i64[0] )
  {
    sub_A17130((__int64)v132);
    sub_A17130((__int64)v138);
    sub_A17130((__int64)v134);
    return 0;
  }
  sub_A17130((__int64)v132);
  sub_A17130((__int64)v138);
  sub_A17130((__int64)v134);
  if ( v13 != v131.m128i_i64[0] - 24 )
    return 0;
  v18 = *(_QWORD *)(a1 - 24);
  v19 = *(_QWORD *)(v13 - 24);
  if ( v19 == v18 )
  {
    v121 = 0;
    v27 = -24;
    v117 = 0;
  }
  else
  {
    v20 = *(_QWORD *)(v13 - 48);
    if ( v18 == v20 )
    {
      v121 = 1;
      v27 = -24;
      v117 = 0;
    }
    else
    {
      v21 = *(_QWORD *)(a1 - 48);
      if ( v19 == v21 )
      {
        v121 = 0;
        v27 = -48;
        v117 = 1;
      }
      else
      {
        if ( v20 != v21 )
          return 0;
        v121 = 1;
        v27 = -48;
        v117 = 1;
      }
    }
  }
  v28 = *(_QWORD *)(a1 + v27);
  v113 = v28;
  if ( v15 == v28 )
    return 0;
  v108 = v13;
  v29 = *(_QWORD *)(v28 + 48);
  v30 = 4;
  while ( 1 )
  {
    if ( !v29 )
      BUG();
    v31 = v29 - 24;
    if ( *(_BYTE *)(v29 - 8) != 77 )
      break;
    if ( !--v30 )
      return 0;
    v34 = sub_1455EB0(v29 - 24, v15);
    if ( *(_BYTE *)(v34 + 16) == 5 )
    {
      if ( (unsigned __int8)sub_1593DF0(v34, v15, v32, v33) )
        return 0;
    }
    v35 = 0x17FFFFFFE8LL;
    v36 = *(_BYTE *)(v29 - 1) & 0x40;
    v37 = *(_DWORD *)(v29 - 4) & 0xFFFFFFF;
    if ( (*(_DWORD *)(v29 - 4) & 0xFFFFFFF) != 0 )
    {
      v32 = 24LL * *(unsigned int *)(v29 + 32) + 8;
      v38 = 0;
      do
      {
        v33 = (_QWORD *)(v31 - 24LL * (unsigned int)v37);
        if ( v36 )
          v33 = *(_QWORD **)(v29 - 32);
        if ( *(_QWORD *)(a1 + 40) == *(_QWORD *)((char *)v33 + v32) )
        {
          v35 = 24 * v38;
          goto LABEL_33;
        }
        ++v38;
        v32 += 8;
      }
      while ( (_DWORD)v37 != (_DWORD)v38 );
      v35 = 0x17FFFFFFE8LL;
    }
LABEL_33:
    if ( v36 )
    {
      v39 = *(_QWORD *)(v29 - 32);
    }
    else
    {
      v37 = (unsigned int)v37;
      v32 = 24LL * (unsigned int)v37;
      v39 = v31 - v32;
    }
    v40 = *(_QWORD *)(v39 + v35);
    if ( *(_BYTE *)(v40 + 16) == 5 )
    {
      if ( (unsigned __int8)sub_1593DF0(v40, v37, v32, v33) )
        return 0;
    }
    v29 = *(_QWORD *)(v29 + 8);
  }
  v44 = v108;
  v45 = *(_QWORD *)(v108 - 24LL * (v121 ^ 1) - 24);
  if ( v15 == v45 )
  {
    v82 = *(_QWORD *)(v15 + 56);
    v133.m128i_i64[0] = (__int64)"infloop";
    v112 = v82;
    LOWORD(v134[0]) = 259;
    v83 = sub_157E9C0(v15);
    v84 = (_QWORD *)sub_22077B0(64);
    v45 = (__int64)v84;
    if ( v84 )
      sub_157FB60(v84, v83, (__int64)&v133, v112, 0);
    v85 = sub_1648A60(56, 1u);
    if ( v85 )
      sub_15F8590((__int64)v85, v45, v45);
  }
  v106 = *(_QWORD **)(a1 - 72);
  v46 = sub_16498A0(a1);
  v47 = *(_QWORD *)(a1 + 48);
  v133.m128i_i64[0] = 0;
  v134[1] = v46;
  v48 = *(_QWORD *)(a1 + 40);
  v134[2] = 0;
  v133.m128i_i64[1] = v48;
  v135 = 0;
  v136 = 0;
  v137 = 0;
  v134[0] = a1 + 24;
  v131.m128i_i64[0] = v47;
  if ( v47 )
  {
    sub_1623A60((__int64)&v131, v47, 2);
    if ( v133.m128i_i64[0] )
      sub_161E7C0((__int64)&v133, v133.m128i_i64[0]);
    v133.m128i_i64[0] = v131.m128i_i64[0];
    if ( v131.m128i_i64[0] )
      sub_1623210((__int64)&v131, (unsigned __int8 *)v131.m128i_i64[0], (__int64)&v133);
  }
  if ( v117 )
  {
    v129 = sub_1649960((__int64)v106);
    v132[0] = 773;
    v130 = v81;
    v131.m128i_i64[0] = (__int64)&v129;
    v131.m128i_i64[1] = (__int64)".not";
    v106 = sub_1B47DD0(v133.m128i_i64, (__int64)v106, v131.m128i_i64);
  }
  v49 = *(_QWORD **)(v44 - 72);
  if ( v121 )
  {
    v111 = *(_QWORD *)(v44 - 72);
    v129 = sub_1649960(v111);
    v132[0] = 773;
    v130 = v80;
    v131.m128i_i64[0] = (__int64)&v129;
    v131.m128i_i64[1] = (__int64)".not";
    v49 = sub_1B47DD0(v133.m128i_i64, v111, v131.m128i_i64);
  }
  v131.m128i_i64[0] = (__int64)"brmerge";
  v132[0] = 259;
  v50 = sub_1B47BD0(v133.m128i_i64, (__int64)v106, (__int64)v49, v131.m128i_i64);
  sub_1593B40((_QWORD *)(a1 - 72), (__int64)v50);
  sub_1593B40((_QWORD *)(a1 - 24), v113);
  sub_1593B40((_QWORD *)(a1 - 48), v45);
  v109 = sub_1625AE0(a1, &v124, &v125);
  v51 = sub_1625AE0(v44, &v126, &v127);
  v105 = v51 | v109;
  if ( v51 | v109 )
  {
    if ( v109 )
    {
      v55 = v124;
      v53 = v125;
      if ( v51 )
      {
        v52 = v126;
        v54 = v127;
        v56 = v127 + v126;
      }
      else
      {
        v56 = 2;
        v54 = 1;
        v52 = 1;
        v127 = 1;
        v126 = 1;
      }
    }
    else
    {
      v52 = v126;
      v53 = 1;
      v54 = v127;
      v125 = 1;
      v124 = 1;
      v55 = 1;
      v56 = v127 + v126;
    }
    if ( !v117 )
    {
      v57 = v55;
      v55 = v53;
      v53 = v57;
    }
    if ( !v121 )
    {
      v58 = v52;
      v52 = v54;
      v54 = v58;
    }
    v131.m128i_i64[0] = v55 * v54 + v53 * v56;
    v131.m128i_i64[1] = v52 * v55;
    sub_1B425D0((unsigned __int64 *)&v131, 2);
    sub_1B423A0(a1, v131.m128i_u32[0], v131.m128i_u32[2]);
  }
  sub_1B44430(v45, *(_QWORD *)(a1 + 40), v15);
  v59 = sub_157F280(v113);
  v128 = v59;
  v110 = v60;
  if ( v60 != v59 )
  {
    v61 = v59;
    do
    {
      v78 = sub_1455EB0(v61, v15);
      v79 = sub_1B46990(v61, *(_QWORD *)(a1 + 40));
      if ( (*(_BYTE *)(v61 + 23) & 0x40) != 0 )
        v62 = *(_QWORD *)(v61 - 8);
      else
        v62 = v61 - 24LL * (*(_DWORD *)(v61 + 20) & 0xFFFFFFF);
      v63 = 3LL * v79;
      v64 = *(_QWORD *)(v62 + 8 * v63);
      v114 = 8 * v63;
      if ( v78 != v64 )
      {
        v107 = *(_QWORD *)(v62 + 8 * v63);
        v65 = sub_1649960(v64);
        v132[0] = 773;
        v129 = v65;
        v130 = v66;
        v131.m128i_i64[0] = (__int64)&v129;
        v131.m128i_i64[1] = (__int64)".mux";
        v67 = sub_1B47760(v133.m128i_i64, (__int64)v106, v107, v78, v131.m128i_i64, 0);
        if ( (*(_BYTE *)(v61 + 23) & 0x40) != 0 )
          v68 = *(_QWORD *)(v61 - 8);
        else
          v68 = v61 - 24LL * (*(_DWORD *)(v61 + 20) & 0xFFFFFFF);
        v69 = (_QWORD *)(v114 + v68);
        if ( *v69 )
        {
          v70 = v69[1];
          v71 = v69[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v71 = v70;
          if ( v70 )
            *(_QWORD *)(v70 + 16) = *(_QWORD *)(v70 + 16) & 3LL | v71;
        }
        *v69 = v67;
        if ( v67 )
        {
          v72 = v67[1];
          v69[1] = v72;
          if ( v72 )
            *(_QWORD *)(v72 + 16) = (unsigned __int64)(v69 + 1) | *(_QWORD *)(v72 + 16) & 3LL;
          v69[2] = (unsigned __int64)(v67 + 1) | v69[2] & 3LL;
          v67[1] = v69;
        }
        if ( v105 )
        {
          v73 = v124;
          v74 = v125;
          if ( v117 )
          {
            v74 = v124;
            v73 = v125;
          }
          v75 = v126;
          v76 = v127;
          if ( v121 )
          {
            v76 = v126;
            v75 = v127;
          }
          v131.m128i_i64[1] = v75 * v74;
          v131.m128i_i64[0] = v73 * (v75 + v76);
          sub_1B425D0((unsigned __int64 *)&v131, 2);
          sub_1B423A0(v77, v131.m128i_u32[0], v131.m128i_u32[2]);
        }
      }
      sub_1B42F80((__int64)&v128);
      v61 = v128;
    }
    while ( v110 != v128 );
  }
  v17 = 1;
  sub_17CD270(v133.m128i_i64);
  return v17;
}
