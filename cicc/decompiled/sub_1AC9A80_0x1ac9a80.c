// Function: sub_1AC9A80
// Address: 0x1ac9a80
//
_BOOL8 __fastcall sub_1AC9A80(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r13
  _BOOL4 v14; // r15d
  __int64 i; // r13
  unsigned __int64 v17; // rdi
  __int64 *v18; // rdx
  __int64 *v19; // r14
  __int64 *v20; // rax
  __int64 *v21; // r13
  __int64 v22; // rbx
  unsigned __int64 v23; // r15
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  bool v28; // zf
  __int64 v29; // r15
  int v30; // r9d
  unsigned __int64 v31; // rax
  __int64 *v32; // rax
  unsigned int v33; // r9d
  unsigned __int64 v34; // r14
  __int64 v35; // r13
  __int64 v36; // r14
  unsigned __int64 v37; // rbx
  unsigned __int64 v38; // rdx
  int v39; // eax
  unsigned __int64 v40; // rax
  __int64 v41; // r13
  unsigned __int64 v42; // rax
  __int64 v43; // r14
  __int64 v44; // rdx
  int v45; // edi
  __int64 v46; // rax
  __int64 v47; // r12
  __int64 v48; // rbx
  __int64 v49; // r15
  unsigned __int64 v50; // rax
  __int64 v51; // r14
  __int64 v52; // r14
  __int64 v53; // rax
  __int64 v54; // r12
  __int64 v55; // r14
  __int64 v56; // rax
  unsigned __int64 v57; // r15
  __int64 v58; // rbx
  __int64 *v59; // r14
  unsigned __int64 *v60; // rcx
  unsigned __int64 v61; // rdx
  double v62; // xmm4_8
  double v63; // xmm5_8
  unsigned __int64 v64; // rdx
  __int64 v65; // rcx
  unsigned __int64 v66; // rax
  __int64 v67; // rdx
  __int64 v68; // r14
  __int64 v69; // rcx
  unsigned __int8 *v70; // rsi
  unsigned __int8 *v71; // rsi
  unsigned __int8 *v72; // rsi
  unsigned __int8 *v73; // rsi
  __int64 v74; // rax
  unsigned __int64 v75; // rbx
  __int64 v76; // r13
  __int64 v77; // r12
  _QWORD *v78; // rbx
  unsigned __int64 *v79; // rcx
  __int64 *v80; // r14
  unsigned __int64 v81; // rdx
  __int64 v82; // rsi
  double v83; // xmm4_8
  double v84; // xmm5_8
  __int64 *v85; // rbx
  unsigned __int64 v86; // rdx
  __int64 v87; // rcx
  unsigned __int64 v88; // rax
  __int64 v89; // r14
  unsigned __int8 *v90; // rsi
  unsigned __int8 *v91; // rsi
  __int64 v92; // r14
  _QWORD *v93; // rdi
  __int64 v94; // rsi
  __int64 v95; // rax
  __int64 v96; // rsi
  __int64 v97; // rdx
  unsigned __int8 *v98; // rsi
  __int64 v99; // rax
  __int64 v100; // r13
  _QWORD *v101; // rdi
  __int64 v102; // [rsp+8h] [rbp-1D8h]
  unsigned __int64 v103; // [rsp+10h] [rbp-1D0h]
  __int64 v104; // [rsp+18h] [rbp-1C8h]
  __int64 v105; // [rsp+18h] [rbp-1C8h]
  __int64 v106; // [rsp+20h] [rbp-1C0h]
  __int64 *v107; // [rsp+20h] [rbp-1C0h]
  int v108; // [rsp+28h] [rbp-1B8h]
  __int64 v109; // [rsp+28h] [rbp-1B8h]
  __int64 v110; // [rsp+30h] [rbp-1B0h]
  unsigned __int64 v111; // [rsp+30h] [rbp-1B0h]
  _QWORD *v112; // [rsp+30h] [rbp-1B0h]
  __int64 *v113; // [rsp+30h] [rbp-1B0h]
  __int64 v114; // [rsp+30h] [rbp-1B0h]
  __int64 v115; // [rsp+30h] [rbp-1B0h]
  __int64 v116; // [rsp+38h] [rbp-1A8h]
  __int64 v117; // [rsp+38h] [rbp-1A8h]
  __int64 *v118; // [rsp+38h] [rbp-1A8h]
  unsigned __int64 v119; // [rsp+38h] [rbp-1A8h]
  __int64 v120; // [rsp+48h] [rbp-198h] BYREF
  __int64 v121; // [rsp+50h] [rbp-190h] BYREF
  __int64 v122; // [rsp+58h] [rbp-188h] BYREF
  __int64 v123; // [rsp+60h] [rbp-180h] BYREF
  __int64 v124; // [rsp+68h] [rbp-178h] BYREF
  unsigned __int8 *v125[2]; // [rsp+70h] [rbp-170h] BYREF
  __int16 v126; // [rsp+80h] [rbp-160h]
  __int64 v127[2]; // [rsp+90h] [rbp-150h] BYREF
  __int64 *v128; // [rsp+A0h] [rbp-140h]
  unsigned __int8 *v129; // [rsp+A8h] [rbp-138h]
  unsigned __int8 *v130; // [rsp+B0h] [rbp-130h] BYREF
  __int64 v131; // [rsp+B8h] [rbp-128h]
  __int64 *v132; // [rsp+C0h] [rbp-120h]
  __int64 v133; // [rsp+C8h] [rbp-118h]
  __int64 v134; // [rsp+D0h] [rbp-110h]
  int v135; // [rsp+D8h] [rbp-108h]
  __int64 v136; // [rsp+E0h] [rbp-100h]
  __int64 v137; // [rsp+E8h] [rbp-F8h]
  unsigned __int8 *v138; // [rsp+100h] [rbp-E0h] BYREF
  __int64 *v139; // [rsp+108h] [rbp-D8h]
  __int64 *v140; // [rsp+110h] [rbp-D0h]
  __int64 v141; // [rsp+118h] [rbp-C8h]
  int v142; // [rsp+120h] [rbp-C0h]
  _BYTE v143[184]; // [rsp+128h] [rbp-B8h] BYREF

  v120 = a2;
  v131 = a1;
  v133 = sub_157E9C0(a1);
  v132 = (__int64 *)(a1 + 40);
  v11 = *(_QWORD *)(a1 + 48);
  v130 = 0;
  v134 = 0;
  v135 = 0;
  v136 = 0;
  v137 = 0;
  if ( !v11 )
    BUG();
  if ( *(_BYTE *)(v11 - 8) == 77 )
  {
LABEL_3:
    v12 = sub_1AA6DC0(a1, &v121, &v122);
    if ( !v12 )
      goto LABEL_6;
    if ( *(_BYTE *)(v12 + 16) <= 0x17u )
      goto LABEL_6;
    v13 = *(_QWORD *)(v12 + 40);
    if ( *(_WORD *)(v13 + 18) )
      goto LABEL_6;
    v46 = sub_1AA6DC0(*(_QWORD *)(v12 + 40), &v123, &v124);
    v47 = v46;
    if ( !v46 )
      goto LABEL_6;
    if ( *(_BYTE *)(v46 + 16) <= 0x17u )
      goto LABEL_6;
    v48 = *(_QWORD *)(v46 + 40);
    v49 = v123;
    if ( v48 != v123 && v48 != v124 )
      goto LABEL_6;
    if ( v13 != v121 && v13 != v122 )
      goto LABEL_6;
    v116 = v121;
    v50 = sub_157EBA0(v13);
    v51 = *(_QWORD *)(v13 + 48);
    v111 = v50;
    if ( v51 )
      v51 -= 24;
    if ( !sub_1AC9680(&v120, v48, v13, v49, v116) || !(v14 = sub_1AC9680(&v120, v48, v13, v124, v122)) )
    {
LABEL_6:
      v14 = 0;
      goto LABEL_7;
    }
    if ( v51 )
    {
      v52 = v51 + 24;
      if ( !v111 || (v111 += 24LL, v52 != v111) )
      {
        v53 = v47;
        v54 = v52;
        v55 = v53;
        do
        {
          if ( !v54 )
            goto LABEL_174;
          if ( *(_BYTE *)(v54 - 8) == 77
            || (unsigned __int8)sub_15F3040(v54 - 24)
            || sub_15F3330(v54 - 24)
            || !(unsigned __int8)sub_14AF470(v54 - 24, 0, 0, 0) )
          {
            goto LABEL_6;
          }
          v54 = *(_QWORD *)(v54 + 8);
        }
        while ( v54 != v111 );
        v47 = v55;
      }
    }
    else if ( v111 )
    {
LABEL_174:
      BUG();
    }
    v117 = v48 + 40;
    v112 = (_QWORD *)(*(_QWORD *)(v48 + 40) & 0xFFFFFFFFFFFFFFF8LL);
    sub_157EA20(v48 + 40, (__int64)(v112 - 3));
    v59 = (__int64 *)(v13 + 40);
    v60 = (unsigned __int64 *)v112[1];
    v61 = *v112 & 0xFFFFFFFFFFFFFFF8LL;
    *v60 = v61 | *v60 & 7;
    *(_QWORD *)(v61 + 8) = v60;
    *v112 &= 7uLL;
    v112[1] = 0;
    sub_164BEC0((__int64)(v112 - 3), (__int64)(v112 - 3), v61, (__int64)v60, a3, a4, a5, a6, v62, v63, a9, a10);
    if ( v13 + 40 != (*(_QWORD *)(v13 + 40) & 0xFFFFFFFFFFFFFFF8LL) && (__int64 *)v117 != v59 )
    {
      v113 = *(__int64 **)(v13 + 48);
      sub_157EA80(v117, v13 + 40, (__int64)v113, v13 + 40);
      if ( v59 != v113 )
      {
        v64 = *(_QWORD *)(v13 + 40) & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)((*v113 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v59;
        *(_QWORD *)(v13 + 40) = *(_QWORD *)(v13 + 40) & 7LL | *v113 & 0xFFFFFFFFFFFFFFF8LL;
        v65 = *(_QWORD *)(v48 + 40);
        *(_QWORD *)(v64 + 8) = v117;
        v65 &= 0xFFFFFFFFFFFFFFF8LL;
        *v113 = v65 | *v113 & 7;
        *(_QWORD *)(v65 + 8) = v113;
        *(_QWORD *)(v48 + 40) = v64 | *(_QWORD *)(v48 + 40) & 7LL;
      }
    }
    v66 = sub_157EBA0(v48);
    v114 = v66;
    if ( *(_BYTE *)(v66 + 16) != 26 )
      BUG();
    v68 = *(_QWORD *)(v66 - 72);
    v109 = v131;
    v107 = v132;
    v69 = v66;
    v70 = *(unsigned __int8 **)(v66 + 48);
    v131 = *(_QWORD *)(v66 + 40);
    v138 = v70;
    v132 = (__int64 *)(v66 + 24);
    if ( v70 )
    {
      sub_1623A60((__int64)&v138, (__int64)v70, 2);
      v71 = v130;
      if ( !v130 )
        goto LABEL_106;
    }
    else
    {
      v71 = v130;
      if ( !v130 )
      {
LABEL_108:
        LOWORD(v128) = 257;
        if ( *(_BYTE *)(v68 + 16) > 0x10u )
          goto LABEL_148;
        if ( sub_1593BB0(v68, (__int64)v71, v67, v69) )
          goto LABEL_112;
        if ( *(_BYTE *)(v47 + 16) > 0x10u )
        {
LABEL_148:
          LOWORD(v140) = 257;
          v47 = sub_15FB440(27, (__int64 *)v47, v68, (__int64)&v138, 0);
          if ( v131 )
          {
            v118 = v132;
            sub_157E9D0(v131 + 40, v47);
            v94 = *v118;
            v95 = *(_QWORD *)(v47 + 24) & 7LL;
            *(_QWORD *)(v47 + 32) = v118;
            v94 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v47 + 24) = v94 | v95;
            *(_QWORD *)(v94 + 8) = v47 + 24;
            *v118 = *v118 & 7 | (v47 + 24);
          }
          sub_164B780(v47, v127);
          if ( v130 )
          {
            v125[0] = v130;
            sub_1623A60((__int64)v125, (__int64)v130, 2);
            v96 = *(_QWORD *)(v47 + 48);
            v97 = v47 + 48;
            if ( v96 )
            {
              sub_161E7C0(v47 + 48, v96);
              v97 = v47 + 48;
            }
            v98 = v125[0];
            *(unsigned __int8 **)(v47 + 48) = v125[0];
            if ( v98 )
              sub_1623210((__int64)v125, v98, v97);
          }
        }
        else
        {
          v47 = sub_15A2D10((__int64 *)v47, v68, *(double *)a3.m128_u64, a4, a5);
        }
LABEL_112:
        sub_1648780(v114, v68, v47);
        v131 = v109;
        v132 = v107;
        if ( v107 != (__int64 *)(v109 + 40) )
        {
          if ( !v107 )
            BUG();
          v72 = (unsigned __int8 *)v107[3];
          v138 = v72;
          if ( v72 )
          {
            sub_1623A60((__int64)&v138, (__int64)v72, 2);
            v73 = v130;
            if ( !v130 )
              goto LABEL_117;
            goto LABEL_116;
          }
          v73 = v130;
          if ( v130 )
          {
LABEL_116:
            sub_161E7C0((__int64)&v130, (__int64)v73);
LABEL_117:
            v130 = v138;
            if ( v138 )
              sub_1623210((__int64)&v138, v138, (__int64)&v130);
          }
        }
        if ( v48 != v123 )
        {
          sub_157EE90(v123);
          sub_157F980(v123);
        }
        if ( v48 != v124 )
        {
          sub_157EE90(v124);
          sub_157F980(v124);
        }
        sub_157EE90(v13);
        sub_157F980(v13);
        goto LABEL_7;
      }
    }
    sub_161E7C0((__int64)&v130, (__int64)v71);
LABEL_106:
    v71 = v138;
    v130 = v138;
    if ( v138 )
      sub_1623210((__int64)&v138, v138, (__int64)&v130);
    goto LABEL_108;
  }
  for ( i = *(_QWORD *)(a1 + 8); i; i = *(_QWORD *)(i + 8) )
  {
    if ( (unsigned __int8)(*((_BYTE *)sub_1648700(i) + 16) - 25) <= 9u )
      break;
  }
  v139 = (__int64 *)v143;
  v138 = 0;
  v140 = (__int64 *)v143;
  v141 = 16;
  v142 = 0;
  sub_1AC99A0((__int64)&v138, i, 0);
  v17 = (unsigned __int64)v140;
  v18 = v139;
  if ( v140 == v139 )
    v19 = &v140[HIDWORD(v141)];
  else
    v19 = &v140[(unsigned int)v141];
  if ( v140 == v19 )
    goto LABEL_18;
  v20 = v140;
  while ( 1 )
  {
    v21 = v20;
    if ( (unsigned __int64)*v20 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v19 == ++v20 )
      goto LABEL_18;
  }
  if ( v19 == v20 )
  {
LABEL_18:
    if ( v18 != (__int64 *)v17 )
      _libc_free(v17);
    goto LABEL_3;
  }
  v106 = 0;
  v108 = -1;
  v110 = 0;
  v104 = 0;
  v22 = *v20;
  do
  {
    v23 = sub_157EBA0(v22);
    if ( *(_BYTE *)(v23 + 16) != 26 )
      goto LABEL_58;
    v24 = sub_157F0B0(v22);
    if ( (*(_DWORD *)(v23 + 20) & 0xFFFFFFF) == 1 )
    {
      if ( v104 || !v24 || !sub_183E920((__int64)&v138, v24) || *(_WORD *)(v22 + 18) )
        goto LABEL_58;
      v104 = v22;
    }
    else
    {
      v25 = *(_QWORD *)(v23 - 72);
      if ( !v25 )
        goto LABEL_58;
      v26 = *(_QWORD *)(v25 + 8);
      if ( !v26 || *(_QWORD *)(v26 + 8) )
        goto LABEL_58;
      if ( v24 && sub_183E920((__int64)&v138, v24) )
      {
        if ( *(_WORD *)(v22 + 18) )
          goto LABEL_58;
        v103 = v23;
        v56 = *(_QWORD *)(v22 + 48);
        v102 = v22;
        v57 = v23 + 24;
        while ( v57 != v56 )
        {
          v58 = *(_QWORD *)(v56 + 8);
          if ( *(_BYTE *)(v56 - 8) == 77 || !(unsigned __int8)sub_14AF470(v56 - 24, 0, 0, 0) )
            goto LABEL_58;
          v56 = v58;
        }
        v23 = v103;
        v22 = v102;
      }
      else
      {
        if ( v110 )
          goto LABEL_58;
        v110 = v22;
      }
      v27 = *(_QWORD *)(v23 - 24);
      v28 = a1 == v27;
      if ( a1 == v27 )
        v27 = *(_QWORD *)(v23 - 48);
      v29 = v27;
      if ( v108 == -1 )
      {
        v108 = !v28;
      }
      else if ( !v28 != v108 )
      {
        goto LABEL_58;
      }
      if ( sub_183E920((__int64)&v138, v27) )
      {
        v31 = sub_157EBA0(v29);
        if ( *(_BYTE *)(v31 + 16) == 26 )
        {
          if ( (*(_DWORD *)(v31 + 20) & 0xFFFFFFF) != 1 )
            v22 = v106;
          v106 = v22;
        }
      }
      else
      {
        v106 = v22;
      }
    }
    v32 = v21 + 1;
    if ( v21 + 1 == v19 )
      break;
    while ( 1 )
    {
      v22 = *v32;
      v21 = v32;
      if ( (unsigned __int64)*v32 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v19 == ++v32 )
        goto LABEL_45;
    }
  }
  while ( v19 != v32 );
LABEL_45:
  LOBYTE(v30) = v110 == 0;
  LOBYTE(v32) = v106 == 0;
  v33 = (unsigned int)v32 | v30;
  LOBYTE(v33) = (v106 == v110) | v33;
  v14 = v33;
  if ( (_BYTE)v33 )
    goto LABEL_58;
  v34 = sub_157EBA0(v106);
  v35 = sub_15F4DF0(v34, 0);
  v36 = sub_15F4DF0(v34, 1u);
  v37 = sub_157EBA0(v35);
  if ( *(_BYTE *)(v37 + 16) != 26 )
  {
    v38 = sub_157EBA0(v36);
    if ( *(_BYTE *)(v38 + 16) == 26 )
      goto LABEL_49;
LABEL_58:
    v17 = (unsigned __int64)v140;
    v18 = v139;
    goto LABEL_18;
  }
  v38 = sub_157EBA0(v36);
  v39 = *(_DWORD *)(v37 + 20) & 0xFFFFFFF;
  if ( *(_BYTE *)(v38 + 16) != 26 )
  {
    if ( v39 == 1 && v36 == sub_15F4DF0(v37, 0) )
      goto LABEL_126;
    goto LABEL_58;
  }
  if ( v39 == 1 )
  {
    v119 = v38;
    v99 = sub_15F4DF0(v37, 0);
    v38 = v119;
    if ( v36 == v99 )
    {
LABEL_126:
      if ( (*(_DWORD *)(v37 + 20) & 0xFFFFFFF) == 1 )
      {
        v74 = *(_QWORD *)(v36 + 48);
        if ( !v74 )
          BUG();
        if ( *(_BYTE *)(v74 - 8) != 77 )
        {
          v75 = sub_157EBA0(v110);
          if ( *(_BYTE *)(v75 + 16) != 26 )
            v75 = 0;
          v127[0] = (__int64)&v130;
          v127[1] = v131;
          v129 = v130;
          v128 = v132;
          v76 = v110;
          v105 = *(_QWORD *)(v75 - 72);
          v115 = v110 + 40;
          while ( 1 )
          {
            v77 = *(_QWORD *)(v75 + -24LL * (1 - v108) - 24);
            v78 = (_QWORD *)(*(_QWORD *)(v76 + 40) & 0xFFFFFFFFFFFFFFF8LL);
            sub_157EA20(v115, (__int64)(v78 - 3));
            v79 = (unsigned __int64 *)v78[1];
            v80 = (__int64 *)(v77 + 40);
            v81 = *v78 & 0xFFFFFFFFFFFFFFF8LL;
            v82 = *v79 & 7;
            *v79 = v81 | v82;
            *(_QWORD *)(v81 + 8) = v79;
            *v78 &= 7uLL;
            v78[1] = 0;
            sub_164BEC0((__int64)(v78 - 3), v82, v81, (__int64)v79, a3, a4, a5, a6, v83, v84, a9, a10);
            if ( v77 + 40 != (*(_QWORD *)(v77 + 40) & 0xFFFFFFFFFFFFFFF8LL) )
            {
              v85 = *(__int64 **)(v77 + 48);
              if ( (__int64 *)v115 != v80 )
              {
                sub_157EA80(v115, v77 + 40, *(_QWORD *)(v77 + 48), v77 + 40);
                if ( v80 != v85 )
                {
                  v86 = *(_QWORD *)(v77 + 40) & 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)((*v85 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v80;
                  *(_QWORD *)(v77 + 40) = *(_QWORD *)(v77 + 40) & 7LL | *v85 & 0xFFFFFFFFFFFFFFF8LL;
                  v87 = *(_QWORD *)(v76 + 40);
                  *(_QWORD *)(v86 + 8) = v115;
                  v87 &= 0xFFFFFFFFFFFFFFF8LL;
                  *v85 = v87 | *v85 & 7;
                  *(_QWORD *)(v87 + 8) = v85;
                  *(_QWORD *)(v76 + 40) = v86 | *(_QWORD *)(v76 + 40) & 7LL;
                }
              }
            }
            v88 = sub_157EBA0(v76);
            v89 = *(_QWORD *)(v88 - 72);
            v75 = v88;
            v90 = *(unsigned __int8 **)(v88 + 48);
            v131 = *(_QWORD *)(v88 + 40);
            v132 = (__int64 *)(v88 + 24);
            v125[0] = v90;
            if ( v90 )
            {
              sub_1623A60((__int64)v125, (__int64)v90, 2);
              v91 = v130;
              if ( !v130 )
                goto LABEL_139;
            }
            else
            {
              v91 = v130;
              if ( !v130 )
                goto LABEL_141;
            }
            sub_161E7C0((__int64)&v130, (__int64)v91);
LABEL_139:
            v130 = v125[0];
            if ( v125[0] )
              sub_1623210((__int64)v125, v125[0], (__int64)&v130);
LABEL_141:
            v126 = 257;
            if ( v108 )
              v105 = sub_1281C00((__int64 *)&v130, v105, v89, (__int64)v125);
            else
              v105 = sub_156D390((__int64 *)&v130, v105, v89, (__int64)v125);
            sub_1648780(v75, v89, v105);
            if ( v77 == v106 )
            {
              sub_157EE90(v77);
              v100 = sub_157E9C0(v77);
              v101 = sub_1648A60(56, 0);
              if ( v101 )
                sub_15F82E0((__int64)v101, v100, v77);
              v14 = 1;
              sub_1705150((__int64)v127);
              if ( v140 != v139 )
                _libc_free((unsigned __int64)v140);
              goto LABEL_7;
            }
            sub_157EE90(v77);
            v92 = sub_157E9C0(v77);
            v93 = sub_1648A60(56, 0);
            if ( v93 )
              sub_15F82E0((__int64)v93, v92, v77);
          }
        }
      }
      goto LABEL_58;
    }
  }
LABEL_49:
  if ( (*(_DWORD *)(v38 + 20) & 0xFFFFFFF) != 1 )
    goto LABEL_58;
  v40 = sub_157EBA0(v36);
  if ( v35 != sub_15F4DF0(v40, 0) )
    goto LABEL_58;
  v41 = v106;
  do
  {
    v42 = sub_157EBA0(v41);
    v43 = v42;
    if ( *(_BYTE *)(v42 + 16) != 26 )
      BUG();
    v44 = *(_QWORD *)(v42 - 72);
    if ( (unsigned __int8)(*(_BYTE *)(v44 + 16) - 75) <= 1u )
    {
      v45 = *(_WORD *)(v44 + 18) & 0x7FFF;
      if ( v45 == 6 || v45 == 33 )
      {
        LOBYTE(v37) = v45 == 6 || v45 == 33;
        v14 = v37;
        *(_WORD *)(v44 + 18) = sub_15FF0F0(v45) | *(_WORD *)(v44 + 18) & 0x8000;
        sub_15F89F0(v43);
      }
    }
    v41 = sub_157F0B0(v41);
  }
  while ( v41 != v110 );
  if ( v140 != v139 )
    _libc_free((unsigned __int64)v140);
  if ( !v14 )
    goto LABEL_3;
LABEL_7:
  if ( v130 )
    sub_161E7C0((__int64)&v130, (__int64)v130);
  return v14;
}
