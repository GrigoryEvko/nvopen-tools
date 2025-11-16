// Function: sub_1EFCAC0
// Address: 0x1efcac0
//
__int64 __fastcall sub_1EFCAC0(__int64 a1)
{
  _QWORD *v1; // r14
  __int64 v2; // rax
  __int64 v3; // rdx
  _QWORD *v4; // r13
  __int64 v5; // rbx
  _QWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  unsigned __int8 *v9; // rsi
  _QWORD *v10; // rax
  unsigned __int8 *v11; // rsi
  unsigned __int8 *v12; // rsi
  unsigned int v13; // eax
  __int64 v14; // rsi
  __int64 **v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r15
  unsigned __int8 *v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rbx
  _QWORD *v21; // r14
  __int64 v22; // rax
  __int64 v23; // rax
  _QWORD *v24; // rax
  _QWORD *v25; // rbx
  unsigned __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rdx
  unsigned __int8 *v30; // rsi
  __int64 v31; // rax
  __int64 v32; // rbx
  unsigned __int8 *v33; // rsi
  _QWORD *v34; // rax
  unsigned __int8 *v35; // rsi
  _QWORD *v36; // rax
  _QWORD *v37; // r8
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // r13
  __int64 v41; // rbx
  __int64 v42; // rax
  __int64 v43; // r14
  _QWORD *v44; // r13
  __int64 v45; // rax
  unsigned __int8 *v46; // rsi
  _QWORD *v47; // rax
  _QWORD *v48; // rax
  _QWORD **v49; // rax
  __int64 *v50; // rax
  __int64 v51; // rsi
  unsigned __int64 *v52; // rbx
  __int64 v53; // rax
  unsigned __int64 v54; // rcx
  __int64 v55; // rsi
  unsigned __int8 *v56; // rsi
  _QWORD *v57; // rax
  unsigned __int64 *v58; // rbx
  __int64 v59; // rax
  unsigned __int64 v60; // rcx
  __int64 v61; // rsi
  unsigned __int8 *v62; // rsi
  _QWORD *v63; // rax
  __int64 *v64; // r13
  __int64 v65; // rax
  __int64 v66; // rcx
  __int64 v67; // rsi
  unsigned __int8 *v68; // rsi
  __int64 result; // rax
  unsigned __int8 *v70; // rsi
  _QWORD *v71; // rax
  _QWORD *v72; // r12
  unsigned __int64 *v73; // rbx
  __int64 v74; // rax
  unsigned __int64 v75; // rcx
  __int64 v76; // rsi
  unsigned __int8 *v77; // rsi
  __int64 v78; // rax
  __int64 v79; // r15
  __int64 *v80; // r12
  __int64 v81; // rax
  __int64 v82; // rcx
  __int64 v83; // rsi
  __int64 v84; // rbx
  unsigned __int8 **v85; // rcx
  _QWORD *v86; // r14
  __int64 v87; // rax
  __int64 v88; // rax
  _QWORD *v89; // rax
  _QWORD *v90; // r15
  unsigned __int64 v91; // rsi
  __int64 v92; // rax
  __int64 v93; // rsi
  __int64 v94; // rdx
  unsigned __int8 *v95; // rsi
  __int64 v96; // rdx
  __int64 v97; // rax
  __int64 v98; // rax
  __int64 v99; // r15
  _QWORD *v100; // rax
  unsigned __int64 *v101; // r15
  __int64 v102; // rax
  unsigned __int64 v103; // rsi
  __int64 v104; // rsi
  __int64 v105; // rdx
  unsigned __int8 *v106; // rsi
  __int64 v107; // [rsp+8h] [rbp-138h]
  unsigned int v108; // [rsp+14h] [rbp-12Ch]
  __int64 *v109; // [rsp+18h] [rbp-128h]
  unsigned __int8 *v110; // [rsp+20h] [rbp-120h]
  __int64 v111; // [rsp+28h] [rbp-118h]
  __int64 v112; // [rsp+30h] [rbp-110h]
  unsigned __int64 *v113; // [rsp+30h] [rbp-110h]
  _QWORD *v114; // [rsp+30h] [rbp-110h]
  __int64 *v115; // [rsp+38h] [rbp-108h]
  __int64 v116; // [rsp+50h] [rbp-F0h]
  _QWORD *v117; // [rsp+58h] [rbp-E8h]
  unsigned __int64 v118; // [rsp+58h] [rbp-E8h]
  _QWORD *v119; // [rsp+60h] [rbp-E0h]
  __int64 v120; // [rsp+60h] [rbp-E0h]
  __int64 v121; // [rsp+60h] [rbp-E0h]
  unsigned __int64 *v122; // [rsp+60h] [rbp-E0h]
  __int64 v123; // [rsp+68h] [rbp-D8h]
  unsigned __int8 *v124; // [rsp+78h] [rbp-C8h] BYREF
  __int64 v125[2]; // [rsp+80h] [rbp-C0h] BYREF
  __int16 v126; // [rsp+90h] [rbp-B0h]
  unsigned __int8 *v127[2]; // [rsp+A0h] [rbp-A0h] BYREF
  __int16 v128; // [rsp+B0h] [rbp-90h]
  unsigned __int8 *v129; // [rsp+C0h] [rbp-80h] BYREF
  _QWORD *v130; // [rsp+C8h] [rbp-78h]
  unsigned __int64 *v131; // [rsp+D0h] [rbp-70h]
  _QWORD *v132; // [rsp+D8h] [rbp-68h]
  __int64 v133; // [rsp+E0h] [rbp-60h]
  int v134; // [rsp+E8h] [rbp-58h]
  __int64 v135; // [rsp+F0h] [rbp-50h]
  __int64 v136; // [rsp+F8h] [rbp-48h]

  v1 = (_QWORD *)a1;
  v2 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v116 = *(_QWORD *)(a1 - 24 * v2);
  v3 = *(_QWORD *)(a1 + 24 * (2 - v2));
  v4 = *(_QWORD **)(v3 + 24);
  if ( *(_DWORD *)(v3 + 32) > 0x40u )
    v4 = (_QWORD *)*v4;
  v5 = *(_QWORD *)v116;
  if ( *(_BYTE *)(*(_QWORD *)v116 + 8LL) != 16 )
    BUG();
  v110 = *(unsigned __int8 **)(a1 + 24 * (1 - v2));
  v111 = *(_QWORD *)(a1 + 24 * (3 - v2));
  v109 = *(__int64 **)(v5 + 24);
  v6 = (_QWORD *)sub_16498A0(a1);
  v9 = *(unsigned __int8 **)(a1 + 48);
  v129 = 0;
  v132 = v6;
  v10 = *(_QWORD **)(a1 + 40);
  v133 = 0;
  v117 = v10;
  v130 = v10;
  v134 = 0;
  v135 = 0;
  v136 = 0;
  v115 = (__int64 *)(a1 + 24);
  v131 = (unsigned __int64 *)(a1 + 24);
  v127[0] = v9;
  if ( v9 )
  {
    sub_1623A60((__int64)v127, (__int64)v9, 2);
    if ( v129 )
      sub_161E7C0((__int64)&v129, (__int64)v129);
    v129 = v127[0];
    if ( v127[0] )
      sub_1623210((__int64)v127, v127[0], (__int64)&v129);
  }
  v11 = *(unsigned __int8 **)(a1 + 48);
  v127[0] = v11;
  if ( v11 )
  {
    sub_1623A60((__int64)v127, (__int64)v11, 2);
    v12 = v129;
    if ( !v129 )
      goto LABEL_12;
    goto LABEL_11;
  }
  v12 = v129;
  if ( v129 )
  {
LABEL_11:
    sub_161E7C0((__int64)&v129, (__int64)v12);
LABEL_12:
    v12 = v127[0];
    v129 = v127[0];
    if ( v127[0] )
      sub_1623210((__int64)v127, v127[0], (__int64)&v129);
    if ( *(_BYTE *)(v111 + 16) > 0x10u )
      goto LABEL_15;
    goto LABEL_95;
  }
  if ( *(_BYTE *)(v111 + 16) > 0x10u )
    goto LABEL_15;
LABEL_95:
  if ( sub_1596070(v111, (__int64)v12, v7, v8) )
  {
    v128 = 257;
    v71 = sub_1648A60(64, 2u);
    v72 = v71;
    if ( v71 )
      sub_15F9650((__int64)v71, v116, (__int64)v110, 0, 0);
    if ( v130 )
    {
      v73 = v131;
      sub_157E9D0((__int64)(v130 + 5), (__int64)v72);
      v74 = v72[3];
      v75 = *v73;
      v72[4] = v73;
      v75 &= 0xFFFFFFFFFFFFFFF8LL;
      v72[3] = v75 | v74 & 7;
      *(_QWORD *)(v75 + 8) = v72 + 3;
      *v73 = *v73 & 7 | (unsigned __int64)(v72 + 3);
    }
    sub_164B780((__int64)v72, (__int64 *)v127);
    if ( v129 )
    {
      v125[0] = (__int64)v129;
      sub_1623A60((__int64)v125, (__int64)v129, 2);
      v76 = v72[6];
      if ( v76 )
        sub_161E7C0((__int64)(v72 + 6), v76);
      v77 = (unsigned __int8 *)v125[0];
      v72[6] = v125[0];
      if ( v77 )
        sub_1623210((__int64)v125, v77, (__int64)(v72 + 6));
    }
    sub_15F9450((__int64)v72, (unsigned int)v4);
    result = sub_15F20C0((_QWORD *)a1);
LABEL_106:
    v70 = v129;
    if ( v129 )
      return sub_161E7C0((__int64)&v129, (__int64)v70);
    return result;
  }
LABEL_15:
  v13 = (unsigned int)sub_16431D0(v5) >> 3;
  if ( v13 >= (unsigned int)v4 )
    LODWORD(v4) = v13;
  v108 = (unsigned int)v4;
  v14 = (__int64)v110;
  v15 = (__int64 **)sub_1647190(v109, *(_DWORD *)(*(_QWORD *)v110 + 8LL) >> 8);
  v126 = 257;
  if ( v15 != *(__int64 ***)v110 )
  {
    if ( v110[16] > 0x10u )
    {
      v128 = 257;
      v78 = sub_15FDBD0(47, (__int64)v110, (__int64)v15, (__int64)v127, 0);
      v110 = (unsigned __int8 *)v78;
      v79 = v78;
      if ( v130 )
      {
        v80 = (__int64 *)v131;
        sub_157E9D0((__int64)(v130 + 5), v78);
        v81 = *(_QWORD *)(v79 + 24);
        v82 = *v80;
        *(_QWORD *)(v79 + 32) = v80;
        v82 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v79 + 24) = v82 | v81 & 7;
        *(_QWORD *)(v82 + 8) = v79 + 24;
        *v80 = *v80 & 7 | (v79 + 24);
      }
      sub_164B780((__int64)v110, v125);
      v14 = (__int64)v129;
      if ( v129 )
      {
        v124 = v129;
        sub_1623A60((__int64)&v124, (__int64)v129, 2);
        v83 = *((_QWORD *)v110 + 6);
        if ( v83 )
          sub_161E7C0((__int64)(v110 + 48), v83);
        v14 = (__int64)v124;
        *((_QWORD *)v110 + 6) = v124;
        if ( v14 )
          sub_1623210((__int64)&v124, (unsigned __int8 *)v14, (__int64)(v110 + 48));
      }
    }
    else
    {
      v110 = (unsigned __int8 *)sub_15A46C0(47, (__int64 ***)v110, v15, 0);
    }
  }
  v16 = *(_QWORD *)(v5 + 32);
  if ( *(_BYTE *)(v111 + 16) == 8 )
  {
    v84 = 0;
    v123 = (unsigned int)v16;
    v85 = &v129;
    if ( (_DWORD)v16 )
    {
      do
      {
        v96 = v84 - (*(_DWORD *)(v111 + 20) & 0xFFFFFFF);
        if ( !sub_1593BB0(*(_QWORD *)(v111 + 24 * v96), v14, v96, (__int64)v85) )
        {
          v126 = 257;
          v97 = sub_1643350(v132);
          v98 = sub_159C470(v97, v84, 0);
          v99 = v98;
          if ( *(_BYTE *)(v116 + 16) > 0x10u || *(_BYTE *)(v98 + 16) > 0x10u )
          {
            v128 = 257;
            v100 = sub_1648A60(56, 2u);
            v86 = v100;
            if ( v100 )
              sub_15FA320((__int64)v100, (_QWORD *)v116, v99, (__int64)v127, 0);
            if ( v130 )
            {
              v101 = v131;
              sub_157E9D0((__int64)(v130 + 5), (__int64)v86);
              v102 = v86[3];
              v103 = *v101;
              v86[4] = v101;
              v103 &= 0xFFFFFFFFFFFFFFF8LL;
              v86[3] = v103 | v102 & 7;
              *(_QWORD *)(v103 + 8) = v86 + 3;
              *v101 = *v101 & 7 | (unsigned __int64)(v86 + 3);
            }
            sub_164B780((__int64)v86, v125);
            if ( v129 )
            {
              v124 = v129;
              sub_1623A60((__int64)&v124, (__int64)v129, 2);
              v104 = v86[6];
              v105 = (__int64)(v86 + 6);
              if ( v104 )
              {
                sub_161E7C0((__int64)(v86 + 6), v104);
                v105 = (__int64)(v86 + 6);
              }
              v106 = v124;
              v86[6] = v124;
              if ( v106 )
                sub_1623210((__int64)&v124, v106, v105);
            }
          }
          else
          {
            v86 = (_QWORD *)sub_15A37D0((_BYTE *)v116, v98, 0);
          }
          v128 = 257;
          v87 = sub_1643350(v132);
          v88 = sub_159C470(v87, v84, 0);
          v121 = sub_17CEC00((__int64 *)&v129, (__int64)v109, v110, v88, (__int64 *)v127);
          v128 = 257;
          v89 = sub_1648A60(64, 2u);
          v90 = v89;
          if ( v89 )
            sub_15F9650((__int64)v89, (__int64)v86, v121, 0, 0);
          if ( v130 )
          {
            v122 = v131;
            sub_157E9D0((__int64)(v130 + 5), (__int64)v90);
            v91 = *v122;
            v92 = v90[3] & 7LL;
            v90[4] = v122;
            v91 &= 0xFFFFFFFFFFFFFFF8LL;
            v90[3] = v91 | v92;
            *(_QWORD *)(v91 + 8) = v90 + 3;
            *v122 = *v122 & 7 | (unsigned __int64)(v90 + 3);
          }
          sub_164B780((__int64)v90, (__int64 *)v127);
          if ( v129 )
          {
            v125[0] = (__int64)v129;
            sub_1623A60((__int64)v125, (__int64)v129, 2);
            v93 = v90[6];
            v94 = (__int64)(v90 + 6);
            if ( v93 )
            {
              sub_161E7C0((__int64)(v90 + 6), v93);
              v94 = (__int64)(v90 + 6);
            }
            v95 = (unsigned __int8 *)v125[0];
            v90[6] = v125[0];
            if ( v95 )
              sub_1623210((__int64)v125, v95, v94);
          }
          v14 = (unsigned int)v4;
          sub_15F9450((__int64)v90, (unsigned int)v4);
        }
        ++v84;
      }
      while ( v84 != v123 );
      v1 = (_QWORD *)a1;
    }
    result = sub_15F20C0(v1);
    goto LABEL_106;
  }
  if ( !(_DWORD)v16 )
    goto LABEL_90;
  v17 = 0;
  v107 = (unsigned int)v16;
  while ( 1 )
  {
    v126 = 257;
    v38 = sub_1643350(v132);
    v39 = sub_159C470(v38, v17, 0);
    v40 = v39;
    if ( *(_BYTE *)(v111 + 16) > 0x10u || *(_BYTE *)(v39 + 16) > 0x10u )
    {
      v128 = 257;
      v63 = sub_1648A60(56, 2u);
      v41 = (__int64)v63;
      if ( v63 )
        sub_15FA320((__int64)v63, (_QWORD *)v111, v40, (__int64)v127, 0);
      if ( v130 )
      {
        v64 = (__int64 *)v131;
        sub_157E9D0((__int64)(v130 + 5), v41);
        v65 = *(_QWORD *)(v41 + 24);
        v66 = *v64;
        *(_QWORD *)(v41 + 32) = v64;
        v66 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v41 + 24) = v66 | v65 & 7;
        *(_QWORD *)(v66 + 8) = v41 + 24;
        *v64 = *v64 & 7 | (v41 + 24);
      }
      sub_164B780(v41, v125);
      if ( v129 )
      {
        v124 = v129;
        sub_1623A60((__int64)&v124, (__int64)v129, 2);
        v67 = *(_QWORD *)(v41 + 48);
        if ( v67 )
          sub_161E7C0(v41 + 48, v67);
        v68 = v124;
        *(_QWORD *)(v41 + 48) = v124;
        if ( v68 )
          sub_1623210((__int64)&v124, v68, v41 + 48);
      }
    }
    else
    {
      v41 = sub_15A37D0((_BYTE *)v111, v39, 0);
    }
    v126 = 257;
    v42 = sub_15A0680(*(_QWORD *)v41, 1, 0);
    v43 = v42;
    if ( *(_BYTE *)(v41 + 16) > 0x10u || *(_BYTE *)(v42 + 16) > 0x10u )
    {
      v128 = 257;
      v48 = sub_1648A60(56, 2u);
      v44 = v48;
      if ( v48 )
      {
        v120 = (__int64)v48;
        v49 = *(_QWORD ***)v41;
        if ( *(_BYTE *)(*(_QWORD *)v41 + 8LL) == 16 )
        {
          v114 = v49[4];
          v50 = (__int64 *)sub_1643320(*v49);
          v51 = (__int64)sub_16463B0(v50, (unsigned int)v114);
        }
        else
        {
          v51 = sub_1643320(*v49);
        }
        sub_15FEC10((__int64)v44, v51, 51, 32, v41, v43, (__int64)v127, 0);
      }
      else
      {
        v120 = 0;
      }
      if ( v130 )
      {
        v52 = v131;
        sub_157E9D0((__int64)(v130 + 5), (__int64)v44);
        v53 = v44[3];
        v54 = *v52;
        v44[4] = v52;
        v54 &= 0xFFFFFFFFFFFFFFF8LL;
        v44[3] = v54 | v53 & 7;
        *(_QWORD *)(v54 + 8) = v44 + 3;
        *v52 = *v52 & 7 | (unsigned __int64)(v44 + 3);
      }
      sub_164B780(v120, v125);
      if ( v129 )
      {
        v124 = v129;
        sub_1623A60((__int64)&v124, (__int64)v129, 2);
        v55 = v44[6];
        if ( v55 )
          sub_161E7C0((__int64)(v44 + 6), v55);
        v56 = v124;
        v44[6] = v124;
        if ( v56 )
          sub_1623210((__int64)&v124, v56, (__int64)(v44 + 6));
      }
    }
    else
    {
      v44 = (_QWORD *)sub_15A37B0(0x20u, (_QWORD *)v41, (_QWORD *)v42, 0);
    }
    v127[0] = "cond.store";
    v128 = 259;
    v45 = sub_157FBF0(v117, v115, (__int64)v127);
    v131 = (unsigned __int64 *)(a1 + 24);
    v119 = (_QWORD *)v45;
    v46 = *(unsigned __int8 **)(a1 + 48);
    v47 = *(_QWORD **)(a1 + 40);
    v127[0] = v46;
    v130 = v47;
    if ( v46 )
    {
      sub_1623A60((__int64)v127, (__int64)v46, 2);
      v18 = v129;
      if ( !v129 )
        goto LABEL_25;
    }
    else
    {
      v18 = v129;
      if ( !v129 )
        goto LABEL_27;
    }
    sub_161E7C0((__int64)&v129, (__int64)v18);
LABEL_25:
    v129 = v127[0];
    if ( v127[0] )
      sub_1623210((__int64)v127, v127[0], (__int64)&v129);
LABEL_27:
    v126 = 257;
    v19 = sub_1643350(v132);
    v20 = sub_159C470(v19, v17, 0);
    if ( *(_BYTE *)(v116 + 16) > 0x10u || *(_BYTE *)(v20 + 16) > 0x10u )
    {
      v128 = 257;
      v57 = sub_1648A60(56, 2u);
      v21 = v57;
      if ( v57 )
        sub_15FA320((__int64)v57, (_QWORD *)v116, v20, (__int64)v127, 0);
      if ( v130 )
      {
        v58 = v131;
        sub_157E9D0((__int64)(v130 + 5), (__int64)v21);
        v59 = v21[3];
        v60 = *v58;
        v21[4] = v58;
        v60 &= 0xFFFFFFFFFFFFFFF8LL;
        v21[3] = v60 | v59 & 7;
        *(_QWORD *)(v60 + 8) = v21 + 3;
        *v58 = *v58 & 7 | (unsigned __int64)(v21 + 3);
      }
      sub_164B780((__int64)v21, v125);
      if ( v129 )
      {
        v124 = v129;
        sub_1623A60((__int64)&v124, (__int64)v129, 2);
        v61 = v21[6];
        if ( v61 )
          sub_161E7C0((__int64)(v21 + 6), v61);
        v62 = v124;
        v21[6] = v124;
        if ( v62 )
          sub_1623210((__int64)&v124, v62, (__int64)(v21 + 6));
      }
    }
    else
    {
      v21 = (_QWORD *)sub_15A37D0((_BYTE *)v116, v20, 0);
    }
    v128 = 257;
    v22 = sub_1643350(v132);
    v23 = sub_159C470(v22, v17, 0);
    v112 = sub_17CEC00((__int64 *)&v129, (__int64)v109, v110, v23, (__int64 *)v127);
    v128 = 257;
    v24 = sub_1648A60(64, 2u);
    v25 = v24;
    if ( v24 )
      sub_15F9650((__int64)v24, (__int64)v21, v112, 0, 0);
    if ( v130 )
    {
      v113 = v131;
      sub_157E9D0((__int64)(v130 + 5), (__int64)v25);
      v26 = *v113;
      v27 = v25[3] & 7LL;
      v25[4] = v113;
      v26 &= 0xFFFFFFFFFFFFFFF8LL;
      v25[3] = v26 | v27;
      *(_QWORD *)(v26 + 8) = v25 + 3;
      *v113 = *v113 & 7 | (unsigned __int64)(v25 + 3);
    }
    sub_164B780((__int64)v25, (__int64 *)v127);
    if ( v129 )
    {
      v125[0] = (__int64)v129;
      sub_1623A60((__int64)v125, (__int64)v129, 2);
      v28 = v25[6];
      v29 = (__int64)(v25 + 6);
      if ( v28 )
      {
        sub_161E7C0((__int64)(v25 + 6), v28);
        v29 = (__int64)(v25 + 6);
      }
      v30 = (unsigned __int8 *)v125[0];
      v25[6] = v125[0];
      if ( v30 )
        sub_1623210((__int64)v125, v30, v29);
    }
    sub_15F9450((__int64)v25, v108);
    v127[0] = (unsigned __int8 *)"else";
    v128 = 259;
    v31 = sub_157FBF0(v119, v115, (__int64)v127);
    v131 = (unsigned __int64 *)(a1 + 24);
    v32 = v31;
    v33 = *(unsigned __int8 **)(a1 + 48);
    v34 = *(_QWORD **)(a1 + 40);
    v127[0] = v33;
    v130 = v34;
    if ( v33 )
    {
      sub_1623A60((__int64)v127, (__int64)v33, 2);
      v35 = v129;
      if ( !v129 )
        goto LABEL_42;
    }
    else
    {
      v35 = v129;
      if ( !v129 )
        goto LABEL_44;
    }
    sub_161E7C0((__int64)&v129, (__int64)v35);
LABEL_42:
    v129 = v127[0];
    if ( v127[0] )
      sub_1623210((__int64)v127, v127[0], (__int64)&v129);
LABEL_44:
    v118 = sub_157EBA0((__int64)v117);
    v36 = sub_1648A60(56, 3u);
    v37 = (_QWORD *)v118;
    if ( v36 )
    {
      sub_15F83E0((__int64)v36, (__int64)v119, v32, (__int64)v44, v118);
      v37 = (_QWORD *)v118;
    }
    ++v17;
    sub_15F20C0(v37);
    if ( v107 == v17 )
      break;
    v117 = (_QWORD *)v32;
  }
  v1 = (_QWORD *)a1;
LABEL_90:
  result = sub_15F20C0(v1);
  v70 = v129;
  if ( v129 )
    return sub_161E7C0((__int64)&v129, (__int64)v70);
  return result;
}
