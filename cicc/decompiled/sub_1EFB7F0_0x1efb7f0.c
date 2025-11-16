// Function: sub_1EFB7F0
// Address: 0x1efb7f0
//
unsigned __int64 __fastcall sub_1EFB7F0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  _QWORD *v3; // rax
  __int64 v4; // rcx
  unsigned __int8 *v5; // rsi
  unsigned __int8 *v6; // rsi
  unsigned __int8 *v7; // rsi
  _QWORD *v8; // rax
  __int64 v9; // rax
  __int64 v10; // rbx
  unsigned __int8 *v11; // rsi
  __int64 v12; // rax
  __int64 v13; // r9
  _QWORD *v14; // r14
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // r10
  _QWORD *v19; // rax
  _QWORD *v20; // r15
  unsigned __int64 *v21; // r14
  __int64 v22; // rax
  unsigned __int64 v23; // rcx
  __int64 v24; // rsi
  unsigned __int8 *v25; // rsi
  __int64 v26; // rax
  __int64 v27; // r15
  unsigned __int8 *v28; // rsi
  __int64 v29; // rax
  unsigned __int8 *v30; // rsi
  _QWORD *v31; // r14
  _QWORD *v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // r13
  __int64 v36; // r15
  __int64 v37; // rax
  __int64 v38; // r14
  _QWORD *v39; // r13
  __int64 v40; // rax
  unsigned __int8 *v41; // rsi
  __int64 v42; // rax
  _QWORD *v43; // rax
  _QWORD **v44; // rax
  __int64 *v45; // rax
  __int64 v46; // rsi
  unsigned __int64 *v47; // r15
  __int64 v48; // rax
  unsigned __int64 v49; // rcx
  __int64 v50; // rsi
  unsigned __int8 *v51; // rsi
  _QWORD *v52; // rax
  __int64 v53; // r10
  __int64 *v54; // r15
  __int64 v55; // rsi
  __int64 v56; // rax
  __int64 v57; // rsi
  __int64 v58; // r15
  unsigned __int8 *v59; // rsi
  _QWORD *v60; // rax
  unsigned __int64 *v61; // r15
  __int64 v62; // rax
  unsigned __int64 v63; // rsi
  __int64 v64; // rsi
  unsigned __int8 *v65; // rsi
  _QWORD *v66; // rax
  __int64 *v67; // r13
  __int64 v68; // rax
  __int64 v69; // rcx
  __int64 v70; // rsi
  unsigned __int8 *v71; // rsi
  __int64 v72; // rbx
  unsigned __int64 result; // rax
  _QWORD *v74; // r12
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // r14
  _QWORD *v78; // r15
  _QWORD *v79; // rax
  _QWORD *v80; // r14
  unsigned __int64 *v81; // r12
  __int64 v82; // rax
  unsigned __int64 v83; // rcx
  __int64 v84; // rsi
  unsigned __int8 *v85; // rsi
  __int64 v86; // rdx
  __int64 v87; // rax
  __int64 v88; // rax
  __int64 v89; // r15
  unsigned __int64 *v90; // r15
  __int64 v91; // rax
  unsigned __int64 v92; // rsi
  __int64 v93; // rsi
  unsigned __int8 *v94; // rsi
  _QWORD *v95; // rax
  unsigned __int64 *v96; // r14
  __int64 v97; // rax
  unsigned __int64 v98; // rcx
  __int64 v99; // rsi
  unsigned __int8 *v100; // rsi
  __int64 v101; // [rsp+8h] [rbp-148h]
  unsigned int v102; // [rsp+14h] [rbp-13Ch]
  __int64 v103; // [rsp+18h] [rbp-138h]
  __int64 v104; // [rsp+20h] [rbp-130h]
  __int64 v105; // [rsp+28h] [rbp-128h]
  _QWORD *v106; // [rsp+28h] [rbp-128h]
  _QWORD *v107; // [rsp+28h] [rbp-128h]
  __int64 v108; // [rsp+28h] [rbp-128h]
  __int64 v109; // [rsp+28h] [rbp-128h]
  __int64 v110; // [rsp+28h] [rbp-128h]
  __int64 v111; // [rsp+28h] [rbp-128h]
  __int64 *v112; // [rsp+30h] [rbp-120h]
  __int64 v114; // [rsp+40h] [rbp-110h]
  _QWORD *v115; // [rsp+50h] [rbp-100h]
  __int64 v116; // [rsp+50h] [rbp-100h]
  __int64 v117; // [rsp+50h] [rbp-100h]
  _QWORD *v118; // [rsp+58h] [rbp-F8h]
  unsigned __int8 *v119; // [rsp+68h] [rbp-E8h] BYREF
  __int64 v120; // [rsp+70h] [rbp-E0h]
  char *v121; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v122; // [rsp+98h] [rbp-B8h]
  __int16 v123; // [rsp+A0h] [rbp-B0h]
  unsigned __int8 *v124[2]; // [rsp+B0h] [rbp-A0h] BYREF
  __int16 v125; // [rsp+C0h] [rbp-90h]
  unsigned __int8 *v126; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v127; // [rsp+D8h] [rbp-78h]
  __int64 *v128; // [rsp+E0h] [rbp-70h]
  _QWORD *v129; // [rsp+E8h] [rbp-68h]
  __int64 v130; // [rsp+F0h] [rbp-60h]
  int v131; // [rsp+F8h] [rbp-58h]
  __int64 v132; // [rsp+100h] [rbp-50h]
  __int64 v133; // [rsp+108h] [rbp-48h]

  v1 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v114 = *(_QWORD *)(a1 - 24 * v1);
  v104 = *(_QWORD *)(a1 + 24 * (1 - v1));
  v2 = *(_QWORD *)(a1 + 24 * (2 - v1));
  v103 = *(_QWORD *)(a1 + 24 * (3 - v1));
  v3 = (_QWORD *)sub_16498A0(a1);
  v4 = *(_QWORD *)(a1 + 40);
  v5 = *(unsigned __int8 **)(a1 + 48);
  v126 = 0;
  v129 = v3;
  v130 = 0;
  v131 = 0;
  v132 = 0;
  v133 = 0;
  v118 = (_QWORD *)v4;
  v127 = v4;
  v112 = (__int64 *)(a1 + 24);
  v128 = (__int64 *)(a1 + 24);
  v124[0] = v5;
  if ( v5 )
  {
    sub_1623A60((__int64)v124, (__int64)v5, 2);
    if ( v126 )
      sub_161E7C0((__int64)&v126, (__int64)v126);
    v126 = v124[0];
    if ( v124[0] )
      sub_1623210((__int64)v124, v124[0], (__int64)&v126);
  }
  v6 = *(unsigned __int8 **)(a1 + 48);
  v124[0] = v6;
  if ( v6 )
  {
    sub_1623A60((__int64)v124, (__int64)v6, 2);
    v7 = v126;
    if ( !v126 )
      goto LABEL_9;
  }
  else
  {
    v7 = v126;
    if ( !v126 )
      goto LABEL_11;
  }
  sub_161E7C0((__int64)&v126, (__int64)v7);
LABEL_9:
  v7 = v124[0];
  v126 = v124[0];
  if ( v124[0] )
    sub_1623210((__int64)v124, v124[0], (__int64)&v126);
LABEL_11:
  v8 = *(_QWORD **)(v2 + 24);
  if ( *(_DWORD *)(v2 + 32) > 0x40u )
    v8 = (_QWORD *)*v8;
  v102 = (unsigned int)v8;
  v9 = *(_QWORD *)(*(_QWORD *)v114 + 32LL);
  if ( *(_BYTE *)(v103 + 16) == 8 )
  {
    v117 = (unsigned int)v9;
    v72 = 0;
    if ( (_DWORD)v9 )
    {
      do
      {
        v86 = v72 - (*(_DWORD *)(v103 + 20) & 0xFFFFFFF);
        if ( !sub_1593BB0(*(_QWORD *)(v103 + 24 * v86), (__int64)v7, v86, v4) )
        {
          LODWORD(v120) = v72;
          v121 = "Elt";
          v122 = v120;
          v123 = 2307;
          v87 = sub_1643350(v129);
          v88 = sub_159C470(v87, v72, 0);
          v89 = v88;
          if ( *(_BYTE *)(v114 + 16) > 0x10u || *(_BYTE *)(v88 + 16) > 0x10u )
          {
            v125 = 257;
            v74 = sub_1648A60(56, 2u);
            if ( v74 )
              sub_15FA320((__int64)v74, (_QWORD *)v114, v89, (__int64)v124, 0);
            if ( v127 )
            {
              v90 = (unsigned __int64 *)v128;
              sub_157E9D0(v127 + 40, (__int64)v74);
              v91 = v74[3];
              v92 = *v90;
              v74[4] = v90;
              v92 &= 0xFFFFFFFFFFFFFFF8LL;
              v74[3] = v92 | v91 & 7;
              *(_QWORD *)(v92 + 8) = v74 + 3;
              *v90 = *v90 & 7 | (unsigned __int64)(v74 + 3);
            }
            sub_164B780((__int64)v74, (__int64 *)&v121);
            if ( v126 )
            {
              v119 = v126;
              sub_1623A60((__int64)&v119, (__int64)v126, 2);
              v93 = v74[6];
              if ( v93 )
                sub_161E7C0((__int64)(v74 + 6), v93);
              v94 = v119;
              v74[6] = v119;
              if ( v94 )
                sub_1623210((__int64)&v119, v94, (__int64)(v74 + 6));
            }
          }
          else
          {
            v74 = (_QWORD *)sub_15A37D0((_BYTE *)v114, v88, 0);
          }
          LODWORD(v120) = v72;
          v121 = "Ptr";
          v122 = v120;
          v123 = 2307;
          v75 = sub_1643350(v129);
          v76 = sub_159C470(v75, v72, 0);
          v77 = v76;
          if ( *(_BYTE *)(v104 + 16) > 0x10u || *(_BYTE *)(v76 + 16) > 0x10u )
          {
            v125 = 257;
            v95 = sub_1648A60(56, 2u);
            v78 = v95;
            if ( v95 )
              sub_15FA320((__int64)v95, (_QWORD *)v104, v77, (__int64)v124, 0);
            if ( v127 )
            {
              v96 = (unsigned __int64 *)v128;
              sub_157E9D0(v127 + 40, (__int64)v78);
              v97 = v78[3];
              v98 = *v96;
              v78[4] = v96;
              v98 &= 0xFFFFFFFFFFFFFFF8LL;
              v78[3] = v98 | v97 & 7;
              *(_QWORD *)(v98 + 8) = v78 + 3;
              *v96 = *v96 & 7 | (unsigned __int64)(v78 + 3);
            }
            sub_164B780((__int64)v78, (__int64 *)&v121);
            if ( v126 )
            {
              v119 = v126;
              sub_1623A60((__int64)&v119, (__int64)v126, 2);
              v99 = v78[6];
              if ( v99 )
                sub_161E7C0((__int64)(v78 + 6), v99);
              v100 = v119;
              v78[6] = v119;
              if ( v100 )
                sub_1623210((__int64)&v119, v100, (__int64)(v78 + 6));
            }
          }
          else
          {
            v78 = (_QWORD *)sub_15A37D0((_BYTE *)v104, v76, 0);
          }
          v125 = 257;
          v79 = sub_1648A60(64, 2u);
          v80 = v79;
          if ( v79 )
            sub_15F9650((__int64)v79, (__int64)v74, (__int64)v78, 0, 0);
          if ( v127 )
          {
            v81 = (unsigned __int64 *)v128;
            sub_157E9D0(v127 + 40, (__int64)v80);
            v82 = v80[3];
            v83 = *v81;
            v80[4] = v81;
            v83 &= 0xFFFFFFFFFFFFFFF8LL;
            v80[3] = v83 | v82 & 7;
            *(_QWORD *)(v83 + 8) = v80 + 3;
            *v81 = *v81 & 7 | (unsigned __int64)(v80 + 3);
          }
          sub_164B780((__int64)v80, (__int64 *)v124);
          if ( v126 )
          {
            v121 = (char *)v126;
            sub_1623A60((__int64)&v121, (__int64)v126, 2);
            v84 = v80[6];
            if ( v84 )
              sub_161E7C0((__int64)(v80 + 6), v84);
            v85 = (unsigned __int8 *)v121;
            v80[6] = v121;
            if ( v85 )
              sub_1623210((__int64)&v121, v85, (__int64)(v80 + 6));
          }
          v7 = (unsigned __int8 *)v102;
          sub_15F9450((__int64)v80, v102);
        }
        ++v72;
      }
      while ( v117 != v72 );
    }
  }
  else if ( (_DWORD)v9 )
  {
    v10 = 0;
    v101 = (unsigned int)v9;
    while ( 1 )
    {
      LODWORD(v120) = v10;
      v121 = "Mask";
      v122 = v120;
      v123 = 2307;
      v33 = sub_1643350(v129);
      v34 = sub_159C470(v33, v10, 0);
      v35 = v34;
      if ( *(_BYTE *)(v103 + 16) > 0x10u || *(_BYTE *)(v34 + 16) > 0x10u )
      {
        v125 = 257;
        v66 = sub_1648A60(56, 2u);
        v36 = (__int64)v66;
        if ( v66 )
          sub_15FA320((__int64)v66, (_QWORD *)v103, v35, (__int64)v124, 0);
        if ( v127 )
        {
          v67 = v128;
          sub_157E9D0(v127 + 40, v36);
          v68 = *(_QWORD *)(v36 + 24);
          v69 = *v67;
          *(_QWORD *)(v36 + 32) = v67;
          v69 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v36 + 24) = v69 | v68 & 7;
          *(_QWORD *)(v69 + 8) = v36 + 24;
          *v67 = *v67 & 7 | (v36 + 24);
        }
        sub_164B780(v36, (__int64 *)&v121);
        if ( v126 )
        {
          v119 = v126;
          sub_1623A60((__int64)&v119, (__int64)v126, 2);
          v70 = *(_QWORD *)(v36 + 48);
          if ( v70 )
            sub_161E7C0(v36 + 48, v70);
          v71 = v119;
          *(_QWORD *)(v36 + 48) = v119;
          if ( v71 )
            sub_1623210((__int64)&v119, v71, v36 + 48);
        }
      }
      else
      {
        v36 = sub_15A37D0((_BYTE *)v103, v34, 0);
      }
      LODWORD(v120) = v10;
      v121 = "ToStore";
      v123 = 2307;
      v122 = v120;
      v37 = sub_15A0680(*(_QWORD *)v36, 1, 0);
      v38 = v37;
      if ( *(_BYTE *)(v36 + 16) > 0x10u || *(_BYTE *)(v37 + 16) > 0x10u )
      {
        v125 = 257;
        v43 = sub_1648A60(56, 2u);
        v39 = v43;
        if ( v43 )
        {
          v116 = (__int64)v43;
          v44 = *(_QWORD ***)v36;
          if ( *(_BYTE *)(*(_QWORD *)v36 + 8LL) == 16 )
          {
            v106 = v44[4];
            v45 = (__int64 *)sub_1643320(*v44);
            v46 = (__int64)sub_16463B0(v45, (unsigned int)v106);
          }
          else
          {
            v46 = sub_1643320(*v44);
          }
          sub_15FEC10((__int64)v39, v46, 51, 32, v36, v38, (__int64)v124, 0);
        }
        else
        {
          v116 = 0;
        }
        if ( v127 )
        {
          v47 = (unsigned __int64 *)v128;
          sub_157E9D0(v127 + 40, (__int64)v39);
          v48 = v39[3];
          v49 = *v47;
          v39[4] = v47;
          v49 &= 0xFFFFFFFFFFFFFFF8LL;
          v39[3] = v49 | v48 & 7;
          *(_QWORD *)(v49 + 8) = v39 + 3;
          *v47 = *v47 & 7 | (unsigned __int64)(v39 + 3);
        }
        sub_164B780(v116, (__int64 *)&v121);
        if ( v126 )
        {
          v119 = v126;
          sub_1623A60((__int64)&v119, (__int64)v126, 2);
          v50 = v39[6];
          if ( v50 )
            sub_161E7C0((__int64)(v39 + 6), v50);
          v51 = v119;
          v39[6] = v119;
          if ( v51 )
            sub_1623210((__int64)&v119, v51, (__int64)(v39 + 6));
        }
      }
      else
      {
        v39 = (_QWORD *)sub_15A37B0(0x20u, (_QWORD *)v36, (_QWORD *)v37, 0);
      }
      v124[0] = "cond.store";
      v125 = 259;
      v40 = sub_157FBF0(v118, v112, (__int64)v124);
      v128 = v112;
      v115 = (_QWORD *)v40;
      v41 = *(unsigned __int8 **)(a1 + 48);
      v42 = *(_QWORD *)(a1 + 40);
      v124[0] = v41;
      v127 = v42;
      if ( v41 )
      {
        sub_1623A60((__int64)v124, (__int64)v41, 2);
        v11 = v126;
        if ( !v126 )
          goto LABEL_18;
      }
      else
      {
        v11 = v126;
        if ( !v126 )
          goto LABEL_20;
      }
      sub_161E7C0((__int64)&v126, (__int64)v11);
LABEL_18:
      v126 = v124[0];
      if ( v124[0] )
        sub_1623210((__int64)v124, v124[0], (__int64)&v126);
LABEL_20:
      LODWORD(v120) = v10;
      v121 = "Elt";
      v122 = v120;
      v123 = 2307;
      v12 = sub_1643350(v129);
      v13 = sub_159C470(v12, v10, 0);
      if ( *(_BYTE *)(v114 + 16) > 0x10u || *(_BYTE *)(v13 + 16) > 0x10u )
      {
        v111 = v13;
        v125 = 257;
        v60 = sub_1648A60(56, 2u);
        v14 = v60;
        if ( v60 )
          sub_15FA320((__int64)v60, (_QWORD *)v114, v111, (__int64)v124, 0);
        if ( v127 )
        {
          v61 = (unsigned __int64 *)v128;
          sub_157E9D0(v127 + 40, (__int64)v14);
          v62 = v14[3];
          v63 = *v61;
          v14[4] = v61;
          v63 &= 0xFFFFFFFFFFFFFFF8LL;
          v14[3] = v63 | v62 & 7;
          *(_QWORD *)(v63 + 8) = v14 + 3;
          *v61 = *v61 & 7 | (unsigned __int64)(v14 + 3);
        }
        sub_164B780((__int64)v14, (__int64 *)&v121);
        if ( v126 )
        {
          v119 = v126;
          sub_1623A60((__int64)&v119, (__int64)v126, 2);
          v64 = v14[6];
          if ( v64 )
            sub_161E7C0((__int64)(v14 + 6), v64);
          v65 = v119;
          v14[6] = v119;
          if ( v65 )
            sub_1623210((__int64)&v119, v65, (__int64)(v14 + 6));
        }
      }
      else
      {
        v14 = (_QWORD *)sub_15A37D0((_BYTE *)v114, v13, 0);
      }
      LODWORD(v120) = v10;
      v121 = "Ptr";
      v123 = 2307;
      v122 = v120;
      v15 = sub_1643350(v129);
      v16 = sub_159C470(v15, v10, 0);
      v17 = v16;
      if ( *(_BYTE *)(v104 + 16) > 0x10u || *(_BYTE *)(v16 + 16) > 0x10u )
      {
        v125 = 257;
        v52 = sub_1648A60(56, 2u);
        v53 = (__int64)v52;
        if ( v52 )
        {
          v107 = v52;
          sub_15FA320((__int64)v52, (_QWORD *)v104, v17, (__int64)v124, 0);
          v53 = (__int64)v107;
        }
        if ( v127 )
        {
          v54 = v128;
          v108 = v53;
          sub_157E9D0(v127 + 40, v53);
          v53 = v108;
          v55 = *v54;
          v56 = *(_QWORD *)(v108 + 24);
          *(_QWORD *)(v108 + 32) = v54;
          v55 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v108 + 24) = v55 | v56 & 7;
          *(_QWORD *)(v55 + 8) = v108 + 24;
          *v54 = *v54 & 7 | (v108 + 24);
        }
        v109 = v53;
        sub_164B780(v53, (__int64 *)&v121);
        v18 = v109;
        if ( v126 )
        {
          v119 = v126;
          sub_1623A60((__int64)&v119, (__int64)v126, 2);
          v18 = v109;
          v57 = *(_QWORD *)(v109 + 48);
          v58 = v109 + 48;
          if ( v57 )
          {
            sub_161E7C0(v109 + 48, v57);
            v18 = v109;
          }
          v59 = v119;
          *(_QWORD *)(v18 + 48) = v119;
          if ( v59 )
          {
            v110 = v18;
            sub_1623210((__int64)&v119, v59, v58);
            v18 = v110;
          }
        }
      }
      else
      {
        v18 = sub_15A37D0((_BYTE *)v104, v16, 0);
      }
      v105 = v18;
      v125 = 257;
      v19 = sub_1648A60(64, 2u);
      v20 = v19;
      if ( v19 )
        sub_15F9650((__int64)v19, (__int64)v14, v105, 0, 0);
      if ( v127 )
      {
        v21 = (unsigned __int64 *)v128;
        sub_157E9D0(v127 + 40, (__int64)v20);
        v22 = v20[3];
        v23 = *v21;
        v20[4] = v21;
        v23 &= 0xFFFFFFFFFFFFFFF8LL;
        v20[3] = v23 | v22 & 7;
        *(_QWORD *)(v23 + 8) = v20 + 3;
        *v21 = *v21 & 7 | (unsigned __int64)(v20 + 3);
      }
      sub_164B780((__int64)v20, (__int64 *)v124);
      if ( v126 )
      {
        v121 = (char *)v126;
        sub_1623A60((__int64)&v121, (__int64)v126, 2);
        v24 = v20[6];
        if ( v24 )
          sub_161E7C0((__int64)(v20 + 6), v24);
        v25 = (unsigned __int8 *)v121;
        v20[6] = v121;
        if ( v25 )
          sub_1623210((__int64)&v121, v25, (__int64)(v20 + 6));
      }
      sub_15F9450((__int64)v20, v102);
      v124[0] = (unsigned __int8 *)"else";
      v125 = 259;
      v26 = sub_157FBF0(v115, v112, (__int64)v124);
      v128 = v112;
      v27 = v26;
      v28 = *(unsigned __int8 **)(a1 + 48);
      v29 = *(_QWORD *)(a1 + 40);
      v124[0] = v28;
      v127 = v29;
      if ( v28 )
      {
        sub_1623A60((__int64)v124, (__int64)v28, 2);
        v30 = v126;
        if ( !v126 )
          goto LABEL_38;
      }
      else
      {
        v30 = v126;
        if ( !v126 )
          goto LABEL_40;
      }
      sub_161E7C0((__int64)&v126, (__int64)v30);
LABEL_38:
      v126 = v124[0];
      if ( v124[0] )
        sub_1623210((__int64)v124, v124[0], (__int64)&v126);
LABEL_40:
      v31 = (_QWORD *)sub_157EBA0((__int64)v118);
      v32 = sub_1648A60(56, 3u);
      if ( v32 )
        sub_15F83E0((__int64)v32, (__int64)v115, v27, (__int64)v39, (__int64)v31);
      ++v10;
      sub_15F20C0(v31);
      if ( v101 == v10 )
        break;
      v118 = (_QWORD *)v27;
    }
  }
  result = sub_15F20C0((_QWORD *)a1);
  if ( v126 )
    return sub_161E7C0((__int64)&v126, (__int64)v126);
  return result;
}
