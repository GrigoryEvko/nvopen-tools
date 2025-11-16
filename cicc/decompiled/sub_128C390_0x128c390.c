// Function: sub_128C390
// Address: 0x128c390
//
__int64 __fastcall sub_128C390(__int64 a1, __int64 a2, unsigned __int8 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rbx
  __int64 v8; // rdx
  _DWORD *v9; // r14
  __int64 *v10; // rsi
  __int64 *v11; // r12
  __int64 v12; // r15
  int v13; // esi
  char v14; // al
  __int64 v15; // rax
  __int64 *v16; // r13
  __int64 *v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 *v20; // r8
  __int64 v21; // r9
  __int64 *v22; // rdi
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // r15
  char *v27; // rax
  __int64 v28; // r13
  __int64 v29; // rax
  int v30; // eax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 *v33; // r10
  __int64 v34; // rax
  __int64 v35; // r9
  __int64 v36; // rdi
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 *v39; // r13
  __int64 v40; // rax
  __int64 v41; // rdi
  char v42; // al
  __int64 *v43; // rdi
  char *v44; // rax
  unsigned __int8 v45; // r9
  char *v46; // rax
  __int64 v47; // rax
  __int64 v48; // rdx
  unsigned int v49; // r8d
  __int64 v50; // r9
  __int64 v51; // rdi
  __int64 v52; // r9
  __int64 v53; // rax
  __int64 v54; // rsi
  const char *v55; // rsi
  __int64 v56; // r13
  const char *v57; // rsi
  __int64 v58; // r13
  __int64 v59; // rax
  __int64 v60; // r13
  __int64 v61; // rax
  __int64 v62; // rdi
  __int64 v63; // r9
  __int64 *v64; // r15
  __int64 v65; // rcx
  __int64 v66; // rax
  __int64 v67; // rsi
  __int64 v68; // r13
  __int64 v69; // rsi
  __int64 v70; // rax
  __int64 *v71; // r10
  __int64 v72; // r9
  __int64 v73; // rdi
  __int64 *v74; // rdx
  __int64 v75; // rsi
  __int64 v76; // rax
  __int64 v77; // rsi
  __int64 v78; // rdx
  __int64 v79; // rsi
  __int64 v80; // r13
  __int64 v81; // rax
  __int64 v82; // r13
  __int64 v83; // r15
  __int64 v84; // r14
  __int64 v85; // rax
  __int64 v86; // r12
  __int64 v87; // r15
  __int64 j; // rbx
  __int64 v89; // r15
  __int64 v90; // r12
  __int64 v91; // rax
  __int64 v92; // rbx
  __int64 v93; // r15
  __int64 i; // r14
  __int64 v95; // [rsp+0h] [rbp-100h]
  __int64 v96; // [rsp+0h] [rbp-100h]
  __int64 v97; // [rsp+8h] [rbp-F8h]
  __int64 *v98; // [rsp+8h] [rbp-F8h]
  __int64 *v99; // [rsp+10h] [rbp-F0h]
  _DWORD *v100; // [rsp+10h] [rbp-F0h]
  __int64 *v101; // [rsp+18h] [rbp-E8h]
  _DWORD *v102; // [rsp+18h] [rbp-E8h]
  __int64 v103; // [rsp+18h] [rbp-E8h]
  __int64 *v104; // [rsp+20h] [rbp-E0h]
  unsigned int v105; // [rsp+20h] [rbp-E0h]
  __int64 *v106; // [rsp+20h] [rbp-E0h]
  __int64 *v107; // [rsp+20h] [rbp-E0h]
  __int64 *v108; // [rsp+20h] [rbp-E0h]
  __int64 v109; // [rsp+20h] [rbp-E0h]
  __int64 v111; // [rsp+28h] [rbp-D8h]
  _BYTE *v112; // [rsp+28h] [rbp-D8h]
  __int64 v113; // [rsp+28h] [rbp-D8h]
  __int64 v114; // [rsp+28h] [rbp-D8h]
  __int64 v115; // [rsp+28h] [rbp-D8h]
  __int64 *v116; // [rsp+28h] [rbp-D8h]
  __int64 v117; // [rsp+28h] [rbp-D8h]
  __int64 v118; // [rsp+28h] [rbp-D8h]
  __int64 v119; // [rsp+28h] [rbp-D8h]
  __int64 v120; // [rsp+30h] [rbp-D0h]
  __int64 v121; // [rsp+30h] [rbp-D0h]
  __int64 v122; // [rsp+30h] [rbp-D0h]
  __int64 v123; // [rsp+30h] [rbp-D0h]
  __int64 v124; // [rsp+30h] [rbp-D0h]
  char v125; // [rsp+38h] [rbp-C8h]
  int v126; // [rsp+3Ch] [rbp-C4h]
  __int64 v127; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v128; // [rsp+48h] [rbp-B8h] BYREF
  char *v129; // [rsp+50h] [rbp-B0h] BYREF
  int v130; // [rsp+58h] [rbp-A8h]
  char v131; // [rsp+5Ch] [rbp-A4h]
  __int64 v132; // [rsp+60h] [rbp-A0h]
  const char *v133; // [rsp+70h] [rbp-90h] BYREF
  _BYTE *v134; // [rsp+78h] [rbp-88h]
  __int64 v135; // [rsp+80h] [rbp-80h]
  __int64 v136; // [rsp+88h] [rbp-78h]
  __int64 v137; // [rsp+90h] [rbp-70h]
  __int64 v138; // [rsp+98h] [rbp-68h]
  const char *v139; // [rsp+A0h] [rbp-60h] BYREF
  _BYTE *v140; // [rsp+A8h] [rbp-58h] BYREF
  __int64 v141; // [rsp+B0h] [rbp-50h]
  __int64 v142; // [rsp+B8h] [rbp-48h]
  __int64 v143; // [rsp+C0h] [rbp-40h]
  __int64 v144; // [rsp+C8h] [rbp-38h]

  v7 = a1;
  v8 = *(_QWORD *)(a2 + 72);
  v125 = a4;
  v9 = (_DWORD *)(v8 + 36);
  sub_1286D80((__int64)&v133, *(_QWORD **)a1, v8, a4, a5);
  v126 = (int)v133;
  v141 = v135;
  v10 = *(__int64 **)a1;
  v140 = v134;
  v142 = v136;
  v139 = v133;
  v143 = v137;
  v144 = v138;
  sub_1287CD0((__int64)&v129, v10, v9, (__int64)&v129, v137, v138, (__int64)v133, v134, v135, v136, v137, v138);
  v11 = (__int64 *)v129;
  LODWORD(v10) = -(a3 == 0);
  v12 = *(_QWORD *)v129;
  v13 = (unsigned int)v10 | 1;
  v14 = *(_BYTE *)(*(_QWORD *)v129 + 8LL);
  if ( v14 == 15 )
  {
    v15 = sub_1643350(*(_QWORD *)(a1 + 16));
    v16 = (__int64 *)sub_159C470(v15, v13, 0);
    if ( *(_BYTE *)(*(_QWORD *)(v12 + 24) + 8LL) != 12 )
    {
      v17 = *(__int64 **)(a1 + 8);
      v139 = "ptrincdec";
      LOWORD(v141) = 259;
      v129 = (char *)v16;
      v127 = sub_128B460(v17, 0, v11, (__int64 **)&v129, 1, (__int64)&v139);
      v21 = v127;
      goto LABEL_4;
    }
    v31 = sub_1643330(*(_QWORD *)(a1 + 16));
    v32 = sub_1646BA0(v31, 0);
    v33 = *(__int64 **)(a1 + 8);
    LOWORD(v132) = 259;
    v129 = "tmp";
    if ( v32 == *v11 )
    {
      v35 = (__int64)v11;
    }
    else if ( *((_BYTE *)v11 + 16) > 0x10u )
    {
      LOWORD(v141) = 257;
      v116 = v33;
      v70 = sub_15FDBD0(47, v11, v32, &v139, 0);
      v71 = v116;
      v72 = v70;
      v73 = v116[1];
      if ( v73 )
      {
        v74 = (__int64 *)v116[2];
        v101 = v116;
        v117 = v70;
        v107 = v74;
        sub_157E9D0(v73 + 40, v70);
        v72 = v117;
        v71 = v101;
        v75 = *v107;
        v76 = *(_QWORD *)(v117 + 24);
        *(_QWORD *)(v117 + 32) = v107;
        v75 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v117 + 24) = v75 | v76 & 7;
        *(_QWORD *)(v75 + 8) = v117 + 24;
        *v107 = *v107 & 7 | (v117 + 24);
      }
      v108 = v71;
      v118 = v72;
      sub_164B780(v72, &v129);
      v35 = v118;
      v77 = *v108;
      if ( *v108 )
      {
        v128 = *v108;
        sub_1623A60(&v128, v77, 2);
        v35 = v118;
        v78 = v118 + 48;
        if ( *(_QWORD *)(v118 + 48) )
        {
          sub_161E7C0(v118 + 48);
          v35 = v118;
          v78 = v118 + 48;
        }
        v79 = v128;
        *(_QWORD *)(v35 + 48) = v128;
        if ( v79 )
        {
          v119 = v35;
          sub_1623210(&v128, v79, v78);
          v35 = v119;
        }
      }
      v33 = *(__int64 **)(v7 + 8);
    }
    else
    {
      v34 = sub_15A46C0(47, v11, v32, 0);
      v33 = *(__int64 **)(a1 + 8);
      v35 = v34;
    }
    v36 = *(_QWORD *)(v7 + 16);
    v104 = v33;
    v127 = v35;
    v112 = (_BYTE *)v35;
    v139 = "ptrincdec";
    LOWORD(v141) = 259;
    v37 = sub_1643330(v36);
    v38 = sub_12815B0(v104, v37, v112, (__int64)v16, (__int64)&v139);
    v39 = *(__int64 **)(v7 + 8);
    v127 = v38;
    v21 = v38;
    LOWORD(v132) = 257;
    v18 = *v11;
    if ( *v11 != *(_QWORD *)v38 )
    {
      if ( *(_BYTE *)(v38 + 16) > 0x10u )
      {
        LOWORD(v141) = 257;
        v61 = sub_15FDBD0(47, v38, v18, &v139, 0);
        v62 = v39[1];
        v63 = v61;
        if ( v62 )
        {
          v64 = (__int64 *)v39[2];
          v115 = v61;
          sub_157E9D0(v62 + 40, v61);
          v63 = v115;
          v65 = *v64;
          v66 = *(_QWORD *)(v115 + 24);
          *(_QWORD *)(v115 + 32) = v64;
          v65 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v115 + 24) = v65 | v66 & 7;
          *(_QWORD *)(v65 + 8) = v115 + 24;
          *v64 = *v64 & 7 | (v115 + 24);
        }
        v123 = v63;
        sub_164B780(v63, &v129);
        v67 = *v39;
        v21 = v123;
        if ( *v39 )
        {
          v128 = *v39;
          sub_1623A60(&v128, v67, 2);
          v21 = v123;
          v20 = &v128;
          v68 = v123 + 48;
          if ( *(_QWORD *)(v123 + 48) )
          {
            sub_161E7C0(v123 + 48);
            v20 = &v128;
            v21 = v123;
          }
          v69 = v128;
          *(_QWORD *)(v21 + 48) = v128;
          if ( v69 )
          {
            v124 = v21;
            sub_1623210(&v128, v69, v68);
            v21 = v124;
          }
        }
      }
      else
      {
        v21 = sub_15A46C0(47, v38, v18, 0);
      }
    }
    goto LABEL_29;
  }
  if ( v14 != 11 )
  {
    if ( (unsigned __int8)(v14 - 1) > 5u )
      sub_127B550("unsupported type in pre/post increment/decrement expression!", (_DWORD *)(a2 + 36), 1);
    v24 = *(_QWORD *)(a1 + 16);
    if ( v12 == sub_16432A0(v24) )
    {
      v58 = sub_1698270(v24, (unsigned int)v13);
      sub_169D3B0(&v129, (float)v13);
      sub_169E320(&v140, &v129, v58);
      sub_1698460(&v129);
      v127 = sub_159CCF0(*(_QWORD *)(v7 + 16), &v139);
      v59 = sub_16982C0();
      v18 = v127;
      v60 = v59;
      if ( v140 == (_BYTE *)v59 )
      {
        v109 = v141;
        if ( v141 )
        {
          v89 = v141 + 32LL * *(_QWORD *)(v141 - 8);
          if ( v141 != v89 )
          {
            v100 = v9;
            v98 = v11;
            v96 = v7;
            do
            {
              v89 -= 32;
              if ( v60 == *(_QWORD *)(v89 + 8) )
              {
                v90 = *(_QWORD *)(v89 + 16);
                if ( v90 )
                {
                  v91 = 32LL * *(_QWORD *)(v90 - 8);
                  v92 = v90 + v91;
                  if ( v90 != v90 + v91 )
                  {
                    v103 = v89;
                    do
                    {
                      v92 -= 32;
                      if ( v60 == *(_QWORD *)(v92 + 8) )
                      {
                        v93 = *(_QWORD *)(v92 + 16);
                        if ( v93 )
                        {
                          for ( i = v93 + 32LL * *(_QWORD *)(v93 - 8); v93 != i; sub_127D120((_QWORD *)(i + 8)) )
                            i -= 32;
                          j_j_j___libc_free_0_0(v93 - 8);
                        }
                      }
                      else
                      {
                        sub_1698460(v92 + 8);
                      }
                    }
                    while ( v90 != v92 );
                    v89 = v103;
                  }
                  j_j_j___libc_free_0_0(v90 - 8);
                }
              }
              else
              {
                sub_1698460(v89 + 8);
              }
            }
            while ( v109 != v89 );
            v9 = v100;
            v11 = v98;
            v7 = v96;
          }
          goto LABEL_100;
        }
LABEL_14:
        v27 = "inc";
        BYTE1(v132) = 1;
        if ( !a3 )
          v27 = "dec";
        LOBYTE(v132) = 3;
        v28 = *(_QWORD *)(v7 + 8);
        v129 = v27;
        if ( *((_BYTE *)v11 + 16) > 0x10u
          || *(_BYTE *)(v18 + 16) > 0x10u
          || (v111 = v18, v29 = sub_15A2A30(12, v11, v18, 0, 0), v18 = v111, (v21 = v29) == 0) )
        {
          LOWORD(v141) = 257;
          v47 = sub_15FB440(12, v11, v18, &v139, 0);
          v48 = *(_QWORD *)(v28 + 32);
          v49 = *(_DWORD *)(v28 + 40);
          v50 = v47;
          if ( v48 )
          {
            v105 = *(_DWORD *)(v28 + 40);
            v113 = v47;
            sub_1625C10(v47, 3, v48);
            v49 = v105;
            v50 = v113;
          }
          v114 = v50;
          sub_15F2440(v50, v49);
          v51 = *(_QWORD *)(v28 + 8);
          v52 = v114;
          if ( v51 )
          {
            v106 = *(__int64 **)(v28 + 16);
            sub_157E9D0(v51 + 40, v114);
            v52 = v114;
            v53 = *(_QWORD *)(v114 + 24);
            v54 = *v106;
            *(_QWORD *)(v114 + 32) = v106;
            v54 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v114 + 24) = v54 | v53 & 7;
            *(_QWORD *)(v54 + 8) = v114 + 24;
            *v106 = *v106 & 7 | (v114 + 24);
          }
          v121 = v52;
          sub_164B780(v52, &v129);
          v55 = *(const char **)v28;
          v21 = v121;
          if ( *(_QWORD *)v28 )
          {
            v139 = *(const char **)v28;
            sub_1623A60(&v139, v55, 2);
            v21 = v121;
            v56 = v121 + 48;
            if ( *(_QWORD *)(v121 + 48) )
            {
              sub_161E7C0(v121 + 48);
              v21 = v121;
            }
            v57 = v139;
            *(_QWORD *)(v21 + 48) = v139;
            if ( v57 )
            {
              v122 = v21;
              sub_1623210(&v139, v57, v56);
              v21 = v122;
            }
          }
        }
        if ( unk_4D04700 && *(_BYTE *)(v21 + 16) > 0x17u )
        {
          v120 = v21;
          v30 = sub_15F24E0(v21);
          sub_15F2440(v120, v30 | 1u);
          v21 = v120;
        }
LABEL_29:
        v127 = v21;
        goto LABEL_4;
      }
    }
    else
    {
      v25 = *(_QWORD *)(v7 + 16);
      v26 = *v11;
      if ( v26 != sub_16432B0(v25) )
        sub_127B550(
          "unsupported floating point type in pre/post increment/decrement expression!",
          (_DWORD *)(a2 + 36),
          1);
      v80 = sub_1698280(v25);
      sub_169D3F0(&v129, (double)v13);
      sub_169E320(&v140, &v129, v80);
      sub_1698460(&v129);
      v127 = sub_159CCF0(*(_QWORD *)(v7 + 16), &v139);
      v81 = sub_16982C0();
      v18 = v127;
      v82 = v81;
      if ( v140 == (_BYTE *)v81 )
      {
        v109 = v141;
        if ( v141 )
        {
          v83 = v141 + 32LL * *(_QWORD *)(v141 - 8);
          if ( v141 != v83 )
          {
            v102 = v9;
            v99 = v11;
            v97 = v7;
            do
            {
              v83 -= 32;
              if ( v82 == *(_QWORD *)(v83 + 8) )
              {
                v84 = *(_QWORD *)(v83 + 16);
                if ( v84 )
                {
                  v85 = 32LL * *(_QWORD *)(v84 - 8);
                  v86 = v84 + v85;
                  if ( v84 != v84 + v85 )
                  {
                    v95 = v83;
                    do
                    {
                      v86 -= 32;
                      if ( v82 == *(_QWORD *)(v86 + 8) )
                      {
                        v87 = *(_QWORD *)(v86 + 16);
                        if ( v87 )
                        {
                          for ( j = v87 + 32LL * *(_QWORD *)(v87 - 8); v87 != j; sub_127D120((_QWORD *)(j + 8)) )
                            j -= 32;
                          j_j_j___libc_free_0_0(v87 - 8);
                        }
                      }
                      else
                      {
                        sub_1698460(v86 + 8);
                      }
                    }
                    while ( v84 != v86 );
                    v83 = v95;
                  }
                  j_j_j___libc_free_0_0(v84 - 8);
                }
              }
              else
              {
                sub_1698460(v83 + 8);
              }
            }
            while ( v109 != v83 );
            v9 = v102;
            v11 = v99;
            v7 = v97;
          }
LABEL_100:
          j_j_j___libc_free_0_0(v109 - 8);
          v18 = v127;
          goto LABEL_14;
        }
        goto LABEL_14;
      }
    }
    sub_1698460(&v140);
    v18 = v127;
    goto LABEL_14;
  }
  v40 = sub_15A0680(*(_QWORD *)v129, v13, a3 ^ 1u);
  v41 = *(_QWORD *)a2;
  v127 = v40;
  v42 = sub_127B3A0(v41);
  v43 = *(__int64 **)(v7 + 8);
  if ( v42 )
  {
    v44 = "inc";
    LOWORD(v141) = 259;
    v45 = 1;
    if ( !a3 )
      v44 = "dec";
    v139 = v44;
  }
  else
  {
    v46 = "inc";
    LOWORD(v141) = 259;
    if ( !a3 )
      v46 = "dec";
    v45 = 0;
    v139 = v46;
  }
  v127 = sub_12899C0(v43, (__int64)v11, v127, (__int64)&v139, 0, v45);
  v21 = v127;
LABEL_4:
  v22 = *(__int64 **)v7;
  if ( v126 == 1 )
  {
    v131 &= ~1u;
    v130 = 0;
    LODWORD(v132) = 0;
    LODWORD(v133) = 1;
    v129 = (char *)v21;
    sub_1282050(v22, v9, &v127, v19, (__int64)v20, v21, v21, 0, 0, (__int64)v133, v134, v135, v136, v137, v138);
  }
  else
  {
    BYTE4(v140) &= ~1u;
    LODWORD(v140) = 0;
    LODWORD(v141) = 0;
    v139 = (const char *)v21;
    sub_12843D0(v22, v9, v18, v19, (__int64)v20, v21, v21, 0, 0, (__int64)v133, v134, v135, v136, v137, v138);
  }
  if ( v125 )
    return v127;
  return (__int64)v11;
}
