// Function: sub_2A02F40
// Address: 0x2a02f40
//
__int64 __fastcall sub_2A02F40(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  __int64 v4; // r13
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  unsigned __int64 v7; // rcx
  int v8; // eax
  unsigned __int64 v9; // rcx
  bool v10; // cf
  __int64 v11; // rax
  _QWORD *v12; // rax
  _QWORD *i; // rdx
  char v14; // cl
  _QWORD *v15; // rax
  _QWORD *v16; // rdx
  char v17; // cl
  char v18; // al
  __int64 v19; // rdi
  char v20; // bl
  __int64 v21; // rcx
  unsigned __int64 *v22; // r8
  __int64 v23; // r9
  __int64 v24; // r14
  __int64 *v25; // rdi
  _QWORD *v26; // r14
  __int64 v27; // rax
  __int64 *p_j; // rdx
  unsigned __int64 *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdi
  __int64 v33; // r8
  __int64 v34; // r9
  char *v35; // r14
  char *v36; // r13
  char *v37; // r13
  unsigned __int8 *v38; // rdi
  unsigned __int64 v39; // rax
  __int64 v40; // rsi
  _QWORD *v41; // rbx
  _QWORD *v42; // r14
  __int64 v43; // rsi
  __int64 v44; // rsi
  _QWORD *v45; // r15
  _QWORD *v46; // r14
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rsi
  _QWORD *v51; // r15
  _QWORD *v52; // r14
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v57; // rsi
  _QWORD *v58; // rbx
  _QWORD *v59; // r14
  __int64 v60; // rsi
  unsigned __int8 *v61; // rax
  unsigned __int64 v62; // rax
  int v63; // edx
  __int64 v64; // rdi
  __int64 v65; // rax
  __int64 v66; // rax
  unsigned __int64 v67; // rdi
  unsigned __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rax
  const char *v71; // rbx
  unsigned __int64 v72; // rdi
  unsigned __int64 v73; // rax
  __int64 v74; // rax
  __int64 v75; // rax
  __int64 v76; // rsi
  __int64 *v77; // rdi
  _QWORD *v78; // rax
  __int64 v79; // rdx
  __int64 v80; // [rsp-10h] [rbp-620h]
  __int64 v81; // [rsp-8h] [rbp-618h]
  unsigned __int8 *v82; // [rsp+8h] [rbp-608h]
  _QWORD *v83; // [rsp+10h] [rbp-600h]
  char v84; // [rsp+1Fh] [rbp-5F1h]
  __int64 v85; // [rsp+20h] [rbp-5F0h]
  __int64 v86; // [rsp+20h] [rbp-5F0h]
  __int64 v87; // [rsp+28h] [rbp-5E8h]
  char v88; // [rsp+30h] [rbp-5E0h]
  unsigned __int64 *v89; // [rsp+30h] [rbp-5E0h]
  char v90; // [rsp+38h] [rbp-5D8h]
  unsigned __int8 v91; // [rsp+38h] [rbp-5D8h]
  const char *v92; // [rsp+40h] [rbp-5D0h] BYREF
  unsigned __int64 v93; // [rsp+48h] [rbp-5C8h] BYREF
  unsigned __int64 v94; // [rsp+50h] [rbp-5C0h]
  __int64 v95; // [rsp+58h] [rbp-5B8h]
  __int64 v96; // [rsp+60h] [rbp-5B0h]
  __int64 v97; // [rsp+68h] [rbp-5A8h]
  const char *v98; // [rsp+70h] [rbp-5A0h] BYREF
  __int64 v99; // [rsp+78h] [rbp-598h] BYREF
  unsigned __int64 v100; // [rsp+80h] [rbp-590h]
  __int64 v101; // [rsp+88h] [rbp-588h]
  __int64 v102; // [rsp+90h] [rbp-580h]
  __int64 v103; // [rsp+98h] [rbp-578h]
  const char *v104; // [rsp+A0h] [rbp-570h] BYREF
  __int64 v105; // [rsp+A8h] [rbp-568h] BYREF
  unsigned __int64 v106; // [rsp+B0h] [rbp-560h] BYREF
  __int64 v107; // [rsp+B8h] [rbp-558h] BYREF
  __int64 j; // [rsp+C0h] [rbp-550h] BYREF
  __int64 v109; // [rsp+C8h] [rbp-548h]
  unsigned __int64 v110; // [rsp+D0h] [rbp-540h] BYREF
  __int64 v111; // [rsp+D8h] [rbp-538h]
  __int64 v112; // [rsp+E0h] [rbp-530h]
  __int64 v113; // [rsp+E8h] [rbp-528h] BYREF
  _QWORD *v114; // [rsp+F0h] [rbp-520h]
  __int64 v115; // [rsp+F8h] [rbp-518h]
  unsigned int v116; // [rsp+100h] [rbp-510h]
  _QWORD *v117; // [rsp+110h] [rbp-500h]
  unsigned int v118; // [rsp+120h] [rbp-4F0h]
  char v119; // [rsp+128h] [rbp-4E8h]
  const char *v120; // [rsp+138h] [rbp-4D8h] BYREF
  __int64 v121; // [rsp+140h] [rbp-4D0h]
  __int64 v122; // [rsp+148h] [rbp-4C8h]
  __int64 v123; // [rsp+150h] [rbp-4C0h]
  __int64 v124; // [rsp+158h] [rbp-4B8h]
  int v125; // [rsp+160h] [rbp-4B0h]
  __int64 v126; // [rsp+168h] [rbp-4A8h]
  __int64 v127; // [rsp+170h] [rbp-4A0h]
  __int64 v128; // [rsp+178h] [rbp-498h]
  __int64 v129; // [rsp+180h] [rbp-490h]
  __int16 v130; // [rsp+188h] [rbp-488h]
  __int64 v131; // [rsp+190h] [rbp-480h]
  void *v132; // [rsp+1A0h] [rbp-470h] BYREF
  __int64 v133; // [rsp+1A8h] [rbp-468h] BYREF
  __int64 v134; // [rsp+1B0h] [rbp-460h]
  __int64 v135; // [rsp+1B8h] [rbp-458h] BYREF
  _QWORD *v136; // [rsp+1C0h] [rbp-450h]
  __int64 v137; // [rsp+1C8h] [rbp-448h]
  unsigned int v138; // [rsp+1D0h] [rbp-440h]
  _QWORD *v139; // [rsp+1E0h] [rbp-430h]
  unsigned int v140; // [rsp+1F0h] [rbp-420h]
  char v141; // [rsp+1F8h] [rbp-418h]
  _QWORD v142[5]; // [rsp+208h] [rbp-408h] BYREF
  int v143; // [rsp+230h] [rbp-3E0h]
  __int64 v144; // [rsp+238h] [rbp-3D8h]
  __int64 v145; // [rsp+240h] [rbp-3D0h]
  __int64 v146; // [rsp+248h] [rbp-3C8h]
  __int64 v147; // [rsp+250h] [rbp-3C0h]
  __int16 v148; // [rsp+258h] [rbp-3B8h]
  __int64 v149; // [rsp+260h] [rbp-3B0h]
  __int64 v150[116]; // [rsp+270h] [rbp-3A0h] BYREF

  v2 = sub_D4B130(*(_QWORD *)(a1 + 56));
  v3 = *(_QWORD *)a1;
  v4 = *(_QWORD *)(a1 + 80);
  v87 = v2;
  *(_QWORD *)(a1 + 64) = v2;
  *(_QWORD *)(a1 + 72) = v2;
  v84 = *(_BYTE *)(a1 + 169);
  v88 = *(_BYTE *)(a1 + 168);
  v5 = sub_B2BEC0(v3);
  sub_27C1C30((__int64)v150, *(__int64 **)(a1 + 16), v5, (__int64)"loop-constrainer", 1);
  v6 = *(_QWORD *)(a1 + 64);
  v7 = *(_QWORD *)(v6 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v7 == v6 + 48 )
  {
    v85 = 0;
  }
  else
  {
    if ( !v7 )
      BUG();
    v8 = *(unsigned __int8 *)(v7 - 24);
    v9 = v7 - 24;
    v10 = (unsigned int)(v8 - 30) < 0xB;
    v11 = 0;
    if ( v10 )
      v11 = v9;
    v85 = v11;
  }
  v110 = 0;
  v111 = 0;
  v112 = 0;
  v113 = 0;
  v116 = 128;
  v12 = (_QWORD *)sub_C7D670(0x2000, 8);
  v115 = 0;
  v114 = v12;
  v133 = 2;
  v136 = 0;
  for ( i = v12 + 1024; i != v12; v12 += 8 )
  {
    if ( v12 )
    {
      v14 = v133;
      v12[2] = 0;
      v12[3] = -4096;
      *v12 = &unk_49DD7B0;
      v12[1] = v14 & 6;
      v12[4] = v136;
    }
  }
  v119 = 0;
  v121 = 0;
  v120 = byte_3F871B3;
  v122 = 0;
  v123 = 0;
  v124 = 0;
  v125 = -1;
  v126 = 0;
  v127 = 0;
  v128 = 0;
  v129 = 0;
  v130 = 256;
  v131 = 0;
  v132 = 0;
  v133 = 0;
  v134 = 0;
  v135 = 0;
  v138 = 128;
  v15 = (_QWORD *)sub_C7D670(0x2000, 8);
  v137 = 0;
  v136 = v15;
  v105 = 2;
  v16 = v15 + 1024;
  v104 = (const char *)&unk_49DD7B0;
  v106 = 0;
  v107 = -4096;
  for ( j = 0; v16 != v15; v15 += 8 )
  {
    if ( v15 )
    {
      v17 = v105;
      v15[2] = 0;
      v15[3] = -4096;
      *v15 = &unk_49DD7B0;
      v15[1] = v17 & 6;
      v15[4] = j;
    }
  }
  v18 = *(_BYTE *)(a1 + 208);
  v142[0] = byte_3F871B3;
  v141 = 0;
  v19 = *(_QWORD *)(a1 + 16);
  memset(&v142[1], 0, 32);
  v20 = *(_BYTE *)(a1 + 192);
  v143 = -1;
  v144 = 0;
  v145 = 0;
  v146 = 0;
  v147 = 0;
  v148 = 256;
  v149 = 0;
  v90 = v18;
  if ( v88 )
  {
    v83 = sub_DA2C50(v19, v4, -1, 1u);
    if ( v20 )
    {
      v24 = *(_QWORD *)(a1 + 184);
      goto LABEL_17;
    }
    v82 = 0;
    if ( v90 )
      goto LABEL_119;
    goto LABEL_26;
  }
  v83 = sub_DA2C50(v19, v4, -1, 1u);
  if ( v90 )
  {
    if ( !sub_F70610(*(_QWORD *)(a1 + 200), *(_QWORD *)(a1 + 56), *(__int64 **)(a1 + 16), v84) )
    {
LABEL_56:
      v91 = 0;
      if ( !v141 )
        goto LABEL_57;
      goto LABEL_47;
    }
    v77 = *(__int64 **)(a1 + 16);
    v106 = *(_QWORD *)(a1 + 200);
    v104 = (const char *)&v106;
    v107 = (__int64)v83;
    v105 = 0x200000002LL;
    v78 = sub_DC7EB0(v77, (__int64)&v104, 0, 0);
    v22 = &v106;
    v24 = (__int64)v78;
    if ( v104 != (const char *)&v106 )
      _libc_free((unsigned __int64)v104);
    v90 = v20;
LABEL_17:
    if ( !(unsigned __int8)sub_F80650(v150, v24, v85, v21, (__int64)v22, v23) )
      goto LABEL_56;
    v82 = (unsigned __int8 *)sub_F8DB90((__int64)v150, v24, v4, v85 + 24, 0);
    v104 = "exit.preloop.at";
    LOWORD(j) = 259;
    sub_BD6B50(v82, &v104);
    if ( !v90 )
    {
      sub_2A01A10((__int64 *)a1, (__int64)&v110, "preloop");
      v92 = 0;
      v93 = 0;
      v94 = 0;
      v95 = 0;
      v96 = 0;
      v97 = 0;
      v86 = 0;
      goto LABEL_123;
    }
    v20 = v88;
    if ( !v88 )
      goto LABEL_20;
LABEL_119:
    v26 = *(_QWORD **)(a1 + 200);
    goto LABEL_120;
  }
  if ( !v20 )
  {
LABEL_26:
    v92 = 0;
    v93 = 0;
    v94 = 0;
    v95 = 0;
    v96 = 0;
    v97 = 0;
    v98 = 0;
    v99 = 0;
    v100 = 0;
    v101 = 0;
    v102 = 0;
    v103 = 0;
LABEL_27:
    v27 = *(_QWORD *)(a1 + 72);
    if ( v87 == v27 )
    {
      p_j = (__int64 *)&v104;
      v104 = 0;
      v107 = 0;
      v105 = (__int64)v92;
      j = 0;
      v106 = v93;
      v109 = 0;
    }
    else
    {
      v104 = 0;
      v107 = 0;
      v105 = (__int64)v92;
      j = 0;
      v106 = v93;
      p_j = (__int64 *)&v104;
      v109 = v27;
    }
    goto LABEL_29;
  }
  v82 = 0;
LABEL_20:
  if ( !sub_F70610(*(_QWORD *)(a1 + 184), *(_QWORD *)(a1 + 56), *(__int64 **)(a1 + 16), v84) )
    goto LABEL_56;
  v25 = *(__int64 **)(a1 + 16);
  v106 = *(_QWORD *)(a1 + 184);
  v104 = (const char *)&v106;
  v107 = (__int64)v83;
  v105 = 0x200000002LL;
  v26 = sub_DC7EB0(v25, (__int64)&v104, 0, 0);
  if ( v104 != (const char *)&v106 )
    _libc_free((unsigned __int64)v104);
  v20 = v90;
LABEL_120:
  if ( !(unsigned __int8)sub_F80650(v150, (__int64)v26, v85, v21, (__int64)v22, v23) )
    goto LABEL_56;
  v61 = (unsigned __int8 *)sub_F8DB90((__int64)v150, (__int64)v26, v4, v85 + 24, 0);
  v104 = "exit.mainloop.at";
  v86 = (__int64)v61;
  LOWORD(j) = 259;
  sub_BD6B50(v61, &v104);
  if ( !v20 )
  {
    sub_2A01A10((__int64 *)a1, (__int64)&v132, "postloop");
    v92 = 0;
    v93 = 0;
    v94 = 0;
    v95 = 0;
    v96 = 0;
    v97 = 0;
    v98 = 0;
    v99 = 0;
    v100 = 0;
    v101 = 0;
    v102 = 0;
    v103 = 0;
    v89 = (unsigned __int64 *)(a1 + 88);
    goto LABEL_131;
  }
  sub_2A01A10((__int64 *)a1, (__int64)&v110, "preloop");
  sub_2A01A10((__int64 *)a1, (__int64)&v132, "postloop");
  v90 = v20;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v97 = 0;
LABEL_123:
  v62 = *(_QWORD *)(v87 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v62 == v87 + 48 )
  {
    v64 = 0;
  }
  else
  {
    if ( !v62 )
      BUG();
    v63 = *(unsigned __int8 *)(v62 - 24);
    v64 = 0;
    v65 = v62 - 24;
    if ( (unsigned int)(v63 - 30) < 0xB )
      v64 = v65;
  }
  sub_BD2ED0(v64, *(_QWORD *)(a1 + 96), v121);
  v89 = (unsigned __int64 *)(a1 + 88);
  v66 = sub_2A00670((__int64 *)a1, a1 + 88, v87, "mainloop");
  *(_QWORD *)(a1 + 72) = v66;
  sub_2A007B0((__int64 *)&v104, (__int64 *)a1, (unsigned __int64 *)&v120, v87, (__int64)v82, v66);
  v67 = v94;
  v92 = v104;
  v93 = v105;
  v68 = v106;
  v106 = 0;
  v94 = v68;
  v69 = v107;
  v107 = 0;
  v95 = v69;
  v70 = j;
  j = 0;
  v96 = v70;
  if ( v67 )
  {
    j_j___libc_free_0(v67);
    v97 = v109;
    if ( v106 )
      j_j___libc_free_0(v106);
  }
  else
  {
    v97 = v109;
  }
  sub_2A00560(a1, (__int64)v89, *(_QWORD *)(a1 + 72), (__int64)&v92);
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v103 = 0;
  if ( !v90 )
    goto LABEL_27;
LABEL_131:
  v71 = (const char *)sub_2A00670((__int64 *)a1, (__int64)v142, v87, "postloop");
  sub_2A007B0((__int64 *)&v104, (__int64 *)a1, v89, *(_QWORD *)(a1 + 72), v86, (__int64)v71);
  v72 = v100;
  v98 = v104;
  v99 = v105;
  v73 = v106;
  v106 = 0;
  v100 = v73;
  v74 = v107;
  v107 = 0;
  v101 = v74;
  v75 = j;
  j = 0;
  v102 = v75;
  if ( v72 )
  {
    j_j___libc_free_0(v72);
    v103 = v109;
    if ( v106 )
      j_j___libc_free_0(v106);
  }
  else
  {
    v103 = v109;
  }
  sub_2A00560(a1, (__int64)v142, (__int64)v71, (__int64)&v98);
  v76 = *(_QWORD *)(a1 + 72);
  if ( v76 == v87 )
    v76 = 0;
  v104 = v71;
  v107 = (__int64)v98;
  v105 = (__int64)v92;
  v106 = v93;
  j = v99;
  v109 = v76;
  if ( v71 )
  {
    if ( v92 )
    {
      if ( v93 )
      {
        if ( v98 )
        {
          if ( v99 )
          {
            v30 = 5 - ((v76 == 0) - 1LL);
            goto LABEL_34;
          }
          p_j = &j;
        }
        else
        {
          p_j = &v107;
        }
      }
      else
      {
        p_j = (__int64 *)&v106;
      }
    }
    else
    {
      p_j = &v105;
    }
  }
  else
  {
    p_j = (__int64 *)&v104;
  }
LABEL_29:
  v29 = (unsigned __int64 *)(p_j + 1);
  do
  {
    if ( *v29 )
      *p_j++ = *v29;
    ++v29;
  }
  while ( v29 != &v110 );
  v30 = ((char *)p_j - (char *)&v104) >> 3;
LABEL_34:
  sub_2A00750(a1, (__int64 *)&v104, v30);
  v31 = *(_QWORD *)a1;
  v32 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)(v32 + 104) = *(_QWORD *)a1;
  *(_DWORD *)(v32 + 120) = *(_DWORD *)(v31 + 92);
  sub_B1F440(v32);
  if ( v110 == v111 )
  {
    if ( (void *)v133 == v132 )
      goto LABEL_40;
    v36 = (char *)sub_2A028D0(a1, *(_QWORD *)(a1 + 56), **(_QWORD **)(a1 + 56), (__int64)&v135, 0);
    goto LABEL_38;
  }
  v35 = (char *)sub_2A028D0(a1, *(_QWORD *)(a1 + 56), **(_QWORD **)(a1 + 56), (__int64)&v113, 0);
  if ( (void *)v133 == v132 )
  {
    v36 = 0;
    if ( !v35 )
      goto LABEL_40;
  }
  else
  {
    v36 = (char *)sub_2A028D0(a1, *(_QWORD *)(a1 + 56), **(_QWORD **)(a1 + 56), (__int64)&v135, 0);
    if ( !v35 )
      goto LABEL_38;
  }
  sub_11D2180((__int64)v35, *(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 16), v33, v34);
  sub_F6AC10(v35, *(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 16), 0, 0, 1u);
  sub_29FE890((__int64)v35);
  v33 = v80;
  v34 = v81;
LABEL_38:
  if ( v36 )
  {
    sub_11D2180((__int64)v36, *(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 16), v33, v34);
    sub_F6AC10(v36, *(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 16), 0, 0, 1u);
    sub_29FE890((__int64)v36);
  }
LABEL_40:
  v37 = *(char **)(a1 + 56);
  sub_11D2180((__int64)v37, *(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 16), v33, v34);
  sub_F6AC10(v37, *(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 16), 0, 0, 1u);
  v38 = *(unsigned __int8 **)(a1 + 136);
  v39 = *v38;
  if ( (unsigned __int8)v39 <= 0x1Cu )
  {
    if ( (_BYTE)v39 != 5 || (*((_WORD *)v38 + 1) & 0xFFFD) != 0xD && (*((_WORD *)v38 + 1) & 0xFFF7) != 0x11 )
      goto LABEL_42;
  }
  else
  {
    if ( (unsigned __int8)v39 > 0x36u )
      goto LABEL_42;
    v79 = 0x40540000000000LL;
    if ( !_bittest64(&v79, v39) )
      goto LABEL_42;
  }
  if ( v84 )
    sub_B44850(v38, 1);
LABEL_42:
  if ( v100 )
    j_j___libc_free_0(v100);
  if ( v94 )
    j_j___libc_free_0(v94);
  v91 = 1;
  if ( v141 )
  {
LABEL_47:
    v40 = v140;
    v141 = 0;
    if ( v140 )
    {
      v41 = v139;
      v42 = &v139[2 * v140];
      do
      {
        if ( *v41 != -4096 && *v41 != -8192 )
        {
          v43 = v41[1];
          if ( v43 )
            sub_B91220((__int64)(v41 + 1), v43);
        }
        v41 += 2;
      }
      while ( v42 != v41 );
      v40 = v140;
    }
    sub_C7D6A0((__int64)v139, 16 * v40, 8);
  }
LABEL_57:
  v44 = v138;
  if ( v138 )
  {
    v45 = v136;
    v93 = 2;
    v94 = 0;
    v46 = &v136[8 * (unsigned __int64)v138];
    v95 = -4096;
    v92 = (const char *)&unk_49DD7B0;
    v98 = (const char *)&unk_49DD7B0;
    v47 = -4096;
    v96 = 0;
    v99 = 2;
    v100 = 0;
    v101 = -8192;
    v102 = 0;
    while ( 1 )
    {
      v48 = v45[3];
      if ( v48 != v47 )
      {
        v47 = v101;
        if ( v48 != v101 )
        {
          v49 = v45[7];
          if ( v49 != 0 && v49 != -4096 && v49 != -8192 )
          {
            sub_BD60C0(v45 + 5);
            v48 = v45[3];
          }
          v47 = v48;
        }
      }
      *v45 = &unk_49DB368;
      if ( v47 != -4096 && v47 != 0 && v47 != -8192 )
        sub_BD60C0(v45 + 1);
      v45 += 8;
      if ( v46 == v45 )
        break;
      v47 = v95;
    }
    v98 = (const char *)&unk_49DB368;
    if ( v101 != 0 && v101 != -4096 && v101 != -8192 )
      sub_BD60C0(&v99);
    v92 = (const char *)&unk_49DB368;
    if ( v95 != 0 && v95 != -4096 && v95 != -8192 )
      sub_BD60C0(&v93);
    v44 = v138;
  }
  sub_C7D6A0((__int64)v136, v44 << 6, 8);
  if ( v132 )
    j_j___libc_free_0((unsigned __int64)v132);
  if ( v119 )
  {
    v57 = v118;
    v119 = 0;
    if ( v118 )
    {
      v58 = v117;
      v59 = &v117[2 * v118];
      do
      {
        if ( *v58 != -4096 && *v58 != -8192 )
        {
          v60 = v58[1];
          if ( v60 )
            sub_B91220((__int64)(v58 + 1), v60);
        }
        v58 += 2;
      }
      while ( v59 != v58 );
      v57 = v118;
    }
    sub_C7D6A0((__int64)v117, 16 * v57, 8);
  }
  v50 = v116;
  if ( v116 )
  {
    v51 = v114;
    v99 = 2;
    v100 = 0;
    v52 = &v114[8 * (unsigned __int64)v116];
    v101 = -4096;
    v98 = (const char *)&unk_49DD7B0;
    v132 = &unk_49DD7B0;
    v53 = -4096;
    v102 = 0;
    v133 = 2;
    v134 = 0;
    v135 = -8192;
    v136 = 0;
    while ( 1 )
    {
      v54 = v51[3];
      if ( v54 != v53 )
      {
        v53 = v135;
        if ( v54 != v135 )
        {
          v55 = v51[7];
          if ( v55 != 0 && v55 != -4096 && v55 != -8192 )
          {
            sub_BD60C0(v51 + 5);
            v54 = v51[3];
          }
          v53 = v54;
        }
      }
      *v51 = &unk_49DB368;
      if ( v53 != 0 && v53 != -4096 && v53 != -8192 )
        sub_BD60C0(v51 + 1);
      v51 += 8;
      if ( v52 == v51 )
        break;
      v53 = v101;
    }
    v132 = &unk_49DB368;
    if ( v135 != -4096 && v135 != 0 && v135 != -8192 )
      sub_BD60C0(&v133);
    v98 = (const char *)&unk_49DB368;
    if ( v101 != 0 && v101 != -4096 && v101 != -8192 )
      sub_BD60C0(&v99);
    v50 = v116;
  }
  sub_C7D6A0((__int64)v114, v50 << 6, 8);
  if ( v110 )
    j_j___libc_free_0(v110);
  sub_27C20B0((__int64)v150);
  return v91;
}
