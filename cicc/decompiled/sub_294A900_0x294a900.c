// Function: sub_294A900
// Address: 0x294a900
//
void __fastcall sub_294A900(_BYTE *a1, char a2, __int64 a3, __int64 a4, _BYTE *a5)
{
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // rbx
  const char *v10; // rsi
  __int64 v11; // r8
  __int64 v12; // r9
  const char *v13; // r14
  unsigned int *v14; // rax
  int v15; // ecx
  unsigned int *v16; // rdx
  unsigned __int64 v17; // rax
  unsigned int v18; // r12d
  int v19; // ebx
  _BYTE *v20; // rax
  __int64 v21; // rax
  unsigned int v22; // r12d
  __int64 v23; // r13
  unsigned __int8 *v24; // r12
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int8 *v27; // r13
  __int64 (__fastcall *v28)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v29; // r15
  __int64 (__fastcall *v30)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v31; // r14
  __int64 v32; // rax
  __int64 v33; // rdi
  unsigned __int64 v34; // rsi
  int v35; // eax
  __int64 v36; // rsi
  __int64 v37; // rax
  unsigned __int8 *v38; // r14
  __int64 (__fastcall *v39)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v40; // r12
  __int64 v41; // rax
  unsigned __int8 *v42; // r14
  __int64 (__fastcall *v43)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v44; // r13
  __int16 v45; // ax
  _QWORD *v46; // rax
  __int64 v47; // r9
  __int64 v48; // r14
  unsigned int *v49; // r15
  unsigned int *v50; // r12
  __int64 v51; // rdx
  unsigned int v52; // esi
  __int64 v53; // r12
  __int64 v54; // rdi
  const char *v55; // rsi
  __int64 v56; // r9
  const char *v57; // r12
  unsigned int *v58; // rax
  int v59; // ecx
  unsigned int *v60; // rdx
  _QWORD *v61; // rax
  unsigned int *v62; // r15
  unsigned int *v63; // r14
  __int64 v64; // rdx
  unsigned int v65; // esi
  _QWORD *v66; // rax
  unsigned int *v67; // r13
  unsigned int *v68; // r14
  __int64 v69; // rdx
  unsigned int v70; // esi
  _QWORD **v71; // rdx
  int v72; // ecx
  __int64 *v73; // rax
  __int64 v74; // rsi
  unsigned int *v75; // r13
  unsigned int *v76; // r12
  __int64 v77; // rdx
  unsigned int v78; // esi
  unsigned int *v79; // r14
  unsigned int *v80; // r12
  __int64 v81; // rdx
  unsigned int v82; // esi
  __int64 v83; // rax
  char v84; // al
  __int16 v85; // cx
  __int64 v86; // rax
  unsigned __int8 *v87; // r12
  __int64 (__fastcall *v88)(__int64, _BYTE *, unsigned __int8 *); // rax
  _QWORD *v89; // rax
  unsigned int *v90; // r13
  unsigned int *v91; // r12
  __int64 v92; // rdx
  unsigned int v93; // esi
  __int64 v94; // rbx
  int v95; // r14d
  __int64 v96; // rax
  __int64 v97; // rax
  unsigned __int8 *v98; // r13
  __int64 (__fastcall *v99)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v100; // r12
  __int64 v101; // rax
  unsigned __int8 *v102; // r13
  __int64 (__fastcall *v103)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v104; // r15
  __int16 v105; // ax
  _QWORD *v106; // rax
  __int64 v107; // r9
  __int64 v108; // r13
  unsigned int *v109; // r14
  unsigned int *v110; // r12
  __int64 v111; // rdx
  unsigned int v112; // esi
  _QWORD *v113; // rax
  unsigned int *v114; // r14
  unsigned int *v115; // r13
  __int64 v116; // rdx
  unsigned int v117; // esi
  _QWORD *v118; // rax
  unsigned int *v119; // rbx
  unsigned int *v120; // r13
  __int64 v121; // rdx
  unsigned int v122; // esi
  __int64 v123; // rax
  char v124; // al
  __int16 v125; // cx
  unsigned __int64 v126; // r8
  unsigned __int64 v127; // r13
  unsigned __int64 v128; // rsi
  __int64 v129; // [rsp+0h] [rbp-220h]
  __int64 v134; // [rsp+50h] [rbp-1D0h]
  char v135; // [rsp+5Eh] [rbp-1C2h]
  char v136; // [rsp+5Fh] [rbp-1C1h]
  __int64 v137; // [rsp+68h] [rbp-1B8h]
  unsigned __int64 v138; // [rsp+70h] [rbp-1B0h]
  unsigned int v139; // [rsp+78h] [rbp-1A8h]
  __int64 v140; // [rsp+78h] [rbp-1A8h]
  __int64 v141; // [rsp+80h] [rbp-1A0h]
  _BYTE *v142; // [rsp+88h] [rbp-198h]
  unsigned __int8 *v143; // [rsp+90h] [rbp-190h]
  __int16 v144; // [rsp+A0h] [rbp-180h]
  __int64 v145; // [rsp+A0h] [rbp-180h]
  __int16 v146; // [rsp+A8h] [rbp-178h]
  __int64 v147; // [rsp+A8h] [rbp-178h]
  __int64 v148; // [rsp+C8h] [rbp-158h]
  _BYTE v149[32]; // [rsp+D0h] [rbp-150h] BYREF
  __int16 v150; // [rsp+F0h] [rbp-130h]
  _QWORD v151[2]; // [rsp+100h] [rbp-120h] BYREF
  int v152; // [rsp+110h] [rbp-110h]
  __int16 v153; // [rsp+120h] [rbp-100h]
  unsigned __int64 v154; // [rsp+130h] [rbp-F0h] BYREF
  unsigned int v155; // [rsp+138h] [rbp-E8h]
  __int16 v156; // [rsp+150h] [rbp-D0h]
  unsigned int *v157; // [rsp+160h] [rbp-C0h] BYREF
  __int64 v158; // [rsp+168h] [rbp-B8h]
  _BYTE v159[32]; // [rsp+170h] [rbp-B0h] BYREF
  __int64 v160; // [rsp+190h] [rbp-90h]
  __int64 v161; // [rsp+198h] [rbp-88h]
  __int64 v162; // [rsp+1A0h] [rbp-80h]
  __int64 *v163; // [rsp+1A8h] [rbp-78h]
  void **v164; // [rsp+1B0h] [rbp-70h]
  void **v165; // [rsp+1B8h] [rbp-68h]
  __int64 v166; // [rsp+1C0h] [rbp-60h]
  int v167; // [rsp+1C8h] [rbp-58h]
  __int16 v168; // [rsp+1CCh] [rbp-54h]
  char v169; // [rsp+1CEh] [rbp-52h]
  __int64 v170; // [rsp+1D0h] [rbp-50h]
  __int64 v171; // [rsp+1D8h] [rbp-48h]
  void *v172; // [rsp+1E0h] [rbp-40h] BYREF
  void *v173; // [rsp+1E8h] [rbp-38h] BYREF

  v7 = *(_DWORD *)(a3 + 4) & 0x7FFFFFF;
  v141 = *(_QWORD *)(a3 - 32 * v7);
  v8 = *(_QWORD *)(v141 + 8);
  v142 = *(_BYTE **)(a3 + 32 * (1 - v7));
  v9 = *(_QWORD *)(a3 + 32 * (2 - v7));
  v134 = *(_QWORD *)(a3 + 32 * (3 - v7));
  v169 = 7;
  v163 = (__int64 *)sub_BD5C60(a3);
  v164 = &v172;
  v165 = &v173;
  v157 = (unsigned int *)v159;
  v172 = &unk_49DA100;
  v158 = 0x200000000LL;
  v166 = 0;
  v167 = 0;
  v168 = 512;
  v170 = 0;
  v171 = 0;
  v160 = 0;
  v161 = 0;
  LOWORD(v162) = 0;
  v173 = &unk_49DA0B0;
  sub_D5F1F0((__int64)&v157, a3);
  v10 = *(const char **)(a3 + 48);
  v154 = (unsigned __int64)v10;
  if ( v10 && (sub_B96E90((__int64)&v154, (__int64)v10, 1), (v13 = (const char *)v154) != 0) )
  {
    v14 = v157;
    v15 = v158;
    v16 = &v157[4 * (unsigned int)v158];
    if ( v157 != v16 )
    {
      while ( *v14 )
      {
        v14 += 4;
        if ( v16 == v14 )
          goto LABEL_126;
      }
      *((_QWORD *)v14 + 1) = v154;
      goto LABEL_8;
    }
LABEL_126:
    if ( (unsigned int)v158 >= (unsigned __int64)HIDWORD(v158) )
    {
      v128 = (unsigned int)v158 + 1LL;
      if ( HIDWORD(v158) < v128 )
      {
        sub_C8D5F0((__int64)&v157, v159, v128, 0x10u, v11, v12);
        v16 = &v157[4 * (unsigned int)v158];
      }
      *(_QWORD *)v16 = 0;
      *((_QWORD *)v16 + 1) = v13;
      v13 = (const char *)v154;
      LODWORD(v158) = v158 + 1;
    }
    else
    {
      if ( v16 )
      {
        *v16 = 0;
        *((_QWORD *)v16 + 1) = v13;
        v15 = v158;
        v13 = (const char *)v154;
      }
      LODWORD(v158) = v15 + 1;
    }
  }
  else
  {
    sub_93FB40((__int64)&v157, 0);
    v13 = (const char *)v154;
  }
  if ( v13 )
LABEL_8:
    sub_B91220((__int64)&v154, (__int64)v13);
  v17 = *(_QWORD *)(v9 + 24);
  if ( *(_DWORD *)(v9 + 32) > 0x40u )
    v17 = *(_QWORD *)v17;
  v136 = 0;
  if ( v17 )
  {
    _BitScanReverse64(&v17, v17);
    v136 = 1;
    v135 = 63 - (v17 ^ 0x3F);
  }
  v139 = *(_DWORD *)(v8 + 32);
  if ( *(_BYTE *)v134 > 0x15u )
  {
LABEL_19:
    if ( v139 == 1 || a2 == 1 )
    {
      v143 = 0;
    }
    else
    {
      v21 = sub_BCD140(v163, v139);
      v156 = 259;
      v154 = (unsigned __int64)"scalar_mask";
      v143 = (unsigned __int8 *)sub_A83570(&v157, v134, v21, (__int64)&v154);
    }
    if ( !v139 )
    {
LABEL_76:
      sub_B43D60((_QWORD *)a3);
      *a5 = 1;
      goto LABEL_77;
    }
    v147 = 0;
    while ( 1 )
    {
      if ( v143 )
      {
        v22 = v139 - 1 - v147;
        if ( !*a1 )
          v22 = v147;
        v155 = v139;
        v23 = 1LL << v22;
        if ( v139 <= 0x40 )
        {
          v154 = 0;
          goto LABEL_29;
        }
        sub_C43690((__int64)&v154, 0, 0);
        if ( v155 <= 0x40 )
LABEL_29:
          v154 |= v23;
        else
          *(_QWORD *)(v154 + 8LL * (v22 >> 6)) |= v23;
        v24 = (unsigned __int8 *)sub_ACCFD0(v163, (__int64)&v154);
        if ( v155 > 0x40 && v154 )
          j_j___libc_free_0_0(v154);
        v153 = 257;
        v25 = sub_BCD140(v163, v139);
        v26 = sub_ACD640(v25, 0, 0);
        v150 = 257;
        v27 = (unsigned __int8 *)v26;
        v28 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v164 + 2);
        if ( v28 != sub_9202E0 )
        {
          v29 = v28((__int64)v164, 28u, v143, v24);
          goto LABEL_38;
        }
        if ( *v143 > 0x15u || *v24 > 0x15u )
          goto LABEL_99;
        v29 = (unsigned __int8)sub_AC47B0(28) ? sub_AD5570(28, (__int64)v143, v24, 0, 0) : sub_AABE40(0x1Cu, v143, v24);
LABEL_38:
        if ( !v29 )
        {
LABEL_99:
          v156 = 257;
          v29 = sub_B504D0(28, (__int64)v143, (__int64)v24, (__int64)&v154, 0, 0);
          (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v165 + 2))(v165, v29, v149, v161, v162);
          v79 = v157;
          v80 = &v157[4 * (unsigned int)v158];
          if ( v157 != v80 )
          {
            do
            {
              v81 = *((_QWORD *)v79 + 1);
              v82 = *v79;
              v79 += 4;
              sub_B99FD0(v29, v82, v81);
            }
            while ( v80 != v79 );
          }
        }
        v30 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, unsigned __int8 *))*((_QWORD *)*v164 + 7);
        if ( v30 == sub_928890 )
        {
          if ( *(_BYTE *)v29 <= 0x15u && *v27 <= 0x15u )
          {
            v31 = sub_AAB310(0x21u, (unsigned __int8 *)v29, v27);
LABEL_43:
            if ( v31 )
              goto LABEL_44;
          }
          v156 = 257;
          v31 = (__int64)sub_BD2C40(72, unk_3F10FD0);
          if ( v31 )
          {
            v71 = *(_QWORD ***)(v29 + 8);
            v72 = *((unsigned __int8 *)v71 + 8);
            if ( (unsigned int)(v72 - 17) > 1 )
            {
              v74 = sub_BCB2A0(*v71);
            }
            else
            {
              BYTE4(v148) = (_BYTE)v72 == 18;
              LODWORD(v148) = *((_DWORD *)v71 + 8);
              v73 = (__int64 *)sub_BCB2A0(*v71);
              v74 = sub_BCE1B0(v73, v148);
            }
            sub_B523C0(v31, v74, 53, 33, v29, (__int64)v27, (__int64)&v154, 0, 0, 0);
          }
          (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v165 + 2))(
            v165,
            v31,
            v151,
            v161,
            v162);
          v75 = v157;
          v76 = &v157[4 * (unsigned int)v158];
          if ( v157 != v76 )
          {
            do
            {
              v77 = *((_QWORD *)v75 + 1);
              v78 = *v75;
              v75 += 4;
              sub_B99FD0(v31, v78, v77);
            }
            while ( v76 != v75 );
          }
          goto LABEL_44;
        }
        v31 = v30((__int64)v164, 33u, (_BYTE *)v29, v27);
        goto LABEL_43;
      }
      v151[0] = "Mask";
      v152 = v147;
      v153 = 2307;
      v86 = sub_BCB2E0(v163);
      v87 = (unsigned __int8 *)sub_ACD640(v86, v147, 0);
      v88 = (__int64 (__fastcall *)(__int64, _BYTE *, unsigned __int8 *))*((_QWORD *)*v164 + 12);
      if ( v88 != sub_948070 )
        break;
      if ( *(_BYTE *)v134 <= 0x15u && *v87 <= 0x15u )
      {
        v31 = sub_AD5840(v134, v87, 0);
        goto LABEL_116;
      }
LABEL_117:
      v156 = 257;
      v89 = sub_BD2C40(72, 2u);
      v31 = (__int64)v89;
      if ( v89 )
        sub_B4DE80((__int64)v89, v134, (__int64)v87, (__int64)&v154, 0, 0);
      (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v165 + 2))(v165, v31, v151, v161, v162);
      v90 = v157;
      v91 = &v157[4 * (unsigned int)v158];
      if ( v157 != v91 )
      {
        do
        {
          v92 = *((_QWORD *)v90 + 1);
          v93 = *v90;
          v90 += 4;
          sub_B99FD0(v31, v93, v92);
        }
        while ( v91 != v90 );
      }
LABEL_44:
      v32 = v137;
      LOWORD(v32) = 0;
      v137 = v32;
      v138 = sub_F38250(v31, (__int64 *)(a3 + 24), v32, 0, 0, a4, 0, 0);
      v33 = *(_QWORD *)(v138 + 40);
      v154 = (unsigned __int64)"cond.store";
      v156 = 259;
      sub_BD6B50((unsigned __int8 *)v33, (const char **)&v154);
      v34 = *(_QWORD *)(v33 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v34 == v33 + 48 )
      {
        v36 = 0;
      }
      else
      {
        if ( !v34 )
          BUG();
        v35 = *(unsigned __int8 *)(v34 - 24);
        v36 = v34 - 24;
        if ( (unsigned int)(v35 - 30) >= 0xB )
          v36 = 0;
      }
      sub_D5F1F0((__int64)&v157, v36);
      v153 = 2307;
      v151[0] = "Elt";
      v152 = v147;
      v37 = sub_BCB2E0(v163);
      v38 = (unsigned __int8 *)sub_ACD640(v37, v147, 0);
      v39 = (__int64 (__fastcall *)(__int64, _BYTE *, unsigned __int8 *))*((_QWORD *)*v164 + 12);
      if ( v39 != sub_948070 )
      {
        v40 = v39((__int64)v164, (_BYTE *)v141, v38);
LABEL_52:
        if ( v40 )
          goto LABEL_53;
        goto LABEL_85;
      }
      if ( *(_BYTE *)v141 <= 0x15u && *v38 <= 0x15u )
      {
        v40 = sub_AD5840(v141, v38, 0);
        goto LABEL_52;
      }
LABEL_85:
      v156 = 257;
      v66 = sub_BD2C40(72, 2u);
      v40 = (__int64)v66;
      if ( v66 )
        sub_B4DE80((__int64)v66, v141, (__int64)v38, (__int64)&v154, 0, 0);
      (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v165 + 2))(v165, v40, v151, v161, v162);
      v67 = v157;
      v68 = &v157[4 * (unsigned int)v158];
      if ( v157 != v68 )
      {
        do
        {
          v69 = *((_QWORD *)v67 + 1);
          v70 = *v67;
          v67 += 4;
          sub_B99FD0(v40, v70, v69);
        }
        while ( v68 != v67 );
      }
LABEL_53:
      v151[0] = "Ptr";
      v153 = 2307;
      v152 = v147;
      v41 = sub_BCB2E0(v163);
      v42 = (unsigned __int8 *)sub_ACD640(v41, v147, 0);
      v43 = (__int64 (__fastcall *)(__int64, _BYTE *, unsigned __int8 *))*((_QWORD *)*v164 + 12);
      if ( v43 != sub_948070 )
      {
        v44 = v43((__int64)v164, v142, v42);
LABEL_57:
        if ( v44 )
          goto LABEL_58;
        goto LABEL_80;
      }
      if ( *v142 <= 0x15u && *v42 <= 0x15u )
      {
        v44 = sub_AD5840((__int64)v142, v42, 0);
        goto LABEL_57;
      }
LABEL_80:
      v156 = 257;
      v61 = sub_BD2C40(72, 2u);
      v44 = (__int64)v61;
      if ( v61 )
        sub_B4DE80((__int64)v61, (__int64)v142, (__int64)v42, (__int64)&v154, 0, 0);
      (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v165 + 2))(v165, v44, v151, v161, v162);
      v62 = v157;
      v63 = &v157[4 * (unsigned int)v158];
      if ( v157 != v63 )
      {
        do
        {
          v64 = *((_QWORD *)v62 + 1);
          v65 = *v62;
          v62 += 4;
          sub_B99FD0(v44, v65, v64);
        }
        while ( v63 != v62 );
      }
LABEL_58:
      HIBYTE(v45) = HIBYTE(v144);
      LOBYTE(v45) = v135;
      v144 = v45;
      if ( !v136 )
      {
        v83 = sub_AA4E30(v160);
        v84 = sub_AE5020(v83, *(_QWORD *)(v40 + 8));
        HIBYTE(v85) = HIBYTE(v144);
        LOBYTE(v85) = v84;
        v144 = v85;
      }
      v156 = 257;
      v46 = sub_BD2C40(80, unk_3F10A10);
      v48 = (__int64)v46;
      if ( v46 )
        sub_B4D3C0((__int64)v46, v40, v44, 0, v144, v47, 0, 0);
      (*((void (__fastcall **)(void **, __int64, unsigned __int64 *, __int64, __int64))*v165 + 2))(
        v165,
        v48,
        &v154,
        v161,
        v162);
      v49 = v157;
      v50 = &v157[4 * (unsigned int)v158];
      if ( v157 != v50 )
      {
        do
        {
          v51 = *((_QWORD *)v49 + 1);
          v52 = *v49;
          v49 += 4;
          sub_B99FD0(v48, v52, v51);
        }
        while ( v50 != v49 );
      }
      v53 = sub_B46EC0(v138, 0);
      v156 = 259;
      v154 = (unsigned __int64)"else";
      sub_BD6B50((unsigned __int8 *)v53, (const char **)&v154);
      v54 = *(_QWORD *)(v53 + 56);
      v160 = v53;
      v161 = v54;
      LOWORD(v162) = 1;
      if ( v54 == v53 + 48 )
        goto LABEL_75;
      if ( v54 )
        v54 -= 24;
      v55 = *(const char **)sub_B46C60(v54);
      v154 = (unsigned __int64)v55;
      if ( !v55 || (sub_B96E90((__int64)&v154, (__int64)v55, 1), (v57 = (const char *)v154) == 0) )
      {
        sub_93FB40((__int64)&v157, 0);
        v57 = (const char *)v154;
        goto LABEL_103;
      }
      v58 = v157;
      v59 = v158;
      v60 = &v157[4 * (unsigned int)v158];
      if ( v157 == v60 )
      {
LABEL_105:
        if ( (unsigned int)v158 >= (unsigned __int64)HIDWORD(v158) )
        {
          v126 = (unsigned int)v158 + 1LL;
          v127 = v129 & 0xFFFFFFFF00000000LL;
          v129 &= 0xFFFFFFFF00000000LL;
          if ( HIDWORD(v158) < v126 )
          {
            sub_C8D5F0((__int64)&v157, v159, v126, 0x10u, v126, v56);
            v60 = &v157[4 * (unsigned int)v158];
          }
          *(_QWORD *)v60 = v127;
          *((_QWORD *)v60 + 1) = v57;
          v57 = (const char *)v154;
          LODWORD(v158) = v158 + 1;
        }
        else
        {
          if ( v60 )
          {
            *v60 = 0;
            *((_QWORD *)v60 + 1) = v57;
            v59 = v158;
            v57 = (const char *)v154;
          }
          LODWORD(v158) = v59 + 1;
        }
LABEL_103:
        if ( !v57 )
          goto LABEL_75;
        goto LABEL_74;
      }
      while ( *v58 )
      {
        v58 += 4;
        if ( v60 == v58 )
          goto LABEL_105;
      }
      *((_QWORD *)v58 + 1) = v154;
LABEL_74:
      sub_B91220((__int64)&v154, (__int64)v57);
LABEL_75:
      if ( v139 == ++v147 )
        goto LABEL_76;
    }
    v31 = v88((__int64)v164, (_BYTE *)v134, v87);
LABEL_116:
    if ( v31 )
      goto LABEL_44;
    goto LABEL_117;
  }
  v18 = 0;
  v19 = *(_DWORD *)(*(_QWORD *)(v134 + 8) + 32LL);
  if ( v19 )
  {
    do
    {
      v20 = (_BYTE *)sub_AD69F0((unsigned __int8 *)v134, v18);
      if ( !v20 || *v20 != 17 )
        goto LABEL_19;
    }
    while ( v19 != ++v18 );
  }
  v145 = v139;
  v94 = 0;
  if ( v139 )
  {
    while ( 1 )
    {
      v95 = v94;
      v96 = sub_AD69F0((unsigned __int8 *)v134, (unsigned int)v94);
      if ( !sub_AC30F0(v96) )
        break;
LABEL_152:
      if ( ++v94 == v145 )
        goto LABEL_153;
    }
    v152 = v94;
    v151[0] = "Elt";
    v153 = 2307;
    v97 = sub_BCB2E0(v163);
    v98 = (unsigned __int8 *)sub_ACD640(v97, v94, 0);
    v99 = (__int64 (__fastcall *)(__int64, _BYTE *, unsigned __int8 *))*((_QWORD *)*v164 + 12);
    if ( v99 == sub_948070 )
    {
      if ( *(_BYTE *)v141 > 0x15u || *v98 > 0x15u )
      {
LABEL_159:
        v156 = 257;
        v118 = sub_BD2C40(72, 2u);
        v100 = (__int64)v118;
        if ( v118 )
          sub_B4DE80((__int64)v118, v141, (__int64)v98, (__int64)&v154, 0, 0);
        (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v165 + 2))(v165, v100, v151, v161, v162);
        if ( v157 != &v157[4 * (unsigned int)v158] )
        {
          v140 = v94;
          v119 = v157;
          v120 = &v157[4 * (unsigned int)v158];
          do
          {
            v121 = *((_QWORD *)v119 + 1);
            v122 = *v119;
            v119 += 4;
            sub_B99FD0(v100, v122, v121);
          }
          while ( v120 != v119 );
          v94 = v140;
        }
LABEL_141:
        v152 = v95;
        v153 = 2307;
        v151[0] = "Ptr";
        v101 = sub_BCB2E0(v163);
        v102 = (unsigned __int8 *)sub_ACD640(v101, v94, 0);
        v103 = (__int64 (__fastcall *)(__int64, _BYTE *, unsigned __int8 *))*((_QWORD *)*v164 + 12);
        if ( v103 == sub_948070 )
        {
          if ( *v142 > 0x15u || *v102 > 0x15u )
            goto LABEL_154;
          v104 = sub_AD5840((__int64)v142, v102, 0);
        }
        else
        {
          v104 = v103((__int64)v164, v142, v102);
        }
        if ( v104 )
        {
LABEL_146:
          HIBYTE(v105) = HIBYTE(v146);
          LOBYTE(v105) = v135;
          v146 = v105;
          if ( !v136 )
          {
            v123 = sub_AA4E30(v160);
            v124 = sub_AE5020(v123, *(_QWORD *)(v100 + 8));
            HIBYTE(v125) = HIBYTE(v146);
            LOBYTE(v125) = v124;
            v146 = v125;
          }
          v156 = 257;
          v106 = sub_BD2C40(80, unk_3F10A10);
          v108 = (__int64)v106;
          if ( v106 )
            sub_B4D3C0((__int64)v106, v100, v104, 0, v146, v107, 0, 0);
          (*((void (__fastcall **)(void **, __int64, unsigned __int64 *, __int64, __int64))*v165 + 2))(
            v165,
            v108,
            &v154,
            v161,
            v162);
          v109 = v157;
          v110 = &v157[4 * (unsigned int)v158];
          if ( v157 != v110 )
          {
            do
            {
              v111 = *((_QWORD *)v109 + 1);
              v112 = *v109;
              v109 += 4;
              sub_B99FD0(v108, v112, v111);
            }
            while ( v110 != v109 );
          }
          goto LABEL_152;
        }
LABEL_154:
        v156 = 257;
        v113 = sub_BD2C40(72, 2u);
        v104 = (__int64)v113;
        if ( v113 )
          sub_B4DE80((__int64)v113, (__int64)v142, (__int64)v102, (__int64)&v154, 0, 0);
        (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v165 + 2))(v165, v104, v151, v161, v162);
        v114 = v157;
        v115 = &v157[4 * (unsigned int)v158];
        if ( v157 != v115 )
        {
          do
          {
            v116 = *((_QWORD *)v114 + 1);
            v117 = *v114;
            v114 += 4;
            sub_B99FD0(v104, v117, v116);
          }
          while ( v115 != v114 );
        }
        goto LABEL_146;
      }
      v100 = sub_AD5840(v141, v98, 0);
    }
    else
    {
      v100 = v99((__int64)v164, (_BYTE *)v141, v98);
    }
    if ( v100 )
      goto LABEL_141;
    goto LABEL_159;
  }
LABEL_153:
  sub_B43D60((_QWORD *)a3);
LABEL_77:
  nullsub_61();
  v172 = &unk_49DA100;
  nullsub_63();
  if ( v157 != (unsigned int *)v159 )
    _libc_free((unsigned __int64)v157);
}
