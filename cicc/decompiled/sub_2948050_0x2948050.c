// Function: sub_2948050
// Address: 0x2948050
//
void __fastcall sub_2948050(_BYTE *a1, char a2, __int64 a3, __int64 a4, _BYTE *a5)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  unsigned __int64 v8; // rax
  char v9; // r12
  __int64 v10; // rbx
  const char *v11; // rsi
  __int64 v12; // r8
  __int64 v13; // r9
  const char *v14; // r15
  unsigned __int64 v15; // rsi
  unsigned int *v16; // rax
  __int64 v17; // rcx
  unsigned int *v18; // rdx
  const char *v19; // rax
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rax
  int v23; // ebx
  unsigned int v24; // r12d
  _BYTE *v25; // rax
  unsigned int v26; // ebx
  __int64 v27; // r12
  unsigned __int8 *v28; // r15
  __int64 v29; // rax
  __int64 (__fastcall *v30)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v31; // r12
  __int64 v32; // r12
  __int64 v33; // rax
  __int64 v34; // rdi
  unsigned __int64 v35; // rsi
  int v36; // eax
  __int64 v37; // rsi
  __int64 v38; // rax
  unsigned __int8 *v39; // r12
  __int64 (__fastcall *v40)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v41; // r15
  __int64 v42; // rbx
  __int16 v43; // ax
  _QWORD *v44; // rax
  __int64 v45; // r9
  __int64 v46; // r12
  unsigned int *v47; // r15
  unsigned int *v48; // rbx
  __int64 v49; // rdx
  unsigned int v50; // esi
  __int64 v51; // r12
  __int64 v52; // rax
  _QWORD *v53; // rax
  unsigned int *v54; // r12
  unsigned int *v55; // rbx
  __int64 v56; // rdx
  unsigned int v57; // esi
  unsigned int *v58; // r15
  unsigned int *v59; // rbx
  __int64 v60; // rdx
  unsigned int v61; // esi
  __int64 v62; // rax
  unsigned __int8 *v63; // r15
  __int64 (__fastcall *v64)(__int64, _BYTE *, unsigned __int8 *); // rax
  _QWORD *v65; // rax
  unsigned int *v66; // r15
  unsigned int *v67; // rbx
  __int64 v68; // rdx
  unsigned int v69; // esi
  __int64 v70; // r15
  unsigned int v71; // r12d
  __int64 v72; // rax
  __int64 v73; // rax
  unsigned __int8 *v74; // rbx
  __int64 (__fastcall *v75)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v76; // r13
  __int64 v77; // rax
  __int64 v78; // rbx
  _QWORD *v79; // rax
  __int64 v80; // r9
  __int64 v81; // r12
  unsigned int *v82; // r13
  unsigned int *v83; // rbx
  __int64 v84; // rdx
  unsigned int v85; // esi
  __int64 v86; // rax
  _QWORD *v87; // rax
  unsigned int *v88; // r12
  unsigned int *v89; // rbx
  __int64 v90; // rdx
  unsigned int v91; // esi
  unsigned __int8 *v92; // rax
  __int64 v93; // r9
  unsigned __int8 *v94; // r14
  unsigned int *v95; // r12
  unsigned int *v96; // rbx
  __int64 v97; // rdx
  unsigned int v98; // esi
  __int64 v99; // rdx
  __int64 v100; // rax
  unsigned __int8 *v101; // r15
  __int64 (__fastcall *v102)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v103; // rbx
  unsigned __int8 *v104; // rbx
  unsigned __int8 *v105; // rdi
  unsigned __int64 v106; // rax
  int v107; // edx
  __int64 v108; // rsi
  __int64 v109; // rax
  unsigned __int8 *v110; // rax
  __int64 v111; // r9
  unsigned __int8 *v112; // r14
  unsigned int *v113; // r12
  unsigned int *v114; // rbx
  __int64 v115; // rdx
  unsigned int v116; // esi
  _QWORD *v117; // rax
  unsigned int *v118; // r12
  unsigned int *v119; // r13
  __int64 v120; // rdx
  unsigned int v121; // esi
  __int64 v126; // [rsp+50h] [rbp-1B0h]
  __int64 v127; // [rsp+58h] [rbp-1A8h]
  _BYTE *v128; // [rsp+68h] [rbp-198h]
  unsigned __int64 v129; // [rsp+68h] [rbp-198h]
  __int64 v130; // [rsp+70h] [rbp-190h]
  __int64 v131; // [rsp+78h] [rbp-188h]
  char v132; // [rsp+83h] [rbp-17Dh]
  unsigned int v133; // [rsp+84h] [rbp-17Ch]
  char v134; // [rsp+88h] [rbp-178h]
  unsigned __int8 *v135; // [rsp+88h] [rbp-178h]
  __int16 v136; // [rsp+90h] [rbp-170h]
  __int64 v137; // [rsp+98h] [rbp-168h]
  __int64 v138; // [rsp+A0h] [rbp-160h]
  __int64 v140; // [rsp+A8h] [rbp-158h]
  char v141; // [rsp+A8h] [rbp-158h]
  _BYTE v142[32]; // [rsp+B0h] [rbp-150h] BYREF
  __int16 v143; // [rsp+D0h] [rbp-130h]
  _QWORD v144[4]; // [rsp+E0h] [rbp-120h] BYREF
  __int16 v145; // [rsp+100h] [rbp-100h]
  char *v146; // [rsp+110h] [rbp-F0h] BYREF
  __int64 v147; // [rsp+118h] [rbp-E8h]
  __int16 v148; // [rsp+130h] [rbp-D0h]
  unsigned int *v149; // [rsp+140h] [rbp-C0h] BYREF
  __int64 v150; // [rsp+148h] [rbp-B8h]
  _BYTE v151[32]; // [rsp+150h] [rbp-B0h] BYREF
  __int64 v152; // [rsp+170h] [rbp-90h]
  __int64 v153; // [rsp+178h] [rbp-88h]
  __int64 v154; // [rsp+180h] [rbp-80h]
  __int64 *v155; // [rsp+188h] [rbp-78h]
  void **v156; // [rsp+190h] [rbp-70h]
  void **v157; // [rsp+198h] [rbp-68h]
  __int64 v158; // [rsp+1A0h] [rbp-60h]
  int v159; // [rsp+1A8h] [rbp-58h]
  __int16 v160; // [rsp+1ACh] [rbp-54h]
  char v161; // [rsp+1AEh] [rbp-52h]
  __int64 v162; // [rsp+1B0h] [rbp-50h]
  __int64 v163; // [rsp+1B8h] [rbp-48h]
  void *v164; // [rsp+1C0h] [rbp-40h] BYREF
  void *v165; // [rsp+1C8h] [rbp-38h] BYREF

  v6 = *(_DWORD *)(a3 + 4) & 0x7FFFFFF;
  v138 = *(_QWORD *)(a3 - 32 * v6);
  v130 = *(_QWORD *)(a3 + 32 * (1 - v6));
  v7 = *(_QWORD *)(a3 + 32 * (2 - v6));
  v137 = *(_QWORD *)(a3 + 32 * (3 - v6));
  v8 = *(_QWORD *)(v7 + 24);
  if ( *(_DWORD *)(v7 + 32) > 0x40u )
    v8 = *(_QWORD *)v8;
  v9 = 0;
  if ( v8 )
  {
    _BitScanReverse64(&v8, v8);
    v9 = 63 - (v8 ^ 0x3F);
  }
  v10 = *(_QWORD *)(v138 + 8);
  v131 = *(_QWORD *)(v10 + 24);
  v155 = (__int64 *)sub_BD5C60(a3);
  v156 = &v164;
  v157 = &v165;
  v149 = (unsigned int *)v151;
  v164 = &unk_49DA100;
  v150 = 0x200000000LL;
  v158 = 0;
  v159 = 0;
  v160 = 512;
  v161 = 7;
  v162 = 0;
  v163 = 0;
  v152 = 0;
  v153 = 0;
  LOWORD(v154) = 0;
  v165 = &unk_49DA0B0;
  sub_D5F1F0((__int64)&v149, a3);
  v11 = *(const char **)(a3 + 48);
  v146 = (char *)v11;
  if ( !v11 || (sub_B96E90((__int64)&v146, (__int64)v11, 1), (v14 = v146) == 0) )
  {
    v15 = 0;
    sub_93FB40((__int64)&v149, 0);
    v14 = v146;
    goto LABEL_88;
  }
  v15 = (unsigned int)v150;
  v16 = v149;
  v17 = (unsigned int)v150;
  v18 = &v149[4 * (unsigned int)v150];
  if ( v149 == v18 )
  {
LABEL_90:
    if ( (unsigned int)v150 >= (unsigned __int64)HIDWORD(v150) )
    {
      v15 = (unsigned int)v150 + 1LL;
      if ( HIDWORD(v150) < v15 )
      {
        v15 = (unsigned __int64)v151;
        sub_C8D5F0((__int64)&v149, v151, (unsigned int)v150 + 1LL, 0x10u, v12, v13);
        v18 = &v149[4 * (unsigned int)v150];
      }
      *(_QWORD *)v18 = 0;
      *((_QWORD *)v18 + 1) = v14;
      v14 = v146;
      LODWORD(v150) = v150 + 1;
    }
    else
    {
      if ( v18 )
      {
        *v18 = 0;
        *((_QWORD *)v18 + 1) = v14;
        LODWORD(v17) = v150;
        v14 = v146;
      }
      v17 = (unsigned int)(v17 + 1);
      LODWORD(v150) = v17;
    }
LABEL_88:
    if ( !v14 )
      goto LABEL_13;
    goto LABEL_12;
  }
  while ( *v16 )
  {
    v16 += 4;
    if ( v18 == v16 )
      goto LABEL_90;
  }
  *((_QWORD *)v16 + 1) = v146;
LABEL_12:
  v15 = (unsigned __int64)v14;
  sub_B91220((__int64)&v146, (__int64)v14);
LABEL_13:
  if ( *(_BYTE *)v137 <= 0x15u && sub_AD7930((_BYTE *)v137, v15, (__int64)v18, v17, v12) )
  {
    v148 = 257;
    v92 = (unsigned __int8 *)sub_BD2C40(80, unk_3F10A10);
    v94 = v92;
    if ( v92 )
      sub_B4D3C0((__int64)v92, v138, v130, 0, v9, v93, 0, 0);
    (*((void (__fastcall **)(void **, unsigned __int8 *, char **, __int64, __int64))*v157 + 2))(
      v157,
      v94,
      &v146,
      v153,
      v154);
    v95 = v149;
    v96 = &v149[4 * (unsigned int)v150];
    if ( v149 != v96 )
    {
      do
      {
        v97 = *((_QWORD *)v95 + 1);
        v98 = *v95;
        v95 += 4;
        sub_B99FD0((__int64)v94, v98, v97);
      }
      while ( v96 != v95 );
    }
    sub_BD6B90(v94, (unsigned __int8 *)a3);
    sub_B47C00((__int64)v94, a3, 0, 0);
    sub_B43D60((_QWORD *)a3);
    goto LABEL_61;
  }
  v19 = (const char *)sub_BCAE30(v131);
  v147 = v20;
  v146 = (char *)v19;
  v21 = sub_CA1930(&v146);
  v132 = -1;
  v22 = -(__int64)((v21 >> 3) | (1LL << v9)) & ((v21 >> 3) | (1LL << v9));
  if ( v22 )
  {
    _BitScanReverse64(&v22, v22);
    v132 = 63 - (v22 ^ 0x3F);
  }
  v133 = *(_DWORD *)(v10 + 32);
  if ( *(_BYTE *)v137 <= 0x15u )
  {
    v23 = *(_DWORD *)(*(_QWORD *)(v137 + 8) + 32LL);
    if ( v23 )
    {
      v134 = v9;
      v24 = 0;
      while ( 1 )
      {
        v25 = (_BYTE *)sub_AD69F0((unsigned __int8 *)v137, v24);
        if ( !v25 || *v25 != 17 )
          break;
        if ( v23 == ++v24 )
          goto LABEL_95;
      }
      v9 = v134;
      goto LABEL_24;
    }
LABEL_95:
    v70 = 0;
    if ( !v133 )
    {
LABEL_107:
      sub_B43D60((_QWORD *)a3);
      goto LABEL_61;
    }
    while ( 1 )
    {
      v71 = v70;
      v72 = sub_AD69F0((unsigned __int8 *)v137, (unsigned int)v70);
      if ( !sub_AC30F0(v72) )
        break;
LABEL_106:
      if ( v133 == ++v70 )
        goto LABEL_107;
    }
    v145 = 257;
    v73 = sub_BCB2E0(v155);
    v74 = (unsigned __int8 *)sub_ACD640(v73, v70, 0);
    v75 = (__int64 (__fastcall *)(__int64, _BYTE *, unsigned __int8 *))*((_QWORD *)*v156 + 12);
    if ( v75 == sub_948070 )
    {
      if ( *(_BYTE *)v138 > 0x15u || *v74 > 0x15u )
        goto LABEL_109;
      v76 = sub_AD5840(v138, v74, 0);
    }
    else
    {
      v76 = v75((__int64)v156, (_BYTE *)v138, v74);
    }
    if ( v76 )
    {
LABEL_102:
      v148 = 257;
      v77 = sub_94B2B0(&v149, v131, v130, v71, (__int64)&v146);
      v148 = 257;
      v78 = v77;
      v79 = sub_BD2C40(80, unk_3F10A10);
      v81 = (__int64)v79;
      if ( v79 )
        sub_B4D3C0((__int64)v79, v76, v78, 0, v132, v80, 0, 0);
      (*((void (__fastcall **)(void **, __int64, char **, __int64, __int64))*v157 + 2))(v157, v81, &v146, v153, v154);
      v82 = v149;
      v83 = &v149[4 * (unsigned int)v150];
      if ( v149 != v83 )
      {
        do
        {
          v84 = *((_QWORD *)v82 + 1);
          v85 = *v82;
          v82 += 4;
          sub_B99FD0(v81, v85, v84);
        }
        while ( v83 != v82 );
      }
      goto LABEL_106;
    }
LABEL_109:
    v148 = 257;
    v87 = sub_BD2C40(72, 2u);
    v76 = (__int64)v87;
    if ( v87 )
      sub_B4DE80((__int64)v87, v138, (__int64)v74, (__int64)&v146, 0, 0);
    (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v157 + 2))(v157, v76, v144, v153, v154);
    if ( v149 != &v149[4 * (unsigned int)v150] )
    {
      v88 = v149;
      v89 = &v149[4 * (unsigned int)v150];
      do
      {
        v90 = *((_QWORD *)v88 + 1);
        v91 = *v88;
        v88 += 4;
        sub_B99FD0(v76, v91, v90);
      }
      while ( v89 != v88 );
      v71 = v70;
    }
    goto LABEL_102;
  }
LABEL_24:
  if ( sub_9B7DA0((char *)v137, 0, 0) )
  {
    v144[0] = sub_BD5D20(v137);
    v145 = 773;
    v144[1] = v99;
    v144[2] = ".first";
    v100 = sub_BCB2E0(v155);
    v101 = (unsigned __int8 *)sub_ACD640(v100, 0, 0);
    v102 = (__int64 (__fastcall *)(__int64, _BYTE *, unsigned __int8 *))*((_QWORD *)*v156 + 12);
    if ( v102 == sub_948070 )
    {
      if ( *(_BYTE *)v137 > 0x15u || *v101 > 0x15u )
        goto LABEL_136;
      v103 = sub_AD5840(v137, v101, 0);
    }
    else
    {
      v103 = v102((__int64)v156, (_BYTE *)v137, v101);
    }
    if ( v103 )
    {
LABEL_127:
      v104 = *(unsigned __int8 **)(sub_F38250(v103, (__int64 *)(a3 + 24), 0, 0, 0, a4, 0, 0) + 40);
      v146 = "cond.store";
      v105 = v104;
      v148 = 259;
      v104 += 48;
      sub_BD6B50(v105, (const char **)&v146);
      v106 = *(_QWORD *)v104 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (unsigned __int8 *)v106 == v104 )
      {
        v108 = 0;
      }
      else
      {
        if ( !v106 )
          BUG();
        v107 = *(unsigned __int8 *)(v106 - 24);
        v108 = 0;
        v109 = v106 - 24;
        if ( (unsigned int)(v107 - 30) < 0xB )
          v108 = v109;
      }
      sub_D5F1F0((__int64)&v149, v108);
      v148 = 257;
      v110 = (unsigned __int8 *)sub_BD2C40(80, unk_3F10A10);
      v112 = v110;
      if ( v110 )
        sub_B4D3C0((__int64)v110, v138, v130, 0, v9, v111, 0, 0);
      (*((void (__fastcall **)(void **, unsigned __int8 *, char **, __int64, __int64))*v157 + 2))(
        v157,
        v112,
        &v146,
        v153,
        v154);
      v113 = v149;
      v114 = &v149[4 * (unsigned int)v150];
      if ( v149 != v114 )
      {
        do
        {
          v115 = *((_QWORD *)v113 + 1);
          v116 = *v113;
          v113 += 4;
          sub_B99FD0((__int64)v112, v116, v115);
        }
        while ( v114 != v113 );
      }
      sub_BD6B90(v112, (unsigned __int8 *)a3);
      sub_B47C00((__int64)v112, a3, 0, 0);
      sub_B43D60((_QWORD *)a3);
      *a5 = 1;
      goto LABEL_61;
    }
LABEL_136:
    v148 = 257;
    v117 = sub_BD2C40(72, 2u);
    v103 = (__int64)v117;
    if ( v117 )
      sub_B4DE80((__int64)v117, v137, (__int64)v101, (__int64)&v146, 0, 0);
    (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v157 + 2))(v157, v103, v144, v153, v154);
    if ( v149 != &v149[4 * (unsigned int)v150] )
    {
      v141 = v9;
      v118 = &v149[4 * (unsigned int)v150];
      v119 = v149;
      do
      {
        v120 = *((_QWORD *)v119 + 1);
        v121 = *v119;
        v119 += 4;
        sub_B99FD0(v103, v121, v120);
      }
      while ( v118 != v119 );
      v9 = v141;
    }
    goto LABEL_127;
  }
  if ( v133 == 1 || a2 == 1 )
  {
    v135 = 0;
  }
  else
  {
    v86 = sub_BCD140(v155, v133);
    v146 = "scalar_mask";
    v148 = 259;
    v135 = (unsigned __int8 *)sub_A83570(&v149, v137, v86, (__int64)&v146);
  }
  if ( v133 )
  {
    v140 = 0;
    while ( 1 )
    {
      if ( v135 )
      {
        v26 = v133 - 1 - v140;
        if ( !*a1 )
          v26 = v140;
        LODWORD(v147) = v133;
        v27 = 1LL << v26;
        if ( v133 <= 0x40 )
        {
          v146 = 0;
          goto LABEL_35;
        }
        sub_C43690((__int64)&v146, 0, 0);
        if ( (unsigned int)v147 <= 0x40 )
LABEL_35:
          v146 = (char *)(v27 | (unsigned __int64)v146);
        else
          *(_QWORD *)&v146[8 * (v26 >> 6)] |= v27;
        v28 = (unsigned __int8 *)sub_ACCFD0(v155, (__int64)&v146);
        if ( (unsigned int)v147 > 0x40 && v146 )
          j_j___libc_free_0_0((unsigned __int64)v146);
        v145 = 257;
        v29 = sub_BCD140(v155, v133);
        v128 = (_BYTE *)sub_ACD640(v29, 0, 0);
        v143 = 257;
        v30 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v156 + 2);
        if ( v30 == sub_9202E0 )
        {
          if ( *v135 > 0x15u || *v28 > 0x15u )
            goto LABEL_71;
          v31 = (unsigned __int8)sub_AC47B0(28)
              ? sub_AD5570(28, (__int64)v135, v28, 0, 0)
              : sub_AABE40(0x1Cu, v135, v28);
LABEL_44:
          if ( !v31 )
          {
LABEL_71:
            v148 = 257;
            v31 = sub_B504D0(28, (__int64)v135, (__int64)v28, (__int64)&v146, 0, 0);
            (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v157 + 2))(
              v157,
              v31,
              v142,
              v153,
              v154);
            v58 = v149;
            v59 = &v149[4 * (unsigned int)v150];
            if ( v149 != v59 )
            {
              do
              {
                v60 = *((_QWORD *)v58 + 1);
                v61 = *v58;
                v58 += 4;
                sub_B99FD0(v31, v61, v60);
              }
              while ( v59 != v58 );
            }
          }
          v32 = sub_92B530(&v149, 0x21u, v31, v128, (__int64)v144);
          goto LABEL_46;
        }
        v31 = v30((__int64)v156, 28u, v135, v28);
        goto LABEL_44;
      }
      v145 = 257;
      v62 = sub_BCB2E0(v155);
      v63 = (unsigned __int8 *)sub_ACD640(v62, v140, 0);
      v64 = (__int64 (__fastcall *)(__int64, _BYTE *, unsigned __int8 *))*((_QWORD *)*v156 + 12);
      if ( v64 != sub_948070 )
        break;
      if ( *(_BYTE *)v137 <= 0x15u && *v63 <= 0x15u )
      {
        v32 = sub_AD5840(v137, v63, 0);
        goto LABEL_80;
      }
LABEL_81:
      v148 = 257;
      v65 = sub_BD2C40(72, 2u);
      v32 = (__int64)v65;
      if ( v65 )
        sub_B4DE80((__int64)v65, v137, (__int64)v63, (__int64)&v146, 0, 0);
      (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v157 + 2))(v157, v32, v144, v153, v154);
      v66 = v149;
      v67 = &v149[4 * (unsigned int)v150];
      if ( v149 != v67 )
      {
        do
        {
          v68 = *((_QWORD *)v66 + 1);
          v69 = *v66;
          v66 += 4;
          sub_B99FD0(v32, v69, v68);
        }
        while ( v67 != v66 );
      }
LABEL_46:
      v33 = v127;
      LOWORD(v33) = 0;
      v127 = v33;
      v129 = sub_F38250(v32, (__int64 *)(a3 + 24), v33, 0, 0, a4, 0, 0);
      v34 = *(_QWORD *)(v129 + 40);
      v146 = "cond.store";
      v148 = 259;
      sub_BD6B50((unsigned __int8 *)v34, (const char **)&v146);
      v35 = *(_QWORD *)(v34 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v35 == v34 + 48 )
      {
        v37 = 0;
      }
      else
      {
        if ( !v35 )
          BUG();
        v36 = *(unsigned __int8 *)(v35 - 24);
        v37 = v35 - 24;
        if ( (unsigned int)(v36 - 30) >= 0xB )
          v37 = 0;
      }
      sub_D5F1F0((__int64)&v149, v37);
      v145 = 257;
      v38 = sub_BCB2E0(v155);
      v39 = (unsigned __int8 *)sub_ACD640(v38, v140, 0);
      v40 = (__int64 (__fastcall *)(__int64, _BYTE *, unsigned __int8 *))*((_QWORD *)*v156 + 12);
      if ( v40 == sub_948070 )
      {
        if ( *(_BYTE *)v138 > 0x15u || *v39 > 0x15u )
        {
LABEL_64:
          v148 = 257;
          v53 = sub_BD2C40(72, 2u);
          v41 = (__int64)v53;
          if ( v53 )
            sub_B4DE80((__int64)v53, v138, (__int64)v39, (__int64)&v146, 0, 0);
          (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v157 + 2))(
            v157,
            v41,
            v144,
            v153,
            v154);
          v54 = v149;
          v55 = &v149[4 * (unsigned int)v150];
          if ( v149 != v55 )
          {
            do
            {
              v56 = *((_QWORD *)v54 + 1);
              v57 = *v54;
              v54 += 4;
              sub_B99FD0(v41, v57, v56);
            }
            while ( v55 != v54 );
          }
          goto LABEL_55;
        }
        v41 = sub_AD5840(v138, v39, 0);
      }
      else
      {
        v41 = v40((__int64)v156, (_BYTE *)v138, v39);
      }
      if ( !v41 )
        goto LABEL_64;
LABEL_55:
      v148 = 257;
      v42 = sub_94B2B0(&v149, v131, v130, v140, (__int64)&v146);
      HIBYTE(v43) = HIBYTE(v136);
      LOBYTE(v43) = v132;
      v148 = 257;
      v136 = v43;
      v44 = sub_BD2C40(80, unk_3F10A10);
      v46 = (__int64)v44;
      if ( v44 )
        sub_B4D3C0((__int64)v44, v41, v42, 0, v136, v45, 0, 0);
      (*((void (__fastcall **)(void **, __int64, char **, __int64, __int64))*v157 + 2))(v157, v46, &v146, v153, v154);
      v47 = v149;
      v48 = &v149[4 * (unsigned int)v150];
      if ( v149 != v48 )
      {
        do
        {
          v49 = *((_QWORD *)v47 + 1);
          v50 = *v47;
          v47 += 4;
          sub_B99FD0(v46, v50, v49);
        }
        while ( v48 != v47 );
      }
      v51 = sub_B46EC0(v129, 0);
      v148 = 259;
      v146 = "else";
      sub_BD6B50((unsigned __int8 *)v51, (const char **)&v146);
      v52 = v126;
      LOWORD(v52) = 1;
      v126 = v52;
      sub_A88F30((__int64)&v149, v51, *(_QWORD *)(v51 + 56), 1);
      if ( v133 == ++v140 )
        goto LABEL_60;
    }
    v32 = v64((__int64)v156, (_BYTE *)v137, v63);
LABEL_80:
    if ( v32 )
      goto LABEL_46;
    goto LABEL_81;
  }
LABEL_60:
  sub_B43D60((_QWORD *)a3);
  *a5 = 1;
LABEL_61:
  nullsub_61();
  v164 = &unk_49DA100;
  nullsub_63();
  if ( v149 != (unsigned int *)v151 )
    _libc_free((unsigned __int64)v149);
}
