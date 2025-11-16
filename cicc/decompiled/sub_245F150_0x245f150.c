// Function: sub_245F150
// Address: 0x245f150
//
__int64 __fastcall sub_245F150(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v4; // r13
  __int64 i; // r12
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // rbx
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // rax
  unsigned __int64 v17; // rdi
  char *v19; // rbx
  __int64 v20; // rax
  __int64 **v21; // r15
  char *v22; // r14
  size_t v23; // r12
  __int64 (__fastcall **v24)(); // rax
  __int64 v25; // r12
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rbx
  __int64 v29; // rbx
  __int64 v30; // r14
  __int64 v31; // rbx
  __int64 v32; // rax
  unsigned int *v33; // rax
  __int64 v34; // rbx
  unsigned __int64 v35; // r14
  __int64 **v36; // rax
  __int64 v37; // r8
  unsigned __int8 *v38; // r12
  __int64 (__fastcall *v39)(void **, __int64, unsigned __int8 *, unsigned __int8 *, __int64, __int64); // rax
  unsigned __int64 v40; // r10
  __int64 v41; // rax
  __int64 (__fastcall *v42)(__int64, __int64, _BYTE *, _BYTE **, __int64, int); // rax
  _BYTE **v43; // rax
  __int64 *v44; // r11
  _BYTE **v45; // rcx
  __int64 v46; // r12
  __int64 v47; // rax
  __int64 v48; // rax
  char v49; // al
  __int16 v50; // cx
  _QWORD *v51; // rax
  __int64 v52; // r14
  __int64 v53; // r12
  const char *v54; // r12
  __int64 v55; // rbx
  __int64 v56; // rdx
  unsigned int v57; // esi
  __int64 (__fastcall *v58)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v59; // r12
  __int64 v60; // rax
  unsigned __int64 v61; // rax
  _QWORD **v62; // rdx
  int v63; // ecx
  __int64 *v64; // rax
  __int64 v65; // rsi
  __int64 v66; // rbx
  const char *v67; // r14
  __int64 v68; // rdx
  unsigned int v69; // esi
  __int64 v70; // r14
  const char *v71; // r12
  __int64 v72; // rbx
  __int64 v73; // rdx
  unsigned int v74; // esi
  __int64 v75; // r10
  unsigned int v76; // r11d
  __int64 v77; // rbx
  const char *v78; // r14
  __int64 v79; // rdx
  unsigned int v80; // esi
  __int64 v81; // rdx
  int v82; // eax
  char v83; // cl
  int v84; // eax
  __int64 v85; // rax
  __int64 (__fastcall **v86)(); // rdi
  __int64 v87; // [rsp-10h] [rbp-2C0h]
  __int64 v88; // [rsp+18h] [rbp-298h]
  __int64 **v89; // [rsp+20h] [rbp-290h]
  __int64 v90; // [rsp+38h] [rbp-278h]
  unsigned __int8 *v91; // [rsp+48h] [rbp-268h]
  __int64 v92; // [rsp+48h] [rbp-268h]
  __int64 v93; // [rsp+48h] [rbp-268h]
  __int64 v94; // [rsp+50h] [rbp-260h]
  __int64 v95; // [rsp+50h] [rbp-260h]
  char **v96; // [rsp+60h] [rbp-250h]
  __int64 *v97; // [rsp+60h] [rbp-250h]
  unsigned __int64 v99; // [rsp+70h] [rbp-240h]
  __int64 v101; // [rsp+78h] [rbp-238h]
  unsigned __int8 *v102; // [rsp+78h] [rbp-238h]
  __int64 v103; // [rsp+78h] [rbp-238h]
  unsigned int v104; // [rsp+80h] [rbp-230h]
  __int16 v105; // [rsp+86h] [rbp-22Ah]
  __int64 v106; // [rsp+88h] [rbp-228h]
  __int64 v107; // [rsp+98h] [rbp-218h]
  __int64 *v108; // [rsp+98h] [rbp-218h]
  __int64 v109; // [rsp+A8h] [rbp-208h]
  _BYTE *v110; // [rsp+B0h] [rbp-200h] BYREF
  char v111[24]; // [rsp+B8h] [rbp-1F8h] BYREF
  __int16 v112; // [rsp+D0h] [rbp-1E0h]
  __int64 v113[4]; // [rsp+E0h] [rbp-1D0h] BYREF
  __int16 v114; // [rsp+100h] [rbp-1B0h]
  _DWORD v115[8]; // [rsp+110h] [rbp-1A0h] BYREF
  __int16 v116; // [rsp+130h] [rbp-180h]
  unsigned __int64 v117; // [rsp+140h] [rbp-170h] BYREF
  unsigned int v118; // [rsp+148h] [rbp-168h]
  unsigned __int64 v119; // [rsp+150h] [rbp-160h]
  unsigned int v120; // [rsp+158h] [rbp-158h]
  __int16 v121; // [rsp+160h] [rbp-150h]
  __int64 (__fastcall **v122)(); // [rsp+170h] [rbp-140h] BYREF
  __int64 v123; // [rsp+178h] [rbp-138h]
  _QWORD v124[2]; // [rsp+180h] [rbp-130h] BYREF
  char *v125; // [rsp+190h] [rbp-120h]
  char *v126; // [rsp+198h] [rbp-118h]
  char *v127; // [rsp+1A0h] [rbp-110h]
  __int64 *v128; // [rsp+1B0h] [rbp-100h] BYREF
  __int64 v129; // [rsp+1B8h] [rbp-F8h]
  _BYTE v130[48]; // [rsp+1C0h] [rbp-F0h] BYREF
  const char *v131; // [rsp+1F0h] [rbp-C0h] BYREF
  __int64 v132; // [rsp+1F8h] [rbp-B8h]
  _BYTE v133[32]; // [rsp+200h] [rbp-B0h] BYREF
  __int64 v134; // [rsp+220h] [rbp-90h]
  __int64 v135; // [rsp+228h] [rbp-88h]
  __int64 v136; // [rsp+230h] [rbp-80h]
  _QWORD *v137; // [rsp+238h] [rbp-78h]
  void **v138; // [rsp+240h] [rbp-70h]
  void **v139; // [rsp+248h] [rbp-68h]
  __int64 v140; // [rsp+250h] [rbp-60h]
  int v141; // [rsp+258h] [rbp-58h]
  __int16 v142; // [rsp+25Ch] [rbp-54h]
  char v143; // [rsp+25Eh] [rbp-52h]
  __int64 v144; // [rsp+260h] [rbp-50h]
  __int64 v145; // [rsp+268h] [rbp-48h]
  void *v146; // [rsp+270h] [rbp-40h] BYREF
  void *v147; // [rsp+278h] [rbp-38h] BYREF

  v96 = *(char ***)(a3 + 40);
  if ( !sub_BA91D0((__int64)v96, "kcfi", 4u) )
  {
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v128 = (__int64 *)v130;
  v129 = 0x600000000LL;
  v3 = *(_QWORD *)(a3 + 80);
  v4 = a3 + 72;
  if ( a3 + 72 == v3 )
  {
    i = 0;
  }
  else
  {
    do
    {
      if ( !v3 )
LABEL_132:
        BUG();
      i = *(_QWORD *)(v3 + 32);
      if ( i != v3 + 24 )
        break;
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( v4 != v3 );
  }
LABEL_6:
  while ( v4 != v3 )
  {
    if ( !i )
      BUG();
    if ( *(_BYTE *)(i - 24) == 85 )
    {
      v107 = i - 24;
      if ( *(char *)(i - 17) < 0 )
      {
        v6 = sub_BD2BC0(i - 24);
        v10 = v6 + v9;
        if ( *(char *)(i - 17) < 0 )
          v10 -= sub_BD2BC0(v107);
        v11 = v10 >> 4;
        if ( (_DWORD)v11 )
        {
          v106 = v3;
          v12 = 0;
          while ( 1 )
          {
            v13 = 0;
            if ( *(char *)(i - 17) < 0 )
              v13 = sub_BD2BC0(v107);
            if ( *(_DWORD *)(*(_QWORD *)(v13 + v12) + 8LL) == 8 )
              break;
            v12 += 16;
            if ( 16LL * (unsigned int)v11 == v12 )
            {
              v3 = v106;
              goto LABEL_21;
            }
          }
          v14 = (unsigned int)v129;
          v3 = v106;
          v15 = (unsigned int)v129 + 1LL;
          if ( v15 > HIDWORD(v129) )
          {
            sub_C8D5F0((__int64)&v128, v130, v15, 8u, v7, v8);
            v14 = (unsigned int)v129;
          }
          v128[v14] = v107;
          LODWORD(v129) = v129 + 1;
        }
      }
    }
LABEL_21:
    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v3 + 32) )
    {
      v16 = v3 - 24;
      if ( !v3 )
        v16 = 0;
      if ( i != v16 + 48 )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( v4 == v3 )
        goto LABEL_6;
      if ( !v3 )
        goto LABEL_132;
    }
  }
  if ( !(_DWORD)v129 )
  {
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    goto LABEL_31;
  }
  v19 = *v96;
  if ( (unsigned __int8)sub_B2D620(a3, "patchable-function-prefix", 0x19u) )
  {
    v133[17] = 1;
    v131 = "-fpatchable-function-entry=N,M, where M>0 is not compatible with -fsanitize=kcfi on this target";
    v133[16] = 3;
    v123 = 6;
    v122 = off_4A16A08;
    v124[0] = &v131;
    sub_B6EB20((__int64)v19, (__int64)&v122);
  }
  v20 = sub_BCB2D0(v19);
  v131 = v19;
  v21 = (__int64 **)v20;
  v88 = sub_B8C340(&v131);
  v122 = (__int64 (__fastcall **)())v124;
  v22 = v96[29];
  v23 = (size_t)v96[30];
  if ( &v22[v23] && !v22 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v131 = v96[30];
  if ( v23 > 0xF )
  {
    v122 = (__int64 (__fastcall **)())sub_22409D0((__int64)&v122, (unsigned __int64 *)&v131, 0);
    v86 = v122;
    v124[0] = v131;
  }
  else
  {
    if ( v23 == 1 )
    {
      LOBYTE(v124[0]) = *v22;
      v24 = (__int64 (__fastcall **)())v124;
      goto LABEL_43;
    }
    if ( !v23 )
    {
      v24 = (__int64 (__fastcall **)())v124;
      goto LABEL_43;
    }
    v86 = (__int64 (__fastcall **)())v124;
  }
  memcpy(v86, v22, v23);
  v23 = (size_t)v131;
  v24 = v122;
LABEL_43:
  v123 = v23;
  *((_BYTE *)v24 + v23) = 0;
  v125 = v96[33];
  v126 = v96[34];
  v127 = v96[35];
  v108 = v128;
  v97 = &v128[(unsigned int)v129];
  if ( v97 != v128 )
  {
    while ( 1 )
    {
      v25 = *v108;
      if ( *(char *)(*v108 + 7) < 0 )
      {
        v26 = sub_BD2BC0(*v108);
        v28 = v26 + v27;
        if ( *(char *)(v25 + 7) < 0 )
          v28 -= sub_BD2BC0(v25);
        v29 = v28 >> 4;
        if ( (_DWORD)v29 )
        {
          v30 = 0;
          v31 = 16LL * (unsigned int)v29;
          while ( 1 )
          {
            v32 = 0;
            if ( *(char *)(v25 + 7) < 0 )
              v32 = sub_BD2BC0(v25);
            v33 = (unsigned int *)(v30 + v32);
            if ( *(_DWORD *)(*(_QWORD *)v33 + 8LL) == 8 )
              break;
            v30 += 16;
            if ( v30 == v31 )
              goto LABEL_54;
          }
          v99 = v25 + 32 * (v33[2] - (unsigned __int64)(*(_DWORD *)(v25 + 4) & 0x7FFFFFF));
        }
      }
LABEL_54:
      if ( *(_DWORD *)(*(_QWORD *)v99 + 32LL) <= 0x40u )
        v101 = *(_QWORD *)(*(_QWORD *)v99 + 24LL);
      else
        v101 = **(_QWORD **)(*(_QWORD *)v99 + 24LL);
      v34 = sub_B57640(v25, (__int64 *)8, v25 + 24, 0);
      sub_B47C00(v34, v25, 0, 0);
      sub_BD84D0(v25, v34);
      sub_B43D60((_QWORD *)v25);
      if ( !sub_B491E0(v34) )
        goto LABEL_85;
      v137 = (_QWORD *)sub_BD5C60(v34);
      v138 = &v146;
      v139 = &v147;
      v131 = v133;
      v132 = 0x200000000LL;
      v146 = &unk_49DA100;
      v140 = 0;
      v141 = 0;
      v147 = &unk_49DA0B0;
      v142 = 512;
      v143 = 7;
      v144 = 0;
      v145 = 0;
      v134 = 0;
      v135 = 0;
      LOWORD(v136) = 0;
      sub_D5F1F0((__int64)&v131, v34);
      v35 = *(_QWORD *)(v34 - 32);
      if ( (unsigned int)((_DWORD)v125 - 36) <= 1 || (unsigned int)((_DWORD)v125 - 1) <= 1 )
        break;
LABEL_66:
      v116 = 257;
      v41 = sub_BCB2D0(v137);
      v110 = (_BYTE *)sub_ACD640(v41, 0xFFFFFFFFLL, 0);
      v42 = (__int64 (__fastcall *)(__int64, __int64, _BYTE *, _BYTE **, __int64, int))*((_QWORD *)*v138 + 8);
      if ( v42 == sub_920540 )
      {
        if ( sub_BCEA30((__int64)v21) )
          goto LABEL_105;
        if ( *(_BYTE *)v35 > 0x15u )
          goto LABEL_105;
        v43 = sub_245EE00(&v110, (__int64)v111);
        if ( v45 != v43 )
          goto LABEL_105;
        LOBYTE(v121) = 0;
        v46 = sub_AD9FD0((__int64)v21, (unsigned __int8 *)v35, v44, 1, 3u, (__int64)&v117, 0);
        if ( (_BYTE)v121 )
        {
          LOBYTE(v121) = 0;
          if ( v120 > 0x40 && v119 )
            j_j___libc_free_0_0(v119);
          if ( v118 > 0x40 && v117 )
            j_j___libc_free_0_0(v117);
        }
      }
      else
      {
        v46 = v42((__int64)v138, (__int64)v21, (_BYTE *)v35, &v110, 1, 3);
      }
      if ( v46 )
        goto LABEL_72;
LABEL_105:
      v121 = 257;
      v46 = (__int64)sub_BD2C40(88, 2u);
      if ( !v46 )
        goto LABEL_108;
      v75 = *(_QWORD *)(v35 + 8);
      v104 = v104 & 0xE0000000 | 2;
      v76 = v104;
      if ( (unsigned int)*(unsigned __int8 *)(v75 + 8) - 17 > 1 )
      {
        v81 = *((_QWORD *)v110 + 1);
        v82 = *(unsigned __int8 *)(v81 + 8);
        if ( v82 == 17 )
        {
          v83 = 0;
LABEL_114:
          v84 = *(_DWORD *)(v81 + 32);
          BYTE4(v113[0]) = v83;
          LODWORD(v113[0]) = v84;
          v85 = sub_BCE1B0((__int64 *)v75, v113[0]);
          v76 = v104;
          v75 = v85;
          goto LABEL_107;
        }
        v83 = 1;
        if ( v82 == 18 )
          goto LABEL_114;
      }
LABEL_107:
      sub_B44260(v46, v75, 34, v76, 0, 0);
      *(_QWORD *)(v46 + 72) = v21;
      *(_QWORD *)(v46 + 80) = sub_B4DC50((__int64)v21, (__int64)&v110, 1);
      sub_B4D9A0(v46, v35, (__int64 *)&v110, 1, (__int64)&v117);
LABEL_108:
      sub_B4DDE0(v46, 3);
      (*((void (__fastcall **)(void **, __int64, _DWORD *, __int64, __int64))*v139 + 2))(v139, v46, v115, v135, v136);
      if ( v131 != &v131[16 * (unsigned int)v132] )
      {
        v95 = v34;
        v77 = (__int64)v131;
        v78 = &v131[16 * (unsigned int)v132];
        do
        {
          v79 = *(_QWORD *)(v77 + 8);
          v80 = *(_DWORD *)v77;
          v77 += 16;
          sub_B99FD0(v46, v80, v79);
        }
        while ( v78 != (const char *)v77 );
        v34 = v95;
      }
LABEL_72:
      v116 = 257;
      v47 = sub_ACD640((__int64)v21, (unsigned int)v101, 0);
      v114 = 257;
      v102 = (unsigned __int8 *)v47;
      v48 = sub_AA4E30(v134);
      v49 = sub_AE5020(v48, (__int64)v21);
      HIBYTE(v50) = HIBYTE(v105);
      v121 = 257;
      LOBYTE(v50) = v49;
      v105 = v50;
      v51 = sub_BD2C40(80, unk_3F10A14);
      v52 = (__int64)v51;
      if ( v51 )
        sub_B4D190((__int64)v51, (__int64)v21, v46, (__int64)&v117, 0, v105, 0, 0);
      (*((void (__fastcall **)(void **, __int64, __int64 *, __int64, __int64))*v139 + 2))(v139, v52, v113, v135, v136);
      v53 = 16LL * (unsigned int)v132;
      if ( v131 != &v131[v53] )
      {
        v94 = v34;
        v54 = &v131[v53];
        v55 = (__int64)v131;
        do
        {
          v56 = *(_QWORD *)(v55 + 8);
          v57 = *(_DWORD *)v55;
          v55 += 16;
          sub_B99FD0(v52, v57, v56);
        }
        while ( v54 != (const char *)v55 );
        v34 = v94;
      }
      v58 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, unsigned __int8 *))*((_QWORD *)*v138 + 7);
      if ( v58 == sub_928890 )
      {
        if ( *(_BYTE *)v52 > 0x15u || *v102 > 0x15u )
        {
LABEL_93:
          v121 = 257;
          v59 = (__int64)sub_BD2C40(72, unk_3F10FD0);
          if ( v59 )
          {
            v62 = *(_QWORD ***)(v52 + 8);
            v63 = *((unsigned __int8 *)v62 + 8);
            if ( (unsigned int)(v63 - 17) > 1 )
            {
              v65 = sub_BCB2A0(*v62);
            }
            else
            {
              BYTE4(v109) = (_BYTE)v63 == 18;
              LODWORD(v109) = *((_DWORD *)v62 + 8);
              v64 = (__int64 *)sub_BCB2A0(*v62);
              v65 = sub_BCE1B0(v64, v109);
            }
            sub_B523C0(v59, v65, 53, 33, v52, (__int64)v102, (__int64)&v117, 0, 0, 0);
          }
          (*((void (__fastcall **)(void **, __int64, _DWORD *, __int64, __int64))*v139 + 2))(
            v139,
            v59,
            v115,
            v135,
            v136);
          if ( v131 != &v131[16 * (unsigned int)v132] )
          {
            v103 = v34;
            v66 = (__int64)v131;
            v67 = &v131[16 * (unsigned int)v132];
            do
            {
              v68 = *(_QWORD *)(v66 + 8);
              v69 = *(_DWORD *)v66;
              v66 += 16;
              sub_B99FD0(v59, v69, v68);
            }
            while ( v67 != (const char *)v66 );
            v34 = v103;
          }
          goto LABEL_83;
        }
        v59 = sub_AAB310(0x21u, (unsigned __int8 *)v52, v102);
      }
      else
      {
        v59 = v58((__int64)v138, 33u, (_BYTE *)v52, v102);
      }
      if ( !v59 )
        goto LABEL_93;
LABEL_83:
      v60 = v90;
      LOWORD(v60) = 0;
      v90 = v60;
      v61 = sub_F38250(v59, (__int64 *)(v34 + 24), v60, 0, v88, 0, 0, 0);
      sub_D5F1F0((__int64)&v131, v61);
      v115[1] = 0;
      v121 = 257;
      sub_B33D10((__int64)&v131, 0x48u, 0, 0, 0, 0, v115[0], (__int64)&v117);
      nullsub_61();
      v146 = &unk_49DA100;
      nullsub_63();
      if ( v131 != v133 )
        _libc_free((unsigned __int64)v131);
LABEL_85:
      if ( v97 == ++v108 )
        goto LABEL_86;
    }
    v116 = 257;
    v36 = *(__int64 ***)(v35 + 8);
    v114 = 257;
    v89 = v36;
    v91 = (unsigned __int8 *)sub_ACD640((__int64)v21, -2, 0);
    v112 = 257;
    v38 = (unsigned __int8 *)sub_245EEC0((__int64 *)&v131, 0x2Fu, v35, v21, (__int64)&v110, 0, v117, 0);
    v39 = (__int64 (__fastcall *)(void **, __int64, unsigned __int8 *, unsigned __int8 *, __int64, __int64))*((_QWORD *)*v138 + 2);
    if ( (char *)v39 == (char *)sub_9202E0 )
    {
      if ( *v38 > 0x15u || *v91 > 0x15u )
        goto LABEL_101;
      if ( (unsigned __int8)sub_AC47B0(28) )
        v40 = sub_AD5570(28, (__int64)v38, v91, 0, 0);
      else
        v40 = sub_AABE40(0x1Cu, v38, v91);
    }
    else
    {
      v40 = v39(v138, 28, v38, v91, v37, v87);
    }
    if ( v40 )
    {
LABEL_65:
      v35 = sub_245EEC0((__int64 *)&v131, 0x30u, v40, v89, (__int64)v115, 0, v117, 0);
      goto LABEL_66;
    }
LABEL_101:
    v121 = 257;
    v92 = sub_B504D0(28, (__int64)v38, (__int64)v91, (__int64)&v117, 0, 0);
    (*((void (__fastcall **)(void **, __int64, __int64 *, __int64, __int64))*v139 + 2))(v139, v92, v113, v135, v136);
    v70 = (__int64)v131;
    v40 = v92;
    v71 = &v131[16 * (unsigned int)v132];
    if ( v131 != v71 )
    {
      v93 = v34;
      v72 = v40;
      do
      {
        v73 = *(_QWORD *)(v70 + 8);
        v74 = *(_DWORD *)v70;
        v70 += 16;
        sub_B99FD0(v72, v74, v73);
      }
      while ( v71 != (const char *)v70 );
      v40 = v72;
      v34 = v93;
    }
    goto LABEL_65;
  }
LABEL_86:
  memset((void *)a1, 0, 0x60u);
  *(_DWORD *)(a1 + 16) = 2;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
  if ( v122 != v124 )
  {
    j_j___libc_free_0((unsigned __int64)v122);
    v17 = (unsigned __int64)v128;
    if ( v128 == (__int64 *)v130 )
      return a1;
    goto LABEL_32;
  }
LABEL_31:
  v17 = (unsigned __int64)v128;
  if ( v128 != (__int64 *)v130 )
LABEL_32:
    _libc_free(v17);
  return a1;
}
