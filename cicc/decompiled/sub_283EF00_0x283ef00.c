// Function: sub_283EF00
// Address: 0x283ef00
//
__int64 __fastcall sub_283EF00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  void *v5; // r13
  __int64 v6; // rax
  _QWORD *v7; // rbx
  __int64 v8; // r12
  _QWORD *v9; // r13
  _QWORD *v10; // rax
  _QWORD *v11; // rdi
  _QWORD *v12; // rdi
  _QWORD *v13; // rbx
  __int64 v14; // r12
  _QWORD *i; // r13
  _QWORD *v16; // rax
  _QWORD *v17; // rdi
  _QWORD *v18; // rdi
  __int64 v19; // r12
  __int64 v20; // rbx
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 *v25; // r9
  __int64 v26; // rax
  unsigned __int64 *v27; // rax
  char v28; // al
  unsigned int v29; // ebx
  char v30; // al
  __int64 *v31; // r13
  __int64 *v32; // r14
  int v33; // esi
  __int64 v34; // r8
  unsigned int v35; // ecx
  __int64 **v36; // rdx
  __int64 *v37; // rsi
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  __int64 v40; // r12
  char v41; // r10
  unsigned int v42; // esi
  __int64 **v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 v54; // r9
  __int64 v55; // rdx
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  unsigned int v64; // eax
  __int64 *v65; // rdi
  unsigned int v66; // edx
  _BYTE *v67; // rcx
  char v68; // si
  int v69; // eax
  __int64 *v70; // r12
  __int64 *v71; // r13
  __int64 *v72; // r8
  int v73; // edi
  unsigned int v74; // esi
  __int64 *v75; // rdx
  __int64 *v76; // r11
  __int64 v77; // rax
  __int64 *v78; // rdx
  __int64 v79; // rcx
  __int64 v80; // rsi
  __int64 v81; // rsi
  __int64 (__fastcall *v82)(__int64, __int64); // rax
  void **v83; // rdx
  __int64 v84; // rcx
  __int64 v85; // r8
  __int64 v86; // rbx
  _QWORD *v87; // rax
  _QWORD *v88; // rbx
  __int64 v89; // r12
  _QWORD *v90; // r14
  __int64 v91; // rdx
  char *v92; // rsi
  _QWORD *v93; // rdi
  void (__fastcall *v94)(_QWORD *, char *, __int64, char *); // r9
  char *(*v95)(); // rax
  void **v96; // rax
  void **v97; // rdx
  __int64 **v98; // rax
  int j; // edx
  int v100; // r9d
  void **v101; // rax
  int v102; // r11d
  int v103; // ecx
  __int64 v104; // rdx
  int v105; // esi
  __int64 *v106; // rax
  void **v107; // rsi
  int v108; // ecx
  __int64 v109; // rdx
  int v110; // esi
  __int64 *v112; // [rsp+18h] [rbp-1D8h]
  void *v114; // [rsp+28h] [rbp-1C8h]
  void *v115; // [rsp+30h] [rbp-1C0h]
  __int64 v117; // [rsp+40h] [rbp-1B0h]
  __int64 v118; // [rsp+40h] [rbp-1B0h]
  __int64 v120; // [rsp+58h] [rbp-198h] BYREF
  _QWORD v121[3]; // [rsp+60h] [rbp-190h] BYREF
  char v122; // [rsp+78h] [rbp-178h]
  char v123; // [rsp+79h] [rbp-177h]
  char v124; // [rsp+7Ah] [rbp-176h]
  __int64 v125; // [rsp+80h] [rbp-170h]
  _QWORD v126[9]; // [rsp+90h] [rbp-160h] BYREF
  __int64 v127; // [rsp+D8h] [rbp-118h]
  char v128[8]; // [rsp+E0h] [rbp-110h] BYREF
  void **v129; // [rsp+E8h] [rbp-108h]
  char v130; // [rsp+F0h] [rbp-100h]
  int v131; // [rsp+F4h] [rbp-FCh]
  char v132; // [rsp+FCh] [rbp-F4h]
  char v133[8]; // [rsp+110h] [rbp-E0h] BYREF
  void **v134; // [rsp+118h] [rbp-D8h]
  int v135; // [rsp+124h] [rbp-CCh]
  char v136; // [rsp+12Ch] [rbp-C4h]
  _QWORD *v137; // [rsp+140h] [rbp-B0h] BYREF
  unsigned __int64 v138; // [rsp+148h] [rbp-A8h]
  __int64 *v139; // [rsp+150h] [rbp-A0h] BYREF
  unsigned int v140; // [rsp+158h] [rbp-98h]
  char v141; // [rsp+15Ch] [rbp-94h]
  char v142[16]; // [rsp+160h] [rbp-90h] BYREF
  char v143[8]; // [rsp+170h] [rbp-80h] BYREF
  unsigned __int64 v144; // [rsp+178h] [rbp-78h]
  char v145; // [rsp+18Ch] [rbp-64h]
  _BYTE *v146; // [rsp+190h] [rbp-60h] BYREF
  __int64 v147; // [rsp+198h] [rbp-58h]
  _BYTE v148[80]; // [rsp+1A0h] [rbp-50h] BYREF

  v4 = a1;
  v5 = (void *)(a1 + 32);
  v115 = (void *)(a1 + 80);
  v6 = *(_QWORD *)(sub_BC1CD0(a4, &qword_4F8A320, a3) + 8);
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  v120 = v6;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  if ( v6 )
  {
    v7 = *(_QWORD **)(v6 + 288);
    v8 = 4LL * *(unsigned int *)(v6 + 296);
    if ( v7 != &v7[v8] )
    {
      v114 = (void *)(a1 + 32);
      v9 = &v7[v8];
      do
      {
        v137 = 0;
        v10 = (_QWORD *)sub_22077B0(0x10u);
        if ( v10 )
        {
          v10[1] = a3;
          *v10 = &unk_49DB0A8;
        }
        v11 = v137;
        v137 = v10;
        if ( v11 )
          (*(void (__fastcall **)(_QWORD *))(*v11 + 8LL))(v11);
        v12 = v7;
        if ( (v7[3] & 2) == 0 )
          v12 = (_QWORD *)*v7;
        (*(void (__fastcall **)(_QWORD *, char *, __int64, _QWORD **))(v7[3] & 0xFFFFFFFFFFFFFFF8LL))(
          v12,
          "PassManager<llvm::Function>]",
          27,
          &v137);
        if ( v137 )
          (*(void (__fastcall **)(_QWORD *))(*v137 + 8LL))(v137);
        v7 += 4;
      }
      while ( v9 != v7 );
      v5 = v114;
      v4 = a1;
    }
  }
  sub_BC2570((__int64)&v137, (_QWORD *)(a2 + 8), a3, a4);
  sub_C8CF80(v4, v5, 2, (__int64)v142, (__int64)&v137);
  sub_C8CF80(v4 + 48, v115, 2, (__int64)&v146, (__int64)v143);
  if ( !v145 )
    _libc_free(v144);
  if ( !v141 )
    _libc_free(v138);
  if ( v120 )
  {
    v13 = *(_QWORD **)(v120 + 432);
    v14 = 4LL * *(unsigned int *)(v120 + 440);
    for ( i = &v13[v14]; i != v13; v13 += 4 )
    {
      v137 = 0;
      v16 = (_QWORD *)sub_22077B0(0x10u);
      if ( v16 )
      {
        v16[1] = a3;
        *v16 = &unk_49DB0A8;
      }
      v17 = v137;
      v137 = v16;
      if ( v17 )
        (*(void (__fastcall **)(_QWORD *))(*v17 + 8LL))(v17);
      v18 = v13;
      if ( (v13[3] & 2) == 0 )
        v18 = (_QWORD *)*v13;
      (*(void (__fastcall **)(_QWORD *, char *, __int64, _QWORD **, __int64))(v13[3] & 0xFFFFFFFFFFFFFFF8LL))(
        v18,
        "PassManager<llvm::Function>]",
        27,
        &v137,
        v4);
      if ( v137 )
        (*(void (__fastcall **)(_QWORD *))(*v137 + 8LL))(v137);
    }
  }
  v19 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  if ( *(_QWORD *)(v19 + 40) == *(_QWORD *)(v19 + 48) )
    return v4;
  v20 = 0;
  if ( *(_BYTE *)(a2 + 48) )
  {
    v20 = *(_QWORD *)(sub_BC1CD0(a4, &unk_4F8F810, a3) + 8);
    if ( !*(_BYTE *)(a2 + 49) )
      goto LABEL_32;
  }
  else if ( !*(_BYTE *)(a2 + 49) )
  {
LABEL_32:
    v117 = 0;
    goto LABEL_33;
  }
  sub_B2EE70((__int64)&v137, a3, 0);
  if ( !(_BYTE)v139 )
    goto LABEL_32;
  v117 = sub_BC1CD0(a4, &unk_4F8D9A8, a3) + 8;
LABEL_33:
  if ( *(_BYTE *)(a2 + 50) && (sub_B2EE70((__int64)&v137, a3, 0), (_BYTE)v139) )
    v21 = sub_BC1CD0(a4, &unk_4F8E5A8, a3) + 8;
  else
    v21 = 0;
  v126[0] = sub_BC1CD0(a4, &unk_4F86540, a3) + 8;
  v126[1] = sub_BC1CD0(a4, &unk_4F86630, a3) + 8;
  v126[2] = sub_BC1CD0(a4, &unk_4F81450, a3) + 8;
  v126[3] = sub_BC1CD0(a4, &unk_4F875F0, a3) + 8;
  v126[4] = sub_BC1CD0(a4, &unk_4F881D0, a3) + 8;
  v126[5] = sub_BC1CD0(a4, &unk_4F6D3F8, a3) + 8;
  v127 = v20;
  v126[8] = v21;
  v126[6] = sub_BC1CD0(a4, &unk_4F89C30, a3) + 8;
  v126[7] = v117;
  v22 = sub_BC1CD0(a4, &unk_4FDBCE0, a3);
  if ( *(_BYTE *)(a2 + 48) )
    *(_BYTE *)(v22 + 24) = 1;
  v26 = *(_QWORD *)(v22 + 8);
  v137 = 0;
  v138 = 1;
  v118 = v26;
  v27 = (unsigned __int64 *)&v139;
  do
  {
    *v27 = -4096;
    v27 += 2;
  }
  while ( v27 != (unsigned __int64 *)&v146 );
  v121[0] = &v137;
  v146 = v148;
  v147 = 0x400000000LL;
  v121[1] = v118;
  v28 = *(_BYTE *)(a2 + 51);
  v124 = 0;
  v123 = v28;
  if ( !v28 )
  {
    sub_F774D0(v19 + 8, (__int64)&v137, (__int64)&v146, v23, v24, (__int64)v25);
    v67 = v146;
    v29 = v147;
    v30 = v138;
    goto LABEL_69;
  }
  v29 = 0;
  v30 = v138;
  if ( *(_QWORD *)(v19 + 48) == *(_QWORD *)(v19 + 40) )
  {
    v67 = v148;
    goto LABEL_69;
  }
  v31 = *(__int64 **)(v19 + 40);
  v32 = *(__int64 **)(v19 + 48);
  do
  {
    v40 = *v31;
    v41 = v30 & 1;
    if ( (v30 & 1) != 0 )
    {
      v33 = 3;
      v34 = (__int64)&v139;
    }
    else
    {
      v42 = v140;
      v34 = (__int64)v139;
      if ( !v140 )
      {
        v64 = v138;
        v137 = (_QWORD *)((char *)v137 + 1);
        v65 = 0;
        v66 = ((unsigned int)v138 >> 1) + 1;
        goto LABEL_62;
      }
      v33 = v140 - 1;
    }
    v35 = v33 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
    v36 = (__int64 **)(v34 + 16LL * v35);
    v25 = *v36;
    if ( (__int64 *)v40 != *v36 )
    {
      v102 = 1;
      v65 = 0;
      while ( v25 != (__int64 *)-4096LL )
      {
        if ( !v65 && v25 == (__int64 *)-8192LL )
          v65 = (__int64 *)v36;
        v35 = v33 & (v102 + v35);
        v36 = (__int64 **)(v34 + 16LL * v35);
        v25 = *v36;
        if ( (__int64 *)v40 == *v36 )
          goto LABEL_44;
        ++v102;
      }
      v64 = v138;
      if ( !v65 )
        v65 = (__int64 *)v36;
      v137 = (_QWORD *)((char *)v137 + 1);
      v66 = ((unsigned int)v138 >> 1) + 1;
      if ( v41 )
      {
        v34 = 12;
        v42 = 4;
        if ( 4 * v66 >= 0xC )
        {
LABEL_139:
          sub_F76580((__int64)&v137, 2 * v42);
          if ( (v138 & 1) != 0 )
          {
            v103 = 3;
            v25 = (__int64 *)&v139;
          }
          else
          {
            v25 = v139;
            if ( !v140 )
              goto LABEL_197;
            v103 = v140 - 1;
          }
          v64 = v138;
          LODWORD(v104) = v103 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
          v65 = &v25[2 * (unsigned int)v104];
          v34 = *v65;
          if ( v40 != *v65 )
          {
            v105 = 1;
            v106 = 0;
            while ( v34 != -4096 )
            {
              if ( v34 == -8192 && !v106 )
                v106 = v65;
              v104 = v103 & (unsigned int)(v104 + v105);
              v65 = &v25[2 * v104];
              v34 = *v65;
              if ( v40 == *v65 )
                goto LABEL_178;
              ++v105;
            }
LABEL_176:
            if ( v106 )
              v65 = v106;
LABEL_178:
            v64 = v138;
            goto LABEL_64;
          }
          goto LABEL_64;
        }
LABEL_63:
        if ( v42 - HIDWORD(v138) - v66 <= v42 >> 3 )
        {
          sub_F76580((__int64)&v137, v42);
          if ( (v138 & 1) != 0 )
          {
            v108 = 3;
            v25 = (__int64 *)&v139;
          }
          else
          {
            v25 = v139;
            if ( !v140 )
            {
LABEL_197:
              LODWORD(v138) = (2 * ((unsigned int)v138 >> 1) + 2) | v138 & 1;
              BUG();
            }
            v108 = v140 - 1;
          }
          v64 = v138;
          LODWORD(v109) = v108 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
          v65 = &v25[2 * (unsigned int)v109];
          v34 = *v65;
          if ( v40 != *v65 )
          {
            v110 = 1;
            v106 = 0;
            while ( v34 != -4096 )
            {
              if ( !v106 && v34 == -8192 )
                v106 = v65;
              v109 = v108 & (unsigned int)(v109 + v110);
              v65 = &v25[2 * v109];
              v34 = *v65;
              if ( v40 == *v65 )
                goto LABEL_178;
              ++v110;
            }
            goto LABEL_176;
          }
        }
LABEL_64:
        LODWORD(v138) = (2 * (v64 >> 1) + 2) | v64 & 1;
        if ( *v65 != -4096 )
          --HIDWORD(v138);
        *v65 = v40;
        v65[1] = v29;
        v38 = (unsigned int)v147;
        v39 = (unsigned int)v147 + 1LL;
        if ( v39 <= HIDWORD(v147) )
          goto LABEL_46;
LABEL_67:
        sub_C8D5F0((__int64)&v146, v148, v39, 8u, v34, (__int64)v25);
        v38 = (unsigned int)v147;
        goto LABEL_46;
      }
      v42 = v140;
LABEL_62:
      v34 = 3 * v42;
      if ( 4 * v66 >= (unsigned int)v34 )
        goto LABEL_139;
      goto LABEL_63;
    }
LABEL_44:
    v37 = v36[1];
    if ( v37 == (__int64 *)(v29 - 1LL) )
      goto LABEL_47;
    *(_QWORD *)&v146[8 * (_QWORD)v37] = 0;
    v36[1] = (__int64 *)(unsigned int)v147;
    v38 = (unsigned int)v147;
    v39 = (unsigned int)v147 + 1LL;
    if ( v39 > HIDWORD(v147) )
      goto LABEL_67;
LABEL_46:
    *(_QWORD *)&v146[8 * v38] = v40;
    v29 = v147 + 1;
    v30 = v138;
    LODWORD(v147) = v147 + 1;
LABEL_47:
    ++v31;
  }
  while ( v32 != v31 );
  v67 = v146;
LABEL_69:
  v68 = v30;
  v69 = v29;
  v70 = (__int64 *)a2;
  v71 = *(__int64 **)&v67[8 * v29 - 8];
  if ( (v68 & 1) == 0 )
    goto LABEL_98;
LABEL_70:
  v72 = (__int64 *)&v139;
  v73 = 3;
LABEL_71:
  v74 = v73 & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
  v75 = &v72[2 * v74];
  v76 = (__int64 *)*v75;
  if ( v71 != (__int64 *)*v75 )
  {
    for ( j = 1; ; j = v100 )
    {
      if ( v76 == (__int64 *)-4096LL )
        goto LABEL_73;
      v100 = j + 1;
      v74 = v73 & (j + v74);
      v75 = &v72[2 * v74];
      v76 = (__int64 *)*v75;
      if ( v71 == (__int64 *)*v75 )
        break;
    }
  }
  *v75 = -8192;
  ++HIDWORD(v138);
  v67 = v146;
  LODWORD(v138) = (2 * ((unsigned int)v138 >> 1) - 2) | v138 & 1;
  v69 = v147;
  while ( 1 )
  {
LABEL_73:
    v77 = (unsigned int)(v69 - 1);
    v78 = (__int64 *)&v67[8 * v77 - 8];
    do
    {
      LODWORD(v147) = v77;
      if ( !(_DWORD)v77 )
        break;
      v79 = *v78;
      LODWORD(v77) = v77 - 1;
      --v78;
    }
    while ( !v79 );
    v121[2] = v71;
    v80 = *v70;
    v122 = 0;
    v125 = *v71;
    if ( (unsigned __int8)sub_283D940(&v120, v80, (__int64)v71) )
    {
      v81 = *v70;
      v82 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)*v70 + 16LL);
      if ( v82 == sub_2302D00 )
        sub_283EA00((__int64)v128, v81 + 8, v71, v118, (__int64)v126, (__int64)v121);
      else
        ((void (__fastcall *)(char *, __int64, __int64 *, __int64, _QWORD *, _QWORD *))v82)(
          v128,
          v81,
          v71,
          v118,
          v126,
          v121);
      if ( v122 )
      {
        if ( v120 )
        {
          v85 = *(_QWORD *)(v120 + 576);
          v86 = 32LL * *(unsigned int *)(v120 + 584);
          v87 = (_QWORD *)(v85 + v86);
          if ( v85 != v85 + v86 )
          {
            v112 = v70;
            v88 = *(_QWORD **)(v120 + 576);
            v89 = *v70;
            v90 = v87;
            do
            {
              v95 = *(char *(**)())(*(_QWORD *)v89 + 32LL);
              if ( v95 == sub_230FAA0 )
              {
                v91 = 149;
                v92 = "PassManager<llvm::Loop, llvm::AnalysisManager<llvm::Loop, llvm::LoopStandardAnalysisResults&>, llvm::LoopStandardAnalysisResults&, llvm::LPMUpdater&>]";
              }
              else
              {
                v92 = (char *)((__int64 (__fastcall *)(__int64))v95)(v89);
              }
              v93 = v88;
              v94 = *(void (__fastcall **)(_QWORD *, char *, __int64, char *))(v88[3] & 0xFFFFFFFFFFFFFFF8LL);
              if ( (v88[3] & 2) == 0 )
                v93 = (_QWORD *)*v88;
              v88 += 4;
              v94(v93, v92, v91, v128);
            }
            while ( v90 != v88 );
            v70 = v112;
          }
        }
      }
      else
      {
        sub_283D2B0(v120, *v70, (__int64)v71, (__int64)v128);
      }
      if ( v127 )
      {
        if ( v136 )
        {
          v96 = v134;
          v97 = &v134[v135];
          if ( v134 != v97 )
          {
            while ( *v96 != &unk_4F8F810 )
            {
              if ( v97 == ++v96 )
                goto LABEL_111;
            }
LABEL_107:
            sub_C64ED0("Loop pass manager using MemorySSA contains a pass that does not preserve MemorySSA", 0);
          }
        }
        else if ( sub_C8CA60((__int64)v133, (__int64)&unk_4F8F810) )
        {
          goto LABEL_107;
        }
LABEL_111:
        if ( v132 )
        {
          v83 = v129;
          v84 = (__int64)&v129[v131];
          if ( v129 == (void **)v84 )
            goto LABEL_107;
          v98 = (__int64 **)v129;
          while ( *v98 != &qword_4F82400 )
          {
            if ( (__int64 **)v84 == ++v98 )
              goto LABEL_131;
          }
        }
        else if ( !sub_C8CA60((__int64)v128, (__int64)&qword_4F82400) )
        {
          if ( v132 )
          {
            v83 = v129;
            v98 = (__int64 **)&v129[v131];
            if ( v98 == (__int64 **)v129 )
              goto LABEL_107;
LABEL_131:
            while ( *v83 != &unk_4F8F810 )
            {
              if ( v98 == (__int64 **)++v83 )
                goto LABEL_107;
            }
          }
          else if ( !sub_C8CA60((__int64)v128, (__int64)&unk_4F8F810) )
          {
            goto LABEL_107;
          }
        }
      }
      if ( !v122 )
        sub_22D08B0(v118, (__int64)v71, (__int64)v128);
      sub_BBADB0(v4, (__int64)v128, (__int64)v83, v84);
      if ( !v136 )
        _libc_free((unsigned __int64)v134);
      if ( !v132 )
        _libc_free((unsigned __int64)v129);
    }
    v69 = v147;
    if ( !(_DWORD)v147 )
      break;
    v67 = v146;
    v71 = *(__int64 **)&v146[8 * (unsigned int)v147 - 8];
    if ( (v138 & 1) != 0 )
      goto LABEL_70;
LABEL_98:
    v72 = v139;
    if ( v140 )
    {
      v73 = v140 - 1;
      goto LABEL_71;
    }
  }
  if ( *(_DWORD *)(v4 + 68) == *(_DWORD *)(v4 + 72) )
  {
    if ( *(_BYTE *)(v4 + 28) )
    {
      v101 = *(void ***)(v4 + 8);
      v107 = &v101[*(unsigned int *)(v4 + 20)];
      v44 = *(unsigned int *)(v4 + 20);
      v43 = (__int64 **)v101;
      if ( v101 == v107 )
        goto LABEL_160;
      while ( *v43 != &qword_4F82400 )
      {
        if ( v107 == (void **)++v43 )
        {
LABEL_126:
          while ( *v101 != &unk_4FDBCE8 )
          {
            if ( v43 == (__int64 **)++v101 )
              goto LABEL_160;
          }
          break;
        }
      }
    }
    else if ( !sub_C8CA60(v4, (__int64)&qword_4F82400) )
    {
      goto LABEL_122;
    }
  }
  else
  {
LABEL_122:
    if ( !*(_BYTE *)(v4 + 28) )
      goto LABEL_51;
    v101 = *(void ***)(v4 + 8);
    v44 = *(unsigned int *)(v4 + 20);
    v43 = (__int64 **)&v101[v44];
    if ( v43 != (__int64 **)v101 )
      goto LABEL_126;
LABEL_160:
    if ( (unsigned int)v44 < *(_DWORD *)(v4 + 16) )
    {
      v44 = (unsigned int)(v44 + 1);
      *(_DWORD *)(v4 + 20) = v44;
      *v43 = (__int64 *)&unk_4FDBCE8;
      ++*(_QWORD *)v4;
    }
    else
    {
LABEL_51:
      sub_C8CC70(v4, (__int64)&unk_4FDBCE8, (__int64)v43, v44, v45, v46);
    }
  }
  sub_283D3D0(v4, (__int64)&unk_4FDBCE0, (__int64)v43, v44, v45, v46);
  sub_283D3D0(v4, (__int64)&unk_4F81450, v47, v48, v49, v50);
  sub_283D3D0(v4, (__int64)&unk_4F875F0, v51, v52, v53, v54);
  sub_283D3D0(v4, (__int64)&unk_4F881D0, v55, v56, v57, v58);
  if ( *(_BYTE *)(a2 + 49) )
  {
    sub_B2EE70((__int64)v128, a3, 0);
    if ( v130 )
      sub_283D3D0(v4, (__int64)&unk_4F8D9A8, v59, v60, v61, v62);
  }
  if ( *(_BYTE *)(a2 + 50) )
  {
    sub_B2EE70((__int64)v128, a3, 0);
    if ( v130 )
      sub_283D3D0(v4, (__int64)&unk_4F8E5A8, v59, v60, v61, v62);
  }
  if ( *(_BYTE *)(a2 + 48) )
    sub_283D3D0(v4, (__int64)&unk_4F8F810, v59, v60, v61, v62);
  if ( v146 != v148 )
    _libc_free((unsigned __int64)v146);
  if ( (v138 & 1) == 0 )
    sub_C7D6A0((__int64)v139, 16LL * v140, 8);
  return v4;
}
