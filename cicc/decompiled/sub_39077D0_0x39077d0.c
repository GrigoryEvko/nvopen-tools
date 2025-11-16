// Function: sub_39077D0
// Address: 0x39077d0
//
__int64 __fastcall sub_39077D0(__int64 a1, char a2, __int64 a3)
{
  __int64 v4; // rdi
  __int64 v5; // rax
  void *v6; // r14
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  _DWORD *v10; // rdi
  int v11; // r13d
  unsigned int v12; // r15d
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rax
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // rax
  __int64 v27; // rsi
  unsigned int v28; // r13d
  __int64 v29; // r14
  char v30; // r15
  size_t v31; // r8
  char *v32; // r9
  const char *v33; // rax
  __int64 v34; // rdi
  __int64 result; // rax
  const char *v36; // rax
  __int64 v37; // rcx
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rdi
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rdi
  void *v44; // rdi
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  _BYTE *v48; // rax
  __int64 *v49; // rax
  _QWORD *v50; // rax
  __int64 v51; // rax
  __int64 v52; // r13
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rdx
  unsigned __int64 v57; // rax
  size_t v58; // rcx
  unsigned __int64 v59; // rcx
  void *v60; // rax
  __int64 v61; // rdi
  char v62; // al
  char v63; // al
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 v66; // r12
  const char *v67; // rax
  __int64 v68; // rdi
  __int64 v69; // r8
  __int64 v70; // r9
  const char *v71; // rax
  __int64 v72; // rdi
  char v73; // al
  __int64 v74; // r12
  __int64 v75; // rdi
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // rdi
  const char *v79; // rax
  __int64 v80; // rdi
  __int64 v81; // rax
  __int64 v82; // r12
  __int64 v83; // rax
  __int64 v84; // rax
  unsigned __int64 v85; // r14
  char *v86; // r9
  unsigned __int64 v87; // rcx
  bool v88; // cc
  __int64 v89; // r15
  char *v90; // r14
  char v91; // al
  char *v92; // rcx
  __int64 v93; // rdx
  int v94; // r14d
  __int64 v95; // r12
  __int64 v96; // r8
  __int64 v97; // r9
  int v98; // eax
  __int64 v99; // rdi
  __int64 v100; // rax
  int v101; // r12d
  __int64 v102; // rax
  __int64 v103; // rdx
  unsigned __int64 v104; // rax
  unsigned __int64 v105; // rcx
  unsigned __int64 v106; // rax
  __int64 v107; // rcx
  __int64 v108; // rdi
  __int64 v109; // rax
  __int64 v110; // rdi
  __int64 v111; // rdx
  __int64 v112; // rax
  __int64 v113; // r8
  __int64 v114; // r9
  __int64 v115; // rdi
  __int64 v116; // rdi
  __int64 v117; // rax
  __int64 v118; // rax
  __int64 v119; // r8
  __int64 v120; // r9
  unsigned __int64 v121; // rax
  __int64 v122; // rdi
  __int64 v123; // rdi
  __int64 v124; // rdi
  unsigned __int64 v125; // rdx
  __int64 v126; // rax
  __int64 v127; // rax
  __int64 v128; // rdi
  __int64 v129; // rdx
  _QWORD *v130; // rax
  __int64 v131; // [rsp-8h] [rbp-E8h]
  void *v132; // [rsp+8h] [rbp-D8h]
  char *v133; // [rsp+8h] [rbp-D8h]
  size_t v136; // [rsp+18h] [rbp-C8h]
  unsigned int v137; // [rsp+18h] [rbp-C8h]
  int v138; // [rsp+18h] [rbp-C8h]
  size_t v139; // [rsp+18h] [rbp-C8h]
  int v140; // [rsp+18h] [rbp-C8h]
  __int64 v141; // [rsp+28h] [rbp-B8h] BYREF
  __int64 v142; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v143; // [rsp+38h] [rbp-A8h] BYREF
  void *v144; // [rsp+40h] [rbp-A0h] BYREF
  size_t n; // [rsp+48h] [rbp-98h]
  __int64 v146; // [rsp+50h] [rbp-90h] BYREF
  __int64 v147; // [rsp+58h] [rbp-88h]
  _QWORD *v148; // [rsp+60h] [rbp-80h] BYREF
  __int64 v149; // [rsp+68h] [rbp-78h]
  void *s1; // [rsp+70h] [rbp-70h] BYREF
  __int64 v151; // [rsp+78h] [rbp-68h]
  __int16 v152; // [rsp+80h] [rbp-60h]
  __int64 v153[2]; // [rsp+90h] [rbp-50h] BYREF
  __int16 v154; // [rsp+A0h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 8);
  v144 = 0;
  n = 0;
  v5 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 40LL))(v4);
  v6 = (void *)sub_3909290(v5);
  v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
  v10 = *(_DWORD **)(a1 + 8);
  if ( **(_DWORD **)(v7 + 8) == 3 )
  {
    v55 = sub_3909460(v10);
    v56 = v55;
    if ( *(_DWORD *)v55 == 2 )
    {
      v60 = *(void **)(v55 + 8);
      v58 = *(_QWORD *)(v56 + 16);
    }
    else
    {
      v57 = *(_QWORD *)(v55 + 16);
      v58 = 0;
      if ( v57 )
      {
        v59 = v57 - 1;
        if ( v57 == 1 )
          v59 = 1;
        if ( v59 > v57 )
          v59 = v57;
        v57 = 1;
        v58 = v59 - 1;
      }
      v60 = (void *)(*(_QWORD *)(v56 + 8) + v57);
    }
    v61 = *(_QWORD *)(a1 + 8);
    v144 = v60;
    n = v58;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v61 + 136LL))(v61);
  }
  else
  {
    v11 = v10[8];
    if ( v11 )
      goto LABEL_49;
    do
    {
      v15 = (*(__int64 (__fastcall **)(_DWORD *))(*(_QWORD *)v10 + 40LL))(v10);
      v16 = sub_3909290(v15);
      if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 25
        || **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
      {
        break;
      }
      v17 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
      v18 = *(_QWORD *)(a1 + 8);
      if ( **(_DWORD **)(v17 + 8) == 3 )
      {
        v24 = sub_3909460(v18);
        v25 = *(_QWORD *)(v24 + 16);
        if ( *(_DWORD *)v24 != 2 && v25 )
        {
          v26 = v25 - 1;
          if ( v25 == 1 )
            v26 = 1;
          if ( v26 <= v25 )
            LODWORD(v25) = v26;
          LODWORD(v25) = v25 - 1;
        }
        v12 = v25 + 2;
        (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
      }
      else
      {
        v19 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v18 + 40LL))(v18);
        v20 = *(_QWORD *)(a1 + 8);
        if ( **(_DWORD **)(v19 + 8) == 2 )
        {
          v21 = sub_3909460(v20);
          v22 = *(_QWORD *)(v21 + 16);
          if ( *(_DWORD *)v21 != 2 && v22 )
          {
            v23 = v22 - 1;
            if ( v22 == 1 )
              v23 = 1;
            if ( v23 <= v22 )
              LODWORD(v22) = v23;
            LODWORD(v22) = v22 - 1;
          }
          v12 = v22;
          (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
        }
        else
        {
          v12 = *(_DWORD *)(sub_3909460(v20) + 16);
          (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
        }
      }
      v13 = *(_QWORD *)(a1 + 8);
      v144 = v6;
      n = v11 + v12;
      v11 += v12;
      v14 = sub_3909460(v13);
      if ( v12 + v16 != sub_39092A0(v14) )
        break;
      v10 = *(_DWORD **)(a1 + 8);
    }
    while ( !v10[8] );
    if ( !v11 )
    {
      v10 = *(_DWORD **)(a1 + 8);
LABEL_49:
      HIBYTE(v154) = 1;
      v36 = "expected identifier in directive";
LABEL_50:
      v153[0] = (__int64)v36;
      LOBYTE(v154) = 3;
      return sub_3909CF0(v10, v153, 0, 0, v8, v9);
    }
  }
  v146 = 0;
  v27 = n;
  v147 = 0;
  v141 = 0;
  v148 = 0;
  v149 = 0;
  v142 = 0;
  v143 = -1;
  if ( (unsigned __int8)sub_3905F30(v144, n, ".rodata.", 8u) )
    goto LABEL_29;
  if ( n == 8 )
  {
    if ( *(_QWORD *)v144 == 0x31617461646F722ELL )
    {
LABEL_29:
      v28 = 2;
      goto LABEL_30;
    }
  }
  else if ( n == 5
         && (*(_DWORD *)v144 == 1852401198 && *((_BYTE *)v144 + 4) == 105
          || *(_DWORD *)v144 == 1768843566 && *((_BYTE *)v144 + 4) == 116) )
  {
    goto LABEL_93;
  }
  v27 = n;
  if ( (unsigned __int8)sub_3905F30(v144, n, ".text.", 6u) )
  {
LABEL_93:
    v28 = 6;
    goto LABEL_30;
  }
  v27 = n;
  if ( (unsigned __int8)sub_3905F30(v144, n, ".data.", 6u)
    || n == 6 && *(_DWORD *)v144 == 1952539694 && *((_WORD *)v144 + 2) == 12641
    || (v27 = n, (unsigned __int8)sub_3905F30(v144, n, ".bss.", 5u))
    || (v27 = n, (unsigned __int8)sub_3905F30(v144, n, ".init_array.", 0xCu))
    || (v27 = n, (unsigned __int8)sub_3905F30(v144, n, ".fini_array.", 0xCu))
    || (v27 = n, (unsigned __int8)sub_3905F30(v144, n, ".preinit_array.", 0xFu)) )
  {
    v28 = 3;
  }
  else
  {
    v27 = n;
    if ( (unsigned __int8)sub_3905F30(v144, n, ".tdata.", 7u)
      || (v27 = n, v28 = 0, (unsigned __int8)sub_3905F30(v144, n, ".tbss.", 6u)) )
    {
      v28 = 1027;
    }
  }
LABEL_30:
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 25 )
  {
    v29 = 0;
    v30 = 0;
    goto LABEL_32;
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  if ( a2
    && **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 3 )
  {
    v27 = (__int64)&v142;
    v30 = sub_3909510(*(_QWORD *)(a1 + 8), &v142);
    if ( v30 )
      return 1;
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 25 )
    {
      v29 = 0;
      goto LABEL_32;
    }
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  }
  v39 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
  v40 = *(_QWORD *)(a1 + 8);
  if ( **(_DWORD **)(v39 + 8) != 3 )
  {
    if ( !*(_BYTE *)(*(_QWORD *)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v40 + 48LL))(v40) + 16) + 280LL)
      || **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 37 )
    {
      v43 = *(_QWORD *)(a1 + 8);
      v153[0] = (__int64)"expected string in directive";
      v154 = 259;
      return sub_3909CF0(v43, v153, 0, 0, v41, v42);
    }
    v101 = 0;
    while ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 37 )
    {
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
      if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 2 )
        goto LABEL_192;
      v102 = sub_3909460(*(_QWORD *)(a1 + 8));
      v103 = v102;
      if ( *(_DWORD *)v102 == 2 )
      {
        v107 = *(_QWORD *)(v102 + 8);
        v106 = *(_QWORD *)(v102 + 16);
      }
      else
      {
        v104 = *(_QWORD *)(v102 + 16);
        if ( !v104 )
          goto LABEL_192;
        v27 = 1;
        v105 = v104 - 1;
        if ( v104 == 1 )
          v105 = 1;
        if ( v105 <= v104 )
          v104 = v105;
        v106 = v104 - 1;
        v107 = *(_QWORD *)(v103 + 8) + 1LL;
      }
      if ( v106 == 5 )
      {
        if ( *(_DWORD *)v107 == 1869376609 && *(_BYTE *)(v107 + 4) == 99 )
        {
          v101 |= 2u;
        }
        else
        {
          if ( *(_DWORD *)v107 != 1953067639 || *(_BYTE *)(v107 + 4) != 101 )
            goto LABEL_192;
          v101 |= 1u;
        }
      }
      else if ( v106 == 9 )
      {
        if ( *(_QWORD *)v107 != 0x74736E6963657865LL || *(_BYTE *)(v107 + 8) != 114 )
          goto LABEL_192;
        v101 |= 4u;
      }
      else
      {
        if ( v106 != 3 || *(_WORD *)v107 != 27764 || *(_BYTE *)(v107 + 2) != 115 )
          goto LABEL_192;
        v101 |= 0x400u;
      }
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
      if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 25 )
        break;
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    }
    v28 |= v101;
    goto LABEL_261;
  }
  v84 = sub_3909460(v40);
  v85 = *(_QWORD *)(v84 + 16);
  if ( v85 )
  {
    v86 = *(char **)(v84 + 16);
    v87 = v85 - 1;
    if ( v85 == 1 )
      v87 = 1;
    v88 = v87 <= v85;
    v85 = 1;
    if ( v88 )
      v86 = (char *)v87;
    v89 = (__int64)(v86 - 1);
  }
  else
  {
    v86 = 0;
    v89 = 0;
  }
  v133 = v86;
  v90 = (char *)(*(_QWORD *)(v84 + 8) + v85);
  v139 = *(_QWORD *)(v84 + 8);
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  v27 = v89;
  v91 = sub_16D2B80((__int64)v90, v89, 0, (unsigned __int64 *)v153);
  v31 = v139;
  v32 = v133;
  v30 = v91;
  if ( !v91 )
  {
    v93 = v153[0];
    if ( v153[0] == LODWORD(v153[0]) )
    {
      if ( LODWORD(v153[0]) == -1 )
      {
LABEL_192:
        HIBYTE(v154) = 1;
        v33 = "unknown flag";
        goto LABEL_34;
      }
      goto LABEL_156;
    }
  }
  v92 = &v133[v139];
  v93 = 0;
  v30 = 0;
  if ( &v133[v139] != v90 )
  {
    do
    {
      switch ( *v90 )
      {
        case '?':
          v30 = 1;
          break;
        case 'G':
          BYTE1(v93) |= 2u;
          break;
        case 'M':
          v93 = (unsigned int)v93 | 0x10;
          break;
        case 'S':
          v93 = (unsigned int)v93 | 0x20;
          break;
        case 'T':
          BYTE1(v93) |= 4u;
          break;
        case 'a':
          v93 = (unsigned int)v93 | 2;
          break;
        case 'c':
        case 'y':
          v93 = (unsigned int)v93 | 0x20000000;
          break;
        case 'd':
          v93 = (unsigned int)v93 | 0x10000000;
          break;
        case 'e':
          v93 = (unsigned int)v93 | 0x80000000;
          break;
        case 'o':
          LOBYTE(v93) = v93 | 0x80;
          break;
        case 'w':
          v93 = (unsigned int)v93 | 1;
          break;
        case 'x':
          v93 = (unsigned int)v93 | 4;
          break;
        default:
          goto LABEL_192;
      }
      ++v90;
    }
    while ( v92 != v90 );
LABEL_156:
    v28 |= v93;
    v140 = v28 & 0x10;
    v94 = (v28 >> 9) & 1;
    if ( v30 && (v28 & 0x200) != 0 )
    {
      v108 = *(_QWORD *)(a1 + 8);
      v153[0] = (__int64)"Section cannot specifiy a group name while also acting as a member of the last group";
      v154 = 259;
      return sub_3909CF0(v108, v153, 0, 0, v31, v133);
    }
    goto LABEL_158;
  }
LABEL_261:
  v140 = 0;
  v30 = 0;
  LOBYTE(v94) = 0;
LABEL_158:
  v95 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, char *, size_t, char *))(**(_QWORD **)(a1 + 8) + 40LL))(
          *(_QWORD *)(a1 + 8),
          v27,
          v93,
          v92,
          v31,
          v32);
  if ( **(_DWORD **)(v95 + 8) != 25 )
    goto LABEL_166;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  v98 = **(_DWORD **)(v95 + 8);
  switch ( v98 )
  {
    case 45:
      goto LABEL_235;
    case 3:
LABEL_232:
      v27 = (__int64)&v146;
      if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 8) + 144LL))(
              *(_QWORD *)(a1 + 8),
              &v146) )
        goto LABEL_166;
      v99 = *(_QWORD *)(a1 + 8);
      v153[0] = (__int64)"expected identifier in directive";
      v154 = 259;
      goto LABEL_165;
    case 36:
LABEL_235:
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
      if ( **(_DWORD **)(v95 + 8) == 4 )
      {
        v109 = sub_3909460(*(_QWORD *)(a1 + 8));
        v110 = *(_QWORD *)(a1 + 8);
        v111 = *(_QWORD *)(v109 + 16);
        v112 = *(_QWORD *)(v109 + 8);
        v147 = v111;
        v146 = v112;
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v110 + 136LL))(v110);
        goto LABEL_166;
      }
      goto LABEL_232;
  }
  v99 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(v95 + 113) )
    v153[0] = (__int64)"expected '@<type>', '%<type>' or \"<type>\"";
  else
    v153[0] = (__int64)"expected '%<type>' or \"<type>\"";
  v154 = 259;
LABEL_165:
  v27 = (__int64)v153;
  if ( (unsigned __int8)sub_3909CF0(v99, v153, 0, 0, v96, v97) )
    return 1;
LABEL_166:
  v100 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8));
  if ( v147 )
  {
    if ( !v140 )
      goto LABEL_121;
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 25 )
    {
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
      v27 = (__int64)&v141;
      if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 8) + 200LL))(
             *(_QWORD *)(a1 + 8),
             &v141) )
      {
        return 1;
      }
      if ( v141 > 0 )
      {
LABEL_121:
        v10 = *(_DWORD **)(a1 + 8);
        if ( !(_BYTE)v94 )
          goto LABEL_126;
        v66 = (*(__int64 (__fastcall **)(_DWORD *))(*(_QWORD *)v10 + 40LL))(v10);
        if ( **(_DWORD **)(v66 + 8) != 25 )
        {
          HIBYTE(v154) = 1;
          v67 = "expected group name";
          goto LABEL_124;
        }
        (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
        v123 = *(_QWORD *)(a1 + 8);
        if ( **(_DWORD **)(v66 + 8) == 4 )
        {
          v127 = sub_3909460(v123);
          v128 = *(_QWORD *)(a1 + 8);
          v129 = *(_QWORD *)(v127 + 16);
          v130 = *(_QWORD **)(v127 + 8);
          v149 = v129;
          v148 = v130;
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v128 + 136LL))(v128);
        }
        else
        {
          v27 = (__int64)&v148;
          if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD **))(*(_QWORD *)v123 + 144LL))(v123, &v148) )
          {
            HIBYTE(v154) = 1;
            v67 = "invalid group name";
            goto LABEL_124;
          }
        }
        v10 = *(_DWORD **)(a1 + 8);
        if ( **(_DWORD **)(v66 + 8) != 25 )
          goto LABEL_126;
        (*(void (__fastcall **)(_DWORD *))(*(_QWORD *)v10 + 136LL))(v10);
        v124 = *(_QWORD *)(a1 + 8);
        s1 = 0;
        v151 = 0;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, void **))(*(_QWORD *)v124 + 144LL))(v124, &s1) )
        {
          HIBYTE(v154) = 1;
          v67 = "invalid linkage";
        }
        else
        {
          if ( v151 == 6 )
          {
            v27 = (__int64)"comdat";
            if ( !memcmp(s1, "comdat", 6u) )
            {
LABEL_125:
              v10 = *(_DWORD **)(a1 + 8);
              goto LABEL_126;
            }
          }
          HIBYTE(v154) = 1;
          v67 = "Linkage must be 'comdat'";
        }
LABEL_124:
        v68 = *(_QWORD *)(a1 + 8);
        v27 = (__int64)v153;
        v153[0] = (__int64)v67;
        LOBYTE(v154) = 3;
        if ( (unsigned __int8)sub_3909CF0(v68, v153, 0, 0, v64, v65) )
          return 1;
        goto LABEL_125;
      }
      v115 = *(_QWORD *)(a1 + 8);
      v153[0] = (__int64)"entry size must be positive";
      v154 = 259;
    }
    else
    {
      v115 = *(_QWORD *)(a1 + 8);
      v153[0] = (__int64)"expected the entry size";
      v154 = 259;
    }
    v27 = (__int64)v153;
    if ( (unsigned __int8)sub_3909CF0(v115, v153, 0, 0, v113, v114) )
      return 1;
    goto LABEL_121;
  }
  v10 = *(_DWORD **)(a1 + 8);
  if ( v140 )
  {
    HIBYTE(v154) = 1;
    v36 = "Mergeable section must specify the type";
    goto LABEL_50;
  }
  if ( (_BYTE)v94 )
  {
    HIBYTE(v154) = 1;
    v36 = "Group section must specify the type";
    goto LABEL_50;
  }
  if ( **(_DWORD **)(v100 + 8) != 9 )
  {
    HIBYTE(v154) = 1;
    v36 = "unexpected token in directive";
    goto LABEL_50;
  }
LABEL_126:
  if ( (v28 & 0x80u) != 0 )
  {
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_DWORD *))(*(_QWORD *)v10 + 40LL))(v10) + 8) != 25 )
    {
      HIBYTE(v154) = 1;
      v71 = "expected metadata symbol";
LABEL_129:
      v72 = *(_QWORD *)(a1 + 8);
      v27 = (__int64)v153;
      v153[0] = (__int64)v71;
      v29 = 0;
      LOBYTE(v154) = 3;
      v73 = sub_3909CF0(v72, v153, 0, 0, v69, v70);
      goto LABEL_130;
    }
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    v116 = *(_QWORD *)(a1 + 8);
    s1 = 0;
    v151 = 0;
    if ( (*(unsigned __int8 (__fastcall **)(__int64, void **))(*(_QWORD *)v116 + 144LL))(v116, &s1) )
    {
      HIBYTE(v154) = 1;
      v71 = "invalid metadata symbol";
      goto LABEL_129;
    }
    v117 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v27 = (__int64)v153;
    v154 = 261;
    v153[0] = (__int64)&s1;
    v118 = sub_38BD730(v117, (__int64)v153);
    v29 = v118;
    if ( v118 && (*(_WORD *)(v118 + 8) & 0x1C0) == 0x80 )
    {
      v121 = *(_QWORD *)v118 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_QWORD *)v29 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        goto LABEL_283;
      if ( (*(_BYTE *)(v29 + 9) & 0xC) == 8 )
      {
        *(_BYTE *)(v29 + 8) |= 4u;
        v125 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v29 + 24));
        v126 = v125 | *(_QWORD *)v29 & 7LL;
        *(_QWORD *)v29 = v126;
        if ( v125 )
        {
          v121 = v126 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v121 )
          {
            v121 = 0;
            if ( (*(_BYTE *)(v29 + 9) & 0xC) == 8 )
            {
              *(_BYTE *)(v29 + 8) |= 4u;
              v121 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v29 + 24));
              *(_QWORD *)v29 = v121 | *(_QWORD *)v29 & 7LL;
            }
          }
LABEL_283:
          if ( off_4CF6DB8 != (_UNKNOWN *)v121 )
            goto LABEL_131;
        }
      }
    }
    else
    {
      v29 = 0;
    }
    v154 = 1283;
    v122 = *(_QWORD *)(a1 + 8);
    v27 = (__int64)v153;
    v153[0] = (__int64)"symbol is not in a section: ";
    v153[1] = (__int64)&s1;
    v73 = sub_3909CF0(v122, v153, 0, 0, v119, v120);
LABEL_130:
    if ( !v73 )
    {
LABEL_131:
      v10 = *(_DWORD **)(a1 + 8);
      goto LABEL_132;
    }
    return 1;
  }
  v29 = 0;
LABEL_132:
  v74 = (*(__int64 (__fastcall **)(_DWORD *))(*(_QWORD *)v10 + 40LL))(v10);
  if ( **(_DWORD **)(v74 + 8) != 25 )
    goto LABEL_32;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  v75 = *(_QWORD *)(a1 + 8);
  s1 = 0;
  v151 = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, void **))(*(_QWORD *)v75 + 144LL))(v75, &s1) )
  {
    v78 = *(_QWORD *)(a1 + 8);
    v153[0] = (__int64)"expected identifier in directive";
    v154 = 259;
  }
  else
  {
    v78 = *(_QWORD *)(a1 + 8);
    if ( v151 == 6 && *(_DWORD *)s1 == 1902734965 && *((_WORD *)s1 + 2) == 25973 )
    {
      if ( **(_DWORD **)(v74 + 8) == 25 )
      {
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v78 + 136LL))(v78);
        v27 = (__int64)&v143;
        result = (*(__int64 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 8) + 200LL))(
                   *(_QWORD *)(a1 + 8),
                   &v143);
        if ( (_BYTE)result )
          return result;
        v78 = *(_QWORD *)(a1 + 8);
        if ( v143 < 0 )
        {
          HIBYTE(v154) = 1;
          v79 = "unique id must be positive";
        }
        else
        {
          if ( (unsigned int)v143 == v143 && v143 != 0xFFFFFFFFLL )
            goto LABEL_32;
          HIBYTE(v154) = 1;
          v79 = "unique id is too large";
        }
      }
      else
      {
        HIBYTE(v154) = 1;
        v79 = "expected commma";
      }
    }
    else
    {
      HIBYTE(v154) = 1;
      v79 = "expected 'unique'";
    }
    v153[0] = (__int64)v79;
    LOBYTE(v154) = 3;
  }
  v27 = (__int64)v153;
  if ( (unsigned __int8)sub_3909CF0(v78, v153, 0, 0, v76, v77) )
    return 1;
LABEL_32:
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 9 )
  {
    HIBYTE(v154) = 1;
    v33 = "unexpected token in directive";
LABEL_34:
    v34 = *(_QWORD *)(a1 + 8);
    v153[0] = (__int64)v33;
    LOBYTE(v154) = 3;
    return sub_3909CF0(v34, v153, 0, 0, v31, v32);
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  switch ( v147 )
  {
    case 0LL:
      if ( n > 4 && *(_DWORD *)v144 == 1953459758 && *((_BYTE *)v144 + 4) == 101 )
        goto LABEL_119;
      v132 = v144;
      v136 = n;
      v27 = n;
      v44 = v144;
      if ( (unsigned __int8)sub_3905F30(v144, n, ".init_array.", 0xCu) )
      {
LABEL_72:
        v38 = 14;
        goto LABEL_73;
      }
      v62 = sub_3905F30(v44, v27, ".bss.", 5u);
      v27 = v136;
      if ( v62 || (unsigned __int8)sub_3905F30(v132, v136, ".tbss.", 6u) )
        goto LABEL_141;
      v27 = v136;
      if ( !(unsigned __int8)sub_3905F30(v132, v136, ".fini_array.", 0xCu) )
      {
        v63 = sub_3905F30(v132, v136, ".preinit_array.", 0xFu);
        v38 = 1;
        if ( !v63 )
          goto LABEL_73;
        goto LABEL_112;
      }
LABEL_145:
      v38 = 15;
      goto LABEL_73;
    case 10LL:
      if ( *(_QWORD *)v146 == 0x7272615F74696E69LL && *(_WORD *)(v146 + 8) == 31073 )
        goto LABEL_72;
      if ( *(_QWORD *)v146 != 0x7272615F696E6966LL || *(_WORD *)(v146 + 8) != 31073 )
        goto LABEL_61;
      goto LABEL_145;
    case 13LL:
      if ( *(_QWORD *)v146 != 0x5F74696E69657270LL
        || *(_DWORD *)(v146 + 8) != 1634890337
        || *(_BYTE *)(v146 + 12) != 121 )
      {
        goto LABEL_61;
      }
LABEL_112:
      v38 = 16;
      goto LABEL_73;
    case 6LL:
      if ( *(_DWORD *)v146 != 1768058734 || *(_WORD *)(v146 + 4) != 29556 )
      {
        if ( *(_DWORD *)v146 == 1769434741 && *(_WORD *)(v146 + 4) == 25710 )
        {
          v38 = 1879048193;
          goto LABEL_73;
        }
        goto LABEL_61;
      }
LABEL_141:
      v38 = 8;
      goto LABEL_73;
    case 8LL:
      if ( *(_QWORD *)v146 == 0x73746962676F7270LL )
      {
        v38 = 1;
        goto LABEL_73;
      }
      goto LABEL_61;
    case 4LL:
      if ( *(_DWORD *)v146 != 1702129518 )
        goto LABEL_61;
LABEL_119:
      v38 = 7;
      goto LABEL_73;
    case 11LL:
      if ( *(_QWORD *)v146 == 0x72646F5F6D766C6CLL && *(_WORD *)(v146 + 8) == 24948 && *(_BYTE *)(v146 + 10) == 98 )
      {
        v38 = 1879002112;
        goto LABEL_73;
      }
      goto LABEL_61;
  }
  if ( v147 != 19 )
  {
    if ( v147 == 23 )
    {
      v37 = *(_QWORD *)v146 ^ 0x6C61635F6D766C6CLL | *(_QWORD *)(v146 + 8) ^ 0x5F68706172675F6CLL;
      if ( !v37
        && *(_DWORD *)(v146 + 16) == 1718579824
        && *(_WORD *)(v146 + 20) == 27753
        && *(_BYTE *)(v146 + 22) == 101 )
      {
        v38 = 1879002114;
        goto LABEL_73;
      }
    }
LABEL_61:
    v27 = v147;
    if ( sub_16D2B80(v146, v147, 0, (unsigned __int64 *)v153) || (v38 = v153[0], v153[0] != LODWORD(v153[0])) )
    {
      HIBYTE(v154) = 1;
      v33 = "unknown section type";
      goto LABEL_34;
    }
    goto LABEL_73;
  }
  v37 = *(_QWORD *)v146 ^ 0x6E696C5F6D766C6CLL | *(_QWORD *)(v146 + 8) ^ 0x6974706F5F72656BLL;
  if ( v37 || *(_WORD *)(v146 + 16) != 28271 || *(_BYTE *)(v146 + 18) != 115 )
    goto LABEL_61;
  v38 = 1879002113;
LABEL_73:
  if ( v30 )
  {
    v137 = v38;
    v45 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v38 = v137;
    v37 = v45;
    v46 = *(unsigned int *)(v45 + 120);
    if ( (_DWORD)v46 )
    {
      v47 = *(_QWORD *)(*(_QWORD *)(v37 + 112) + 32 * v46 - 32);
      if ( v47 )
      {
        v48 = *(_BYTE **)(v47 + 184);
        if ( v48 )
        {
          if ( (*v48 & 4) != 0 )
          {
            v49 = (__int64 *)*((_QWORD *)v48 - 1);
            v37 = *v49;
            v50 = v49 + 2;
          }
          else
          {
            v37 = 0;
            v50 = 0;
          }
          v148 = v50;
          v28 |= 0x200u;
          v149 = v37;
        }
      }
    }
  }
  v138 = v38;
  v51 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a1 + 8) + 48LL))(
          *(_QWORD *)(a1 + 8),
          v27,
          v38,
          v37);
  v152 = 261;
  v154 = 261;
  v153[0] = (__int64)&v148;
  s1 = &v144;
  v52 = sub_38C3B80(v51, (__int64)&s1, v138, v28, v141, (__int64)v153, v143, v29);
  v53 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v53 + 160LL))(v53, v52, v142);
  if ( !*(_BYTE *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8)) + 1041) )
    return 0;
  v54 = (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8), v131);
  v153[0] = v52;
  if ( !(unsigned __int8)sub_38EA790(v54 + 1048, v153) )
    return 0;
  if ( *(_WORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8)) + 1160) <= 2u )
  {
    v80 = *(_QWORD *)(a1 + 8);
    v153[0] = (__int64)"DWARF2 only supports one section per compilation unit";
    v154 = 259;
    (*(void (__fastcall **)(__int64, __int64, __int64 *, _QWORD, _QWORD))(*(_QWORD *)v80 + 120LL))(v80, a3, v153, 0, 0);
  }
  if ( *(_QWORD *)(v52 + 8) )
    return 0;
  v81 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
  v82 = sub_38BFA60(v81, 1);
  v83 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v83 + 176LL))(v83, v82, 0);
  *(_QWORD *)(v52 + 8) = v82;
  return 0;
}
