// Function: sub_BF0CF0
// Address: 0xbf0cf0
//
void __fastcall sub_BF0CF0(__int64 *a1, __int64 a2)
{
  __int64 v4; // r13
  __int64 v5; // rbx
  __int64 v6; // r14
  int v7; // ebx
  __int64 v8; // rbx
  __int64 v9; // rdi
  _BYTE *v10; // rax
  __int64 v11; // rdi
  _BYTE *v12; // rax
  const char *v13; // rax
  __int64 v14; // r8
  _BYTE *v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned __int16 v19; // dx
  const char *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // r10
  int v25; // ecx
  __int64 v26; // rcx
  __int64 v27; // r9
  int v28; // edx
  char v29; // al
  __int64 v30; // r10
  char v31; // al
  __int64 v32; // rsi
  const char **v33; // r13
  __int64 v34; // rax
  const char **v35; // rbx
  const char *v36; // rax
  unsigned __int8 v37; // dl
  _BYTE *v38; // rdi
  __int64 v39; // rdx
  const void *v40; // rdi
  const char *v41; // rax
  unsigned __int8 *v42; // rax
  unsigned __int8 *v43; // rax
  unsigned __int8 *v44; // r13
  __int64 v45; // rbx
  int v46; // eax
  __int64 v47; // rax
  _QWORD *v48; // rax
  const char *v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // r13
  unsigned int v53; // ebx
  unsigned int v54; // eax
  const char *v55; // rdx
  __int64 v56; // r8
  __int64 v57; // rax
  __int64 v58; // rcx
  __int64 v59; // rsi
  __int64 v60; // rdx
  unsigned __int16 v61; // dx
  __int64 v62; // rdx
  __int64 v63; // rdx
  char v64; // al
  __int64 v65; // rax
  const char *v66; // rax
  unsigned __int8 v67; // dl
  const char *v68; // rax
  _BYTE *v69; // rdx
  __int64 v70; // rax
  __int64 v71; // rcx
  unsigned __int8 v72; // al
  _BYTE *v73; // rax
  __int64 v74; // rax
  __int64 i; // rdx
  const char *v76; // rax
  const char *v77; // rax
  const char **v78; // rbx
  const char **v79; // r13
  const char **v80; // rcx
  int v81; // r13d
  int v82; // ebx
  int v83; // r8d
  unsigned int v84; // edx
  int v85; // eax
  const char *v86; // rax
  _QWORD *v87; // rax
  __int64 v88; // r8
  _BYTE *v89; // rax
  const char *v90; // rax
  __int64 v91; // rax
  _QWORD *v92; // rax
  const char *v93; // rax
  const char *v94; // rax
  const char *v95; // rax
  const char *v96; // rax
  __int64 v97; // rax
  const char *v98; // rax
  int v99; // [rsp+8h] [rbp-208h]
  __int64 v100; // [rsp+10h] [rbp-200h]
  __int64 v101; // [rsp+10h] [rbp-200h]
  __int64 v102; // [rsp+10h] [rbp-200h]
  __int64 v103; // [rsp+18h] [rbp-1F8h]
  __int64 v104; // [rsp+18h] [rbp-1F8h]
  int v105; // [rsp+18h] [rbp-1F8h]
  __int64 v106; // [rsp+18h] [rbp-1F8h]
  unsigned int v107; // [rsp+18h] [rbp-1F8h]
  int v108; // [rsp+18h] [rbp-1F8h]
  const char **v109; // [rsp+18h] [rbp-1F8h]
  char v110; // [rsp+20h] [rbp-1F0h]
  __int64 j; // [rsp+20h] [rbp-1F0h]
  _BYTE *v112; // [rsp+20h] [rbp-1F0h]
  const char **v113; // [rsp+20h] [rbp-1F0h]
  __int64 v114; // [rsp+20h] [rbp-1F0h]
  _QWORD *v115; // [rsp+28h] [rbp-1E8h]
  int v116; // [rsp+28h] [rbp-1E8h]
  __int64 v117; // [rsp+28h] [rbp-1E8h]
  _BYTE *v118; // [rsp+28h] [rbp-1E8h]
  _BYTE *v119; // [rsp+28h] [rbp-1E8h]
  __int64 v120; // [rsp+28h] [rbp-1E8h]
  __int64 v121; // [rsp+28h] [rbp-1E8h]
  __int64 v122; // [rsp+28h] [rbp-1E8h]
  __int64 v123; // [rsp+28h] [rbp-1E8h]
  const char **v124; // [rsp+28h] [rbp-1E8h]
  const char *v125; // [rsp+38h] [rbp-1D8h] BYREF
  _BYTE *v126; // [rsp+40h] [rbp-1D0h] BYREF
  const char *v127; // [rsp+48h] [rbp-1C8h] BYREF
  _BYTE *v128[4]; // [rsp+50h] [rbp-1C0h] BYREF
  const char **v129; // [rsp+70h] [rbp-1A0h] BYREF
  __int64 v130; // [rsp+78h] [rbp-198h]
  _BYTE v131[64]; // [rsp+80h] [rbp-190h] BYREF
  const char *v132; // [rsp+C0h] [rbp-150h] BYREF
  char *v133; // [rsp+C8h] [rbp-148h]
  __int64 v134; // [rsp+D0h] [rbp-140h]
  int v135; // [rsp+D8h] [rbp-138h]
  char v136; // [rsp+DCh] [rbp-134h]
  char v137; // [rsp+E0h] [rbp-130h] BYREF
  char v138; // [rsp+E1h] [rbp-12Fh]

  sub_BE9180((__int64)a1, a2);
  v4 = *(_QWORD *)(a2 + 24);
  v5 = *(_QWORD *)(a2 + 104);
  v6 = a1[18];
  if ( v6 != sub_B2BE50(a2) )
  {
    v129 = (const char **)a2;
    v13 = "Function context does not match Module context!";
    v138 = 1;
LABEL_15:
    v132 = v13;
    v137 = 3;
    sub_BEDA40(a1, (__int64)&v132, (_BYTE **)&v129);
    return;
  }
  if ( (*(_BYTE *)(a2 + 32) & 0xF) == 0xA )
  {
    v129 = (const char **)a2;
    v13 = "Functions may not have common linkage";
    v138 = 1;
    goto LABEL_15;
  }
  if ( *(_DWORD *)(v4 + 12) - 1 != (_DWORD)v5 )
  {
    v138 = 1;
    v132 = "# formal arguments must match # of arguments for function type!";
    v137 = 3;
    sub_BDBF70(a1, (__int64)&v132);
    if ( *a1 )
    {
      sub_BDBD80((__int64)a1, (_BYTE *)a2);
      v14 = *a1;
      v15 = *(_BYTE **)(*a1 + 32);
      if ( (unsigned __int64)v15 >= *(_QWORD *)(*a1 + 24) )
      {
        v14 = sub_CB5D20(*a1, 32);
      }
      else
      {
        *(_QWORD *)(v14 + 32) = v15 + 1;
        *v15 = 32;
      }
      sub_A587F0(v4, v14, 0, 0);
    }
    return;
  }
  if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL) + 8LL) == 13 )
  {
    v129 = (const char **)a2;
    v13 = "Functions cannot return aggregate values!";
    v138 = 1;
    goto LABEL_15;
  }
  v115 = (_QWORD *)(a2 + 120);
  if ( ((unsigned __int8)sub_A74710((_QWORD *)(a2 + 120), 1, 85) || (unsigned __int8)sub_A74710(v115, 2, 85))
    && *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL) + 8LL) != 7 )
  {
    v129 = (const char **)a2;
    v13 = "Invalid struct return type!";
    v138 = 1;
    goto LABEL_15;
  }
  v7 = *(_DWORD *)(v4 + 12);
  v125 = *(const char **)(a2 + 120);
  v132 = v125;
  if ( v7 + 1 < (unsigned int)sub_A74480((__int64)&v132) )
  {
    v129 = (const char **)a2;
    v20 = "Attribute after last parameter!";
    v138 = 1;
LABEL_30:
    v132 = v20;
    v137 = 3;
    sub_BEDA40(a1, (__int64)&v132, (_BYTE **)&v129);
    return;
  }
  v8 = *(_QWORD *)(a2 + 40);
  if ( *(_BYTE *)(a2 + 128) != *(_BYTE *)(v8 + 872) )
  {
    v138 = 1;
    v132 = "Function debug format should match parent module";
    v137 = 3;
    sub_BDD6D0(a1, (__int64)&v132);
    if ( *a1 )
    {
      sub_BDBD80((__int64)a1, (_BYTE *)a2);
      v9 = sub_CB59D0(*a1, *(unsigned __int8 *)(a2 + 128));
      v10 = *(_BYTE **)(v9 + 32);
      if ( (unsigned __int64)v10 >= *(_QWORD *)(v9 + 24) )
      {
        sub_CB5D20(v9, 10);
      }
      else
      {
        *(_QWORD *)(v9 + 32) = v10 + 1;
        *v10 = 10;
      }
      sub_BD9A50(*a1, v8);
      v11 = sub_CB59D0(*a1, *(unsigned __int8 *)(v8 + 872));
      v12 = *(_BYTE **)(v11 + 32);
      if ( (unsigned __int64)v12 >= *(_QWORD *)(v11 + 24) )
      {
        sub_CB5D20(v11, 10);
      }
      else
      {
        *(_QWORD *)(v11 + 32) = v12 + 1;
        *v12 = 10;
      }
    }
    return;
  }
  v110 = *(_BYTE *)(a2 + 33) & 0x20;
  sub_BEB370((__int64)a1, v4, (__int64)v125, (_BYTE *)a2, v110 != 0, 0);
  if ( (unsigned __int8)sub_A73ED0(&v125, 4) )
  {
    v129 = (const char **)a2;
    v20 = "Attribute 'builtin' can only be applied to a callsite.";
    v138 = 1;
    goto LABEL_30;
  }
  if ( (unsigned __int8)sub_A74390((__int64 *)&v125, 82, 0) )
  {
    v129 = (const char **)a2;
    v20 = "Attribute 'elementtype' can only be applied to a callsite.";
    v138 = 1;
    goto LABEL_30;
  }
  v16 = 20;
  if ( (unsigned __int8)sub_A73ED0(&v125, 20) )
  {
    if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
    {
      sub_B2C6D0(a2, 20, v17, v18);
      v56 = *(_QWORD *)(a2 + 96);
      v57 = v56 + 40LL * *(_QWORD *)(a2 + 104);
      if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
      {
        v106 = v56 + 40LL * *(_QWORD *)(a2 + 104);
        sub_B2C6D0(a2, 20, v60, v18);
        v56 = *(_QWORD *)(a2 + 96);
        v57 = v106;
      }
    }
    else
    {
      v56 = *(_QWORD *)(a2 + 96);
      v57 = v56 + 40LL * *(_QWORD *)(a2 + 104);
    }
    for ( ; v57 != v56; v56 += 40 )
    {
      if ( *(_QWORD *)(v56 + 16) )
      {
        v119 = (_BYTE *)v56;
        v138 = 1;
        v132 = "cannot use argument of naked function";
        v137 = 3;
        sub_BDBF70(a1, (__int64)&v132);
        if ( !*a1 )
          return;
        goto LABEL_106;
      }
    }
  }
  v19 = *(_WORD *)(a2 + 2);
  switch ( (v19 >> 4) & 0x3FF )
  {
    case 8:
    case 9:
    case 0x47:
    case 0x48:
    case 0x4D:
      goto LABEL_33;
    case 0x4C:
    case 0x5B:
    case 0x68:
    case 0x69:
      if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL) + 8LL) == 7 )
        goto LABEL_96;
      v129 = (const char **)a2;
      v20 = "Calling convention requires void return type";
      v138 = 1;
      goto LABEL_30;
    case 0x53:
      if ( !*(_QWORD *)(a2 + 104) )
        goto LABEL_35;
      v16 = 1;
      if ( (unsigned __int8)sub_A74710(&v125, 1, 81) )
        goto LABEL_34;
      v129 = (const char **)a2;
      v20 = "Calling convention parameter requires byval";
      v138 = 1;
      goto LABEL_30;
    case 0x57:
    case 0x58:
    case 0x59:
    case 0x5A:
    case 0x5D:
LABEL_96:
      if ( (unsigned __int8)sub_A74710(v115, 1, 85) || (v16 = 2, (unsigned __int8)sub_A74710(v115, 2, 85)) )
      {
        v129 = (const char **)a2;
        v20 = "Calling convention does not allow sret";
        v138 = 1;
        goto LABEL_30;
      }
      v61 = *(_WORD *)(a2 + 2);
      if ( ((v61 >> 4) & 0x3FF) == 0x4C )
        goto LABEL_33;
      v62 = v61 & 1;
      v99 = *(_DWORD *)(a1[17] + 4);
      if ( (_DWORD)v62 )
      {
        sub_B2C6D0(a2, 2, v62, v18);
        v63 = *(_QWORD *)(a2 + 96);
        v102 = v63 + 40LL * *(_QWORD *)(a2 + 104);
        if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
        {
          sub_B2C6D0(a2, 2, v63, v71);
          v63 = *(_QWORD *)(a2 + 96);
        }
      }
      else
      {
        v63 = *(_QWORD *)(a2 + 96);
        v102 = v63 + 40LL * *(_QWORD *)(a2 + 104);
      }
      v122 = v63;
      v18 = 0;
      if ( v63 == v102 )
        goto LABEL_33;
      break;
    default:
      goto LABEL_35;
  }
  do
  {
    v107 = v18 + 1;
    if ( (unsigned __int8)sub_A74710(&v125, (int)v18 + 1, 81) )
    {
      v129 = (const char **)a2;
      v20 = "Calling convention disallows byval";
      v138 = 1;
      goto LABEL_30;
    }
    if ( (unsigned __int8)sub_A74710(&v125, v107, 84) )
    {
      v129 = (const char **)a2;
      v20 = "Calling convention disallows preallocated";
      v138 = 1;
      goto LABEL_30;
    }
    if ( (unsigned __int8)sub_A74710(&v125, v107, 83) )
    {
      v129 = (const char **)a2;
      v20 = "Calling convention disallows inalloca";
      v138 = 1;
      goto LABEL_30;
    }
    v16 = v107;
    v64 = sub_A74710(&v125, v107, 80);
    v18 = v107;
    if ( v64 )
    {
      v65 = *(_QWORD *)(v122 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v65 + 8) - 17 <= 1 )
        v65 = **(_QWORD **)(v65 + 16);
      if ( *(_DWORD *)(v65 + 8) >> 8 == v99 )
      {
        v129 = (const char **)a2;
        v20 = "Calling convention disallows stack byref";
        v138 = 1;
        goto LABEL_30;
      }
    }
    v122 += 40;
  }
  while ( v102 != v122 );
LABEL_33:
  if ( *(_DWORD *)(*(_QWORD *)(a2 + 24) + 8LL) >> 8 )
  {
    v129 = (const char **)a2;
    v20 = "Calling convention does not support varargs or perfect forwarding!";
    v138 = 1;
    goto LABEL_30;
  }
LABEL_34:
  v19 = *(_WORD *)(a2 + 2);
LABEL_35:
  v21 = v19 & 1;
  if ( (_DWORD)v21 )
  {
    sub_B2C6D0(a2, v16, v21, v18);
    v22 = *(_QWORD *)(a2 + 96);
    v23 = v22 + 40LL * *(_QWORD *)(a2 + 104);
    if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
    {
      v120 = v22 + 40LL * *(_QWORD *)(a2 + 104);
      sub_B2C6D0(a2, v16, v22, v58);
      v22 = *(_QWORD *)(a2 + 96);
      v23 = v120;
    }
  }
  else
  {
    v22 = *(_QWORD *)(a2 + 96);
    v23 = v22 + 40LL * *(_QWORD *)(a2 + 104);
  }
  if ( v22 != v23 )
  {
    v24 = v22;
    v25 = 0;
    while ( 1 )
    {
      v26 = (unsigned int)(v25 + 1);
      v27 = *(_QWORD *)(*(_QWORD *)(v4 + 16) + 8 * v26);
      if ( v27 != *(_QWORD *)(v24 + 8) )
      {
        v112 = (_BYTE *)v24;
        v123 = *(_QWORD *)(*(_QWORD *)(v4 + 16) + 8 * v26);
        v138 = 1;
        v132 = "Argument value does not match function argument type!";
        v137 = 3;
        sub_BDBF70(a1, (__int64)&v132);
        if ( *a1 )
        {
          sub_BDBD80((__int64)a1, v112);
          if ( v123 )
            sub_BD9860(*a1, v123);
        }
        return;
      }
      v28 = *(unsigned __int8 *)(v27 + 8);
      if ( (_BYTE)v28 == 13 || v28 == 7 )
        break;
      if ( !v110 )
      {
        switch ( (_BYTE)v28 )
        {
          case 9:
            v129 = (const char **)a2;
            v77 = "Function takes metadata but isn't an intrinsic";
            v128[0] = (_BYTE *)v24;
            v138 = 1;
            goto LABEL_178;
          case 0xB:
            v129 = (const char **)a2;
            v77 = "Function takes token but isn't an intrinsic";
            v128[0] = (_BYTE *)v24;
            v138 = 1;
            goto LABEL_178;
          case 0xA:
            v129 = (const char **)a2;
            v77 = "Function takes x86_amx but isn't an intrinsic";
            v128[0] = (_BYTE *)v24;
            v138 = 1;
LABEL_178:
            v132 = v77;
            v137 = 3;
            sub_BEDB40(a1, (__int64)&v132, v128, (_BYTE **)&v129);
            return;
        }
      }
      v116 = v26;
      v100 = v24;
      v103 = v23;
      v29 = sub_A74710(&v125, v26, 74);
      v25 = v116;
      v23 = v103;
      v30 = v100;
      if ( v29 )
      {
        v59 = v100;
        v101 = v103;
        v105 = v116;
        v121 = v30;
        sub_BDC190((__int64)a1, v59);
        v23 = v101;
        v25 = v105;
        v30 = v121;
      }
      v24 = v30 + 40;
      if ( v23 == v24 )
        goto LABEL_49;
    }
    v119 = (_BYTE *)v24;
    v138 = 1;
    v132 = "Function arguments must have first-class types!";
    v137 = 3;
    sub_BDBF70(a1, (__int64)&v132);
    if ( !*a1 )
      return;
LABEL_106:
    sub_BDBD80((__int64)a1, v119);
    return;
  }
LABEL_49:
  if ( !v110 )
  {
    v31 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL) + 8LL);
    if ( v31 == 11 )
    {
      v129 = (const char **)a2;
      v20 = "Function returns a token but isn't an intrinsic";
      v138 = 1;
      goto LABEL_30;
    }
    if ( v31 == 10 )
    {
      v129 = (const char **)a2;
      v20 = "Function returns a x86_amx but isn't an intrinsic";
      v138 = 1;
      goto LABEL_30;
    }
  }
  v32 = (__int64)&v129;
  v129 = (const char **)v131;
  v130 = 0x400000000LL;
  sub_B9A9D0(a2, (__int64)&v129);
  v33 = v129;
  v34 = 2LL * (unsigned int)v130;
  v35 = &v129[v34];
  if ( v129 == &v129[v34] )
    goto LABEL_65;
  while ( 2 )
  {
    if ( *(_DWORD *)v33 == 2 )
    {
      v36 = v33[1];
      v128[0] = v36;
      v37 = *(v36 - 16);
      if ( (v37 & 2) != 0 )
      {
        if ( *((_DWORD *)v36 - 6) <= 1u )
          goto LABEL_224;
        v38 = (_BYTE *)**((_QWORD **)v36 - 4);
        if ( !v38 )
        {
LABEL_150:
          v138 = 1;
          v41 = "first operand should not be null";
          goto LABEL_64;
        }
      }
      else
      {
        if ( ((*((_WORD *)v36 - 8) >> 6) & 0xFu) <= 1 )
        {
LABEL_224:
          v138 = 1;
          v41 = "!prof annotations should have no less than 2 operands";
          goto LABEL_64;
        }
        v38 = *(_BYTE **)&v36[-16 - 8LL * ((v37 >> 2) & 0xF)];
        if ( !v38 )
          goto LABEL_150;
      }
      if ( *v38 )
      {
        v138 = 1;
        v41 = "expected string with name of the !prof annotation";
        goto LABEL_64;
      }
      v40 = (const void *)sub_B91420((__int64)v38);
      if ( v39 == 20 )
      {
        v32 = (__int64)"function_entry_count";
        if ( memcmp(v40, "function_entry_count", 0x14u) )
          goto LABEL_63;
      }
      else if ( v39 != 30
             || (v32 = (__int64)"synthetic_function_entry_count", memcmp(v40, "synthetic_function_entry_count", 0x1Eu)) )
      {
LABEL_63:
        v138 = 1;
        v41 = "first operand should be 'function_entry_count' or 'synthetic_function_entry_count'";
        goto LABEL_64;
      }
      v72 = *(v128[0] - 16);
      if ( (v72 & 2) != 0 )
      {
        v73 = *(_BYTE **)(*((_QWORD *)v128[0] - 4) + 8LL);
        if ( !v73 )
          goto LABEL_158;
      }
      else
      {
        v73 = *(_BYTE **)&v128[0][-16 - 8LL * ((v72 >> 2) & 0xF) + 8];
        if ( !v73 )
        {
LABEL_158:
          v138 = 1;
          v41 = "second operand should not be null";
          goto LABEL_64;
        }
      }
      if ( *v73 != 1 )
      {
        v138 = 1;
        v41 = "expected integer argument to function_entry_count";
        goto LABEL_64;
      }
      goto LABEL_55;
    }
    if ( *(_DWORD *)v33 != 36 )
    {
LABEL_55:
      v33 += 2;
      if ( v35 == v33 )
        goto LABEL_65;
      continue;
    }
    break;
  }
  v66 = v33[1];
  v128[0] = v66;
  v67 = *(v66 - 16);
  if ( (v67 & 2) != 0 )
  {
    if ( *((_DWORD *)v66 - 6) != 1 )
      goto LABEL_231;
    v68 = (const char *)*((_QWORD *)v66 - 4);
    v69 = *(_BYTE **)v68;
    if ( !*(_QWORD *)v68 )
    {
LABEL_164:
      v138 = 1;
      v41 = "!kcfi_type operand must not be null";
      goto LABEL_64;
    }
    goto LABEL_140;
  }
  if ( ((*((_WORD *)v66 - 8) >> 6) & 0xF) == 1 )
  {
    v68 = &v66[-16 - 8LL * ((v67 >> 2) & 0xF)];
    v69 = *(_BYTE **)v68;
    if ( !*(_QWORD *)v68 )
      goto LABEL_164;
LABEL_140:
    if ( *v69 != 1 )
    {
      v138 = 1;
      v41 = "expected a constant operand for !kcfi_type";
      goto LABEL_64;
    }
    v70 = *(_QWORD *)(*(_QWORD *)v68 + 136LL);
    if ( *(_BYTE *)v70 != 17 || *(_BYTE *)(*(_QWORD *)(v70 + 8) + 8LL) != 12 )
    {
      v138 = 1;
      v41 = "expected a constant integer operand for !kcfi_type";
      goto LABEL_64;
    }
    if ( *(_DWORD *)(v70 + 32) != 32 )
    {
      v138 = 1;
      v41 = "expected a 32-bit integer constant operand for !kcfi_type";
      goto LABEL_64;
    }
    goto LABEL_55;
  }
LABEL_231:
  v138 = 1;
  v41 = "!kcfi_type must have exactly one operand";
LABEL_64:
  v32 = (__int64)&v132;
  v132 = v41;
  v137 = 3;
  sub_BECB10(a1, (__int64)&v132, (const char **)v128);
LABEL_65:
  if ( (*(_BYTE *)(a2 + 2) & 8) != 0 )
  {
    v42 = (unsigned __int8 *)sub_B2E500(a2);
    v43 = sub_BD3990(v42, v32);
    v44 = v43;
    if ( !*v43 )
    {
      v45 = *((_QWORD *)v43 + 5);
      v117 = *(_QWORD *)(a2 + 40);
      if ( v117 != v45 )
      {
        v88 = *a1;
        v138 = 1;
        v132 = "Referencing personality function in another module!";
        v137 = 3;
        if ( v88 )
        {
          v32 = v88;
          v114 = v88;
          sub_CA0E80(&v132, v88);
          v89 = *(_BYTE **)(v114 + 32);
          if ( (unsigned __int64)v89 >= *(_QWORD *)(v114 + 24) )
          {
            v32 = 10;
            sub_CB5D20(v114, 10);
          }
          else
          {
            *(_QWORD *)(v114 + 32) = v89 + 1;
            *v89 = 10;
          }
        }
        *((_BYTE *)a1 + 152) = 1;
        if ( *a1 )
        {
          sub_BDBD80((__int64)a1, (_BYTE *)a2);
          sub_BD9A50(*a1, v117);
          sub_BDBD80((__int64)a1, v44);
          v32 = v45;
          sub_BD9A50(*a1, v45);
        }
        goto LABEL_172;
      }
    }
  }
  sub_BEDCB0((__int64)(a1 + 114), v32);
  if ( (*(_BYTE *)(a2 + 35) & 8) != 0 )
  {
    if ( (_DWORD)v130 )
    {
      v32 = (__int64)&v132;
      v90 = v129[1];
      v127 = (const char *)a2;
      v138 = 1;
      v128[0] = v90;
      v132 = "unmaterialized function cannot have metadata";
      v137 = 3;
      sub_BEDF50(a1, (__int64)&v132, (_BYTE **)&v127, (const char **)v128);
      goto LABEL_172;
    }
    goto LABEL_70;
  }
  if ( !sub_B2FC80(a2) )
  {
    if ( v110 )
    {
      v128[0] = (_BYTE *)a2;
      v96 = "llvm intrinsics cannot be defined!";
      v138 = 1;
      goto LABEL_239;
    }
    v74 = *(_QWORD *)(a2 + 80);
    if ( !v74 )
    {
      v126 = 0;
      BUG();
    }
    v126 = (_BYTE *)(v74 - 24);
    for ( i = *(_QWORD *)(v74 - 8); i; i = *(_QWORD *)(i + 8) )
    {
      if ( (unsigned __int8)(**(_BYTE **)(i + 24) - 30) <= 0xAu )
      {
        v138 = 1;
        v76 = "Entry block to function must not have predecessors!";
LABEL_171:
        v32 = (__int64)&v132;
        v132 = v76;
        v137 = 3;
        sub_BEE1B0(a1, (__int64)&v132, &v126);
        goto LABEL_172;
      }
    }
    if ( (*(_WORD *)(v74 - 22) & 0x7FFF) != 0 )
    {
      v97 = sub_AC3FA0(v74 - 24);
      if ( (unsigned __int8)sub_AC2D20(v97) )
      {
        v138 = 1;
        v76 = "blockaddress may not be used with the entry block!";
        goto LABEL_171;
      }
    }
    v80 = v129;
    v124 = &v129[2 * (unsigned int)v130];
    if ( v124 != v129 )
    {
      v81 = 0;
      v82 = 0;
      v83 = 0;
      do
      {
        v85 = *(_DWORD *)v80;
        if ( *(_DWORD *)v80 == 2 )
        {
          if ( v82 )
          {
            v128[0] = (_BYTE *)a2;
            v98 = "function must have a single !prof attachment";
            v138 = 1;
LABEL_246:
            v32 = (__int64)&v132;
            v132 = v98;
            v137 = 3;
            sub_BEDF50(a1, (__int64)&v132, v128, v80 + 1);
            goto LABEL_172;
          }
          v84 = 0;
          v82 = 1;
        }
        else if ( v85 == 36 )
        {
          if ( v81 )
          {
            v128[0] = (_BYTE *)a2;
            v98 = "function must have a single !kcfi_type attachment";
            v138 = 1;
            goto LABEL_246;
          }
          v84 = 0;
          v81 = 1;
        }
        else
        {
          v84 = 0;
          if ( !v85 )
          {
            if ( v83 )
            {
              v128[0] = (_BYTE *)a2;
              v94 = "function must have a single !dbg attachment";
              v138 = 1;
              goto LABEL_228;
            }
            v86 = v80[1];
            if ( *v86 != 18 )
            {
              v128[0] = (_BYTE *)a2;
              v94 = "function !dbg attachment must be a subprogram";
              v138 = 1;
LABEL_228:
              v32 = (__int64)&v132;
              v132 = v94;
              v137 = 3;
              sub_BEE2B0(a1, (__int64)&v132, v128, v80 + 1);
              goto LABEL_172;
            }
            if ( (v86[1] & 0x7F) != 1 )
            {
              v128[0] = (_BYTE *)a2;
              v95 = "function definition may only have a distinct !dbg attachment";
              v138 = 1;
LABEL_234:
              v32 = (__int64)&v132;
              v132 = v95;
              v137 = 3;
              sub_BEE0A0(a1, (__int64)&v132, v128);
              goto LABEL_172;
            }
            v109 = v80;
            v127 = v80[1];
            v87 = sub_BF0AA0((__int64)(a1 + 92), (__int64 *)&v127);
            v80 = v109;
            if ( *v87 && *v87 != a2 )
            {
              v32 = (__int64)&v132;
              v128[0] = (_BYTE *)a2;
              v138 = 1;
              v132 = "DISubprogram attached to more than one function";
              v137 = 3;
              sub_BEE410((__int64)a1, (__int64)&v132, &v127, v128);
              goto LABEL_172;
            }
            *v87 = a2;
            v84 = 1;
            v83 = 1;
          }
        }
        v32 = (__int64)v80[1];
        v108 = v83;
        v113 = v80;
        sub_BE3890((__int64)a1, v32, v84);
        v83 = v108;
        v80 = v113 + 2;
      }
      while ( v124 != v113 + 2 );
    }
LABEL_70:
    if ( (*(_BYTE *)(a2 + 33) & 0x20) != 0 && !*(_QWORD *)(*(_QWORD *)(a2 + 40) + 160LL) )
    {
      v32 = (__int64)v128;
      if ( (unsigned __int8)sub_B2DDD0(a2, v128, 0, 1, 0, 1, 0) )
      {
        v32 = (__int64)&v132;
        v138 = 1;
        v132 = "Invalid user of intrinsic instruction!";
        v137 = 3;
        sub_BEE560(a1, (__int64)&v132, v128);
        goto LABEL_172;
      }
    }
    v46 = *(_DWORD *)(a2 + 36);
    if ( v46 == 147 )
    {
      v91 = *(_QWORD *)(a2 + 24);
      if ( *(_DWORD *)(v91 + 12) == 2 )
      {
        v92 = *(_QWORD **)(v91 + 16);
        if ( *(_BYTE *)(*v92 + 8LL) == 14 )
        {
          if ( *v92 == v92[1] )
          {
LABEL_77:
            v49 = (const char *)sub_B92180(a2);
            v127 = v49;
            *((_BYTE *)a1 + 825) = v49 != 0;
            if ( v49 )
            {
              v132 = 0;
              v133 = &v137;
              v128[1] = &v127;
              v50 = *(_QWORD *)(a2 + 80);
              v134 = 32;
              v104 = v50;
              v135 = 0;
              v136 = 1;
              v128[0] = &v132;
              v128[3] = a1;
              v128[2] = (_BYTE *)a2;
              while ( a2 + 72 != v104 )
              {
                if ( !v104 )
                  BUG();
                for ( j = *(_QWORD *)(v104 + 32); v104 + 24 != j; j = *(_QWORD *)(j + 8) )
                {
                  if ( !j )
                    BUG();
                  sub_BDED30((__int64 *)v128, (_BYTE *)(j - 24), *(const char **)(j + 24));
                  v32 = 18;
                  v51 = sub_BDB7A0(j - 24, 18);
                  v52 = v51;
                  if ( v51 )
                  {
                    v53 = 1;
                    v118 = (_BYTE *)(v51 - 16);
                    while ( 1 )
                    {
                      v54 = (*(_BYTE *)(v52 - 16) & 2) != 0 ? *(_DWORD *)(v52 - 24) : (*(_WORD *)(v52 - 16) >> 6) & 0xF;
                      if ( v53 >= v54 )
                        break;
                      v55 = *(const char **)&sub_A17150(v118)[8 * v53];
                      if ( v55 && (unsigned __int8)(*v55 - 5) >= 0x20u )
                        v55 = 0;
                      v32 = j - 24;
                      ++v53;
                      sub_BDED30((__int64 *)v128, (_BYTE *)(j - 24), v55);
                    }
                  }
                  if ( *((_BYTE *)a1 + 153) )
                    goto LABEL_222;
                }
                v104 = *(_QWORD *)(v104 + 8);
              }
LABEL_222:
              if ( !v136 )
                _libc_free(v133, v32);
            }
            goto LABEL_172;
          }
          v138 = 1;
          v93 = "gc.get.pointer.base operand and result must be of the same type";
        }
        else
        {
          v138 = 1;
          v93 = "gc.get.pointer.base must return a pointer";
        }
LABEL_220:
        v32 = (__int64)&v132;
        v132 = v93;
        v137 = 3;
        sub_BEE660(a1, (__int64)&v132, (_BYTE *)a2);
        goto LABEL_172;
      }
    }
    else
    {
      if ( v46 != 148 )
        goto LABEL_77;
      v47 = *(_QWORD *)(a2 + 24);
      if ( *(_DWORD *)(v47 + 12) == 2 )
      {
        v48 = *(_QWORD **)(v47 + 16);
        if ( *(_BYTE *)(v48[1] + 8LL) == 14 )
        {
          if ( *(_BYTE *)(*v48 + 8LL) == 12 )
            goto LABEL_77;
          v138 = 1;
          v93 = "gc.get.pointer.offset must return integer";
        }
        else
        {
          v138 = 1;
          v93 = "gc.get.pointer.offset operand must be a pointer";
        }
        goto LABEL_220;
      }
    }
    v138 = 1;
    v93 = "wrong number of parameters";
    goto LABEL_220;
  }
  v78 = v129;
  v79 = &v129[2 * (unsigned int)v130];
  if ( v79 == v129 )
  {
LABEL_240:
    if ( (*(_BYTE *)(a2 + 2) & 8) != 0 )
    {
      v128[0] = (_BYTE *)a2;
      v96 = "Function declaration shouldn't have a personality routine";
      v138 = 1;
      goto LABEL_239;
    }
    goto LABEL_70;
  }
  while ( 2 )
  {
    if ( !*(_DWORD *)v78 )
    {
      v32 = (__int64)v78[1];
      if ( (*(_BYTE *)(v32 + 1) & 0x7F) == 1 )
      {
        v128[0] = (_BYTE *)a2;
        v95 = "function declaration may only have a unique !dbg attachment";
        v138 = 1;
        goto LABEL_234;
      }
      goto LABEL_185;
    }
    if ( *(_DWORD *)v78 != 2 )
    {
      v32 = (__int64)v78[1];
LABEL_185:
      v78 += 2;
      sub_BE3890((__int64)a1, v32, 1u);
      if ( v79 == v78 )
        goto LABEL_240;
      continue;
    }
    break;
  }
  v128[0] = (_BYTE *)a2;
  v96 = "function declaration may not have a !prof attachment";
  v138 = 1;
LABEL_239:
  v32 = (__int64)&v132;
  v132 = v96;
  v137 = 3;
  sub_BEDA40(a1, (__int64)&v132, v128);
LABEL_172:
  if ( v129 != (const char **)v131 )
    _libc_free(v129, v32);
}
