// Function: sub_165E480
// Address: 0x165e480
//
void __fastcall sub_165E480(__int64 *a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // r13
  __int64 v5; // rbx
  __int64 v6; // r14
  int v7; // ebx
  __int64 v8; // rsi
  unsigned __int16 v9; // ax
  __int64 v10; // r13
  _BYTE *v11; // rax
  __int64 v12; // rax
  __int64 v13; // r8
  _BYTE *v14; // rax
  const char *v15; // rax
  unsigned __int64 v16; // rdx
  unsigned __int8 *v17; // r14
  unsigned __int8 *v18; // rcx
  int v19; // esi
  unsigned __int8 *v20; // r12
  __int64 *v21; // rcx
  __int64 v22; // rbx
  __int64 v23; // r8
  int v24; // eax
  char v25; // al
  const char *v26; // rax
  const char *v27; // rax
  unsigned __int64 v28; // rdx
  _BOOL4 v29; // eax
  __int64 v30; // rsi
  _BYTE *v31; // r13
  __int64 v32; // rbx
  __int64 v33; // rax
  __int64 v34; // rbx
  char v35; // dl
  __int64 v36; // r15
  _QWORD *v37; // rax
  _QWORD *v38; // rsi
  _QWORD *v39; // rcx
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  unsigned __int8 *v44; // r13
  unsigned __int8 *v45; // rax
  __int64 v46; // r14
  unsigned __int8 *v47; // r8
  __int64 v48; // rdx
  __int64 v49; // rax
  _BYTE *v50; // rdi
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // r13
  _BYTE *v54; // rax
  __int64 v55; // rsi
  __int64 v56; // rdi
  _BYTE *v57; // rax
  _BYTE *v58; // rax
  const char *v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rbx
  __int64 v63; // r13
  __int64 v64; // r14
  __int64 v65; // r13
  _BYTE *v66; // rbx
  __int64 v67; // rsi
  __int64 *v68; // r12
  const char *v69; // rax
  __int64 v70; // r13
  __int64 v71; // rbx
  const char *v72; // rax
  __int64 v73; // r14
  int v74; // r13d
  __int64 v75; // rsi
  _BYTE *v76; // r8
  unsigned int v77; // esi
  int v78; // r9d
  _QWORD *v79; // rdi
  unsigned int i; // edx
  _QWORD *v81; // rax
  _BYTE *v82; // rcx
  __int64 v83; // rdx
  unsigned __int8 *v84; // rax
  __int64 v85; // r13
  _BYTE *v86; // rax
  bool v87; // zf
  const char *v88; // rax
  int v89; // ecx
  unsigned __int8 *v90; // rdx
  __int64 v91; // rax
  unsigned int v92; // edx
  const char *v93; // rax
  unsigned __int8 *v94; // [rsp+8h] [rbp-238h]
  __int64 v95; // [rsp+10h] [rbp-230h]
  __int64 *v96; // [rsp+20h] [rbp-220h]
  __int64 v97; // [rsp+20h] [rbp-220h]
  unsigned __int8 *v98; // [rsp+28h] [rbp-218h]
  __int64 v99; // [rsp+28h] [rbp-218h]
  int v100; // [rsp+28h] [rbp-218h]
  char v101; // [rsp+30h] [rbp-210h]
  __int64 v102; // [rsp+30h] [rbp-210h]
  _BYTE *v103; // [rsp+30h] [rbp-210h]
  const char *v104; // [rsp+48h] [rbp-1F8h] BYREF
  __int64 v105; // [rsp+50h] [rbp-1F0h] BYREF
  unsigned __int8 *v106; // [rsp+58h] [rbp-1E8h] BYREF
  unsigned __int8 *v107[2]; // [rsp+60h] [rbp-1E0h] BYREF
  char v108; // [rsp+70h] [rbp-1D0h]
  char v109; // [rsp+71h] [rbp-1CFh]
  char v110; // [rsp+80h] [rbp-1C0h]
  _BYTE *v111; // [rsp+90h] [rbp-1B0h] BYREF
  __int64 v112; // [rsp+98h] [rbp-1A8h]
  _BYTE v113[64]; // [rsp+A0h] [rbp-1A0h] BYREF
  const char *v114; // [rsp+E0h] [rbp-160h] BYREF
  _BYTE *v115; // [rsp+E8h] [rbp-158h]
  _BYTE *v116; // [rsp+F0h] [rbp-150h]
  __int64 v117; // [rsp+F8h] [rbp-148h]
  int v118; // [rsp+100h] [rbp-140h]
  _BYTE v119[312]; // [rsp+108h] [rbp-138h] BYREF

  v3 = (__int64)a1;
  sub_1651CA0(a1, a2);
  v4 = *(_QWORD *)(a2 + 24);
  v5 = *(_QWORD *)(a2 + 96);
  v6 = a1[8];
  if ( v6 != sub_15E0530(a2) )
  {
    v10 = *a1;
    v114 = "Function context does not match Module context!";
    LOWORD(v116) = 259;
    if ( v10 )
    {
      sub_16E2CE0(&v114, v10);
      v11 = *(_BYTE **)(v10 + 24);
      if ( (unsigned __int64)v11 >= *(_QWORD *)(v10 + 16) )
      {
        sub_16E7DE0(v10, 10);
      }
      else
      {
        *(_QWORD *)(v10 + 24) = v11 + 1;
        *v11 = 10;
      }
      v12 = *a1;
      *((_BYTE *)a1 + 72) = 1;
      if ( v12 )
        sub_164FA80(a1, a2);
    }
    else
    {
      *((_BYTE *)a1 + 72) = 1;
    }
    return;
  }
  if ( (*(_BYTE *)(a2 + 32) & 0xF) == 0xA )
  {
    v111 = (_BYTE *)a2;
    v26 = "Functions may not have common linkage";
    BYTE1(v116) = 1;
LABEL_50:
    v114 = v26;
    LOBYTE(v116) = 3;
    sub_165A6B0(a1, (__int64)&v114, (__int64 *)&v111);
    return;
  }
  if ( *(_DWORD *)(v4 + 12) - 1 != (_DWORD)v5 )
  {
    v114 = "# formal arguments must match # of arguments for function type!";
    LOWORD(v116) = 259;
    sub_164FF40(a1, (__int64)&v114);
    if ( *a1 )
    {
      sub_164FA80(a1, a2);
      v13 = *a1;
      v14 = *(_BYTE **)(*a1 + 24);
      if ( (unsigned __int64)v14 >= *(_QWORD *)(*a1 + 16) )
      {
        v13 = sub_16E7DE0(*a1, 32);
      }
      else
      {
        *(_QWORD *)(v13 + 24) = v14 + 1;
        *v14 = 32;
      }
      sub_154E060(v4, v13, 0, 0);
    }
    return;
  }
  if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL) + 8LL) == 12 )
  {
    v111 = (_BYTE *)a2;
    v26 = "Functions cannot return aggregate values!";
    BYTE1(v116) = 1;
    goto LABEL_50;
  }
  if ( ((unsigned __int8)sub_1560290((_QWORD *)(a2 + 112), 0, 53)
     || (unsigned __int8)sub_1560290((_QWORD *)(a2 + 112), 1, 53))
    && *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL) + 8LL) )
  {
    v111 = (_BYTE *)a2;
    v26 = "Invalid struct return type!";
    BYTE1(v116) = 1;
    goto LABEL_50;
  }
  v7 = *(_DWORD *)(v4 + 12);
  v104 = *(const char **)(a2 + 112);
  v114 = v104;
  if ( (unsigned int)sub_15601D0((__int64)&v114) > v7 + 1 )
  {
    v111 = (_BYTE *)a2;
    v15 = "Attribute after last parameter!";
    BYTE1(v116) = 1;
    goto LABEL_23;
  }
  sub_16595D0(a1, v4, (__int64)v104, (__int64 *)a2);
  v8 = 5;
  v101 = sub_1560180((__int64)&v104, 5);
  if ( v101 )
  {
    v111 = (_BYTE *)a2;
    v15 = "Attribute 'builtin' can only be applied to a callsite.";
    BYTE1(v116) = 1;
    goto LABEL_23;
  }
  v9 = (*(_WORD *)(a2 + 18) >> 4) & 0x3FF;
  if ( v9 > 9u )
  {
    switch ( v9 )
    {
      case 'G':
      case 'H':
      case 'M':
        goto LABEL_43;
      case 'L':
      case '[':
        if ( !*(_BYTE *)(**(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL) + 8LL) )
          goto LABEL_40;
        v111 = (_BYTE *)a2;
        v15 = "Calling convention requires void return type";
        BYTE1(v116) = 1;
        goto LABEL_23;
      case 'W':
      case 'X':
      case 'Y':
      case 'Z':
      case ']':
LABEL_40:
        if ( !(unsigned __int8)sub_1560290((_QWORD *)(a2 + 112), 0, 53) )
        {
          v8 = 1;
          if ( !(unsigned __int8)sub_1560290((_QWORD *)(a2 + 112), 1, 53) )
            goto LABEL_43;
        }
        v111 = (_BYTE *)a2;
        v15 = "Calling convention does not allow sret";
        BYTE1(v116) = 1;
        break;
      default:
        goto LABEL_25;
    }
    goto LABEL_23;
  }
  if ( ((*(_WORD *)(a2 + 18) >> 4) & 0x3F8) != 0 )
  {
LABEL_43:
    if ( !(*(_DWORD *)(*(_QWORD *)(a2 + 24) + 8LL) >> 8) )
      goto LABEL_25;
    v111 = (_BYTE *)a2;
    v15 = "Calling convention does not support varargs or perfect forwarding!";
    BYTE1(v116) = 1;
LABEL_23:
    v114 = v15;
    LOBYTE(v116) = 3;
    sub_165A6B0((_BYTE *)v3, (__int64)&v114, (__int64 *)&v111);
    return;
  }
LABEL_25:
  sub_1649960(a2);
  if ( v16 > 4 )
  {
    v27 = sub_1649960(a2);
    if ( v28 > 4 )
    {
      v29 = *(_DWORD *)v27 != 1836477548 || v27[4] != 46;
      v101 = !v29;
    }
  }
  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
  {
    sub_15E08E0(a2, v8);
    v17 = *(unsigned __int8 **)(a2 + 88);
    v18 = &v17[40 * *(_QWORD *)(a2 + 96)];
    if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
    {
      v98 = &v17[40 * *(_QWORD *)(a2 + 96)];
      sub_15E08E0(a2, v8);
      v17 = *(unsigned __int8 **)(a2 + 88);
      v18 = v98;
    }
  }
  else
  {
    v17 = *(unsigned __int8 **)(a2 + 88);
    v18 = &v17[40 * *(_QWORD *)(a2 + 96)];
  }
  if ( v18 != v17 )
  {
    v19 = 0;
    v20 = v18;
    v21 = (__int64 *)&v104;
    while ( 1 )
    {
      v22 = (unsigned int)(v19 + 1);
      v23 = *(_QWORD *)(*(_QWORD *)(v4 + 16) + 8 * v22);
      if ( v23 != *(_QWORD *)v17 )
      {
        v102 = *(_QWORD *)(*(_QWORD *)(v4 + 16) + 8 * v22);
        v114 = "Argument value does not match function argument type!";
        LOWORD(v116) = 259;
        sub_164FF40(a1, (__int64)&v114);
        if ( *a1 )
        {
          sub_164FA80(a1, (__int64)v17);
          if ( v102 )
            sub_164ECF0(*a1, v102);
        }
        return;
      }
      v24 = *(unsigned __int8 *)(v23 + 8);
      if ( !*(_BYTE *)(v23 + 8) || v24 == 12 )
      {
        v114 = "Function arguments must have first-class types!";
        LOWORD(v116) = 259;
        sub_164FF40(a1, (__int64)&v114);
        if ( *a1 )
          sub_164FA80(a1, (__int64)v17);
        return;
      }
      if ( !v101 )
      {
        if ( (_BYTE)v24 == 8 )
        {
          v68 = a1;
          v107[0] = v17;
          v69 = "Function takes metadata but isn't an intrinsic";
          BYTE1(v116) = 1;
          v111 = (_BYTE *)a2;
          goto LABEL_162;
        }
        if ( (_BYTE)v24 == 10 )
        {
          v68 = a1;
          v107[0] = v17;
          v69 = "Function takes token but isn't an intrinsic";
          BYTE1(v116) = 1;
          v111 = (_BYTE *)a2;
LABEL_162:
          v114 = v69;
          LOBYTE(v116) = 3;
          sub_165A7A0(v68, (__int64)&v114, (__int64 *)v107, (__int64 *)&v111);
          return;
        }
      }
      v96 = v21;
      v25 = sub_1560290(v21, v19, 54);
      v21 = v96;
      if ( v25 )
      {
        sub_165B9A0((__int64)a1, (__int64)v17);
        v21 = v96;
      }
      v17 += 40;
      if ( v20 == v17 )
      {
        v3 = (__int64)a1;
        break;
      }
      ++v19;
    }
  }
  if ( !v101 && *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL) + 8LL) == 10 )
  {
    v111 = (_BYTE *)a2;
    v15 = "Functions returns a token but isn't an intrinsic";
    BYTE1(v116) = 1;
    goto LABEL_23;
  }
  v111 = v113;
  v112 = 0x400000000LL;
  sub_1626D60(a2, (__int64)&v111);
  v30 = 16LL * (unsigned int)v112;
  v31 = &v111[v30];
  if ( v111 == &v111[v30] )
    goto LABEL_73;
  v32 = (__int64)v111;
  while ( *(_DWORD *)v32 != 2 )
  {
LABEL_72:
    v32 += 16;
    if ( v31 == (_BYTE *)v32 )
      goto LABEL_73;
  }
  v48 = *(_QWORD *)(v32 + 8);
  v107[0] = (unsigned __int8 *)v48;
  v49 = *(unsigned int *)(v48 + 8);
  if ( (unsigned int)v49 <= 1 )
  {
    BYTE1(v116) = 1;
    v59 = "!prof annotations should have no less than 2 operands";
    goto LABEL_143;
  }
  v50 = *(_BYTE **)(v48 - 8 * v49);
  if ( !v50 )
  {
    BYTE1(v116) = 1;
    v59 = "first operand should not be null";
    goto LABEL_143;
  }
  if ( *v50 )
  {
    BYTE1(v116) = 1;
    v59 = "expected string with name of the !prof annotation";
    goto LABEL_143;
  }
  v51 = sub_161E970((__int64)v50);
  if ( v52 != 20 )
  {
    if ( v52 != 30
      || *(_QWORD *)v51 ^ 0x69746568746E7973LL | *(_QWORD *)(v51 + 8) ^ 0x6974636E75665F63LL
      || *(_QWORD *)(v51 + 16) != 0x7972746E655F6E6FLL
      || *(_DWORD *)(v51 + 24) != 1970234207
      || *(_WORD *)(v51 + 28) != 29806 )
    {
      goto LABEL_125;
    }
LABEL_140:
    v58 = *(_BYTE **)&v107[0][8 * (1LL - *((unsigned int *)v107[0] + 2))];
    if ( v58 )
    {
      if ( *v58 == 1 )
        goto LABEL_72;
      BYTE1(v116) = 1;
      v59 = "expected integer argument to function_entry_count";
    }
    else
    {
      BYTE1(v116) = 1;
      v59 = "second operand should not be null";
    }
LABEL_143:
    v114 = v59;
    LOBYTE(v116) = 3;
    sub_1659E40((_BYTE *)v3, (__int64)&v114, v107);
    goto LABEL_73;
  }
  if ( !(*(_QWORD *)v51 ^ 0x6E6F6974636E7566LL | *(_QWORD *)(v51 + 8) ^ 0x635F7972746E655FLL)
    && *(_DWORD *)(v51 + 16) == 1953396079 )
  {
    goto LABEL_140;
  }
LABEL_125:
  v53 = *(_QWORD *)v3;
  v114 = "first operand should be 'function_entry_count' or 'synthetic_function_entry_count'";
  LOWORD(v116) = 259;
  if ( v53 )
  {
    sub_16E2CE0(&v114, v53);
    v54 = *(_BYTE **)(v53 + 24);
    if ( (unsigned __int64)v54 >= *(_QWORD *)(v53 + 16) )
    {
      sub_16E7DE0(v53, 10);
    }
    else
    {
      *(_QWORD *)(v53 + 24) = v54 + 1;
      *v54 = 10;
    }
    v55 = *(_QWORD *)v3;
    *(_BYTE *)(v3 + 72) = 1;
    if ( v55 && v107[0] )
    {
      sub_15562E0(v107[0], v55, v3 + 16, *(_QWORD *)(v3 + 8));
      v56 = *(_QWORD *)v3;
      v57 = *(_BYTE **)(*(_QWORD *)v3 + 24LL);
      if ( (unsigned __int64)v57 >= *(_QWORD *)(*(_QWORD *)v3 + 16LL) )
      {
        sub_16E7DE0(v56, 10);
      }
      else
      {
        *(_QWORD *)(v56 + 24) = v57 + 1;
        *v57 = 10;
      }
    }
  }
  else
  {
    *(_BYTE *)(v3 + 72) = 1;
  }
LABEL_73:
  if ( (*(_BYTE *)(a2 + 18) & 8) != 0 )
  {
    v60 = sub_15E38F0(a2);
    v61 = sub_1649C60(v60);
    v62 = v61;
    if ( !*(_BYTE *)(v61 + 16) )
    {
      v63 = *(_QWORD *)(v61 + 40);
      v64 = *(_QWORD *)(a2 + 40);
      if ( v64 != v63 )
      {
        v114 = "Referencing personality function in another module!";
        LOWORD(v116) = 259;
        sub_164FF40((__int64 *)v3, (__int64)&v114);
        if ( *(_QWORD *)v3 )
        {
          sub_164FA80((__int64 *)v3, a2);
          sub_164EDD0(*(_QWORD *)v3, v64);
          sub_164FA80((__int64 *)v3, v62);
          sub_164EDD0(*(_QWORD *)v3, v63);
        }
        goto LABEL_115;
      }
    }
  }
  if ( (*(_BYTE *)(a2 + 34) & 0x40) != 0 )
  {
    if ( (_DWORD)v112 )
    {
      v84 = (unsigned __int8 *)*((_QWORD *)v111 + 1);
      v106 = (unsigned __int8 *)a2;
      v107[0] = v84;
      v114 = "unmaterialized function cannot have metadata";
      LOWORD(v116) = 259;
      sub_165A900((__int64 *)v3, (__int64)&v114, (__int64 *)&v106, v107);
      goto LABEL_115;
    }
    goto LABEL_76;
  }
  if ( sub_15E4F60(a2) )
  {
    v65 = (__int64)v111;
    v66 = &v111[16 * (unsigned int)v112];
    if ( v111 == v66 )
    {
LABEL_198:
      if ( (*(_BYTE *)(a2 + 18) & 8) == 0 )
        goto LABEL_76;
      v107[0] = (unsigned __int8 *)a2;
      v88 = "Function declaration shouldn't have a personality routine";
      BYTE1(v116) = 1;
    }
    else
    {
      while ( 1 )
      {
        if ( !*(_DWORD *)v65 )
        {
          v114 = "function declaration may not have a !dbg attachment";
          LOWORD(v116) = 259;
          sub_16521E0((__int64 *)v3, (__int64)&v114);
          if ( *(_QWORD *)v3 )
            sub_164FA80((__int64 *)v3, a2);
          goto LABEL_115;
        }
        if ( *(_DWORD *)v65 == 2 )
          break;
        v67 = *(_QWORD *)(v65 + 8);
        v65 += 16;
        sub_1656110((_QWORD *)v3, v67);
        if ( v66 == (_BYTE *)v65 )
          goto LABEL_198;
      }
      v107[0] = (unsigned __int8 *)a2;
      v88 = "function declaration may not have a !prof attachment";
      BYTE1(v116) = 1;
    }
LABEL_200:
    v114 = v88;
    LOBYTE(v116) = 3;
    sub_165A6B0((_BYTE *)v3, (__int64)&v114, (__int64 *)v107);
    goto LABEL_115;
  }
  if ( v101 )
  {
    v107[0] = (unsigned __int8 *)a2;
    v88 = "llvm intrinsics cannot be defined!";
    BYTE1(v116) = 1;
    goto LABEL_200;
  }
  v70 = *(_QWORD *)(a2 + 80);
  if ( !v70 )
  {
    v105 = 0;
    BUG();
  }
  v105 = v70 - 24;
  v71 = *(_QWORD *)(v70 - 16);
  if ( v71 )
  {
    while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v71) + 16) - 25) > 9u )
    {
      v71 = *(_QWORD *)(v71 + 8);
      if ( !v71 )
        goto LABEL_174;
    }
    BYTE1(v116) = 1;
    v72 = "Entry block to function must not have predecessors!";
LABEL_172:
    v114 = v72;
    LOBYTE(v116) = 3;
    sub_165AA30((_BYTE *)v3, (__int64)&v114, &v105);
    goto LABEL_115;
  }
LABEL_174:
  if ( *(_WORD *)(v70 - 6) )
  {
    v91 = sub_1594E40(v70 - 24);
    if ( (unsigned __int8)sub_1593E70(v91) )
    {
      BYTE1(v116) = 1;
      v72 = "blockaddress may not be used with the entry block!";
      goto LABEL_172;
    }
  }
  v73 = (__int64)v111;
  v103 = &v111[16 * (unsigned int)v112];
  if ( v111 != v103 )
  {
    v100 = 0;
    v74 = 0;
    do
    {
      if ( *(_DWORD *)v73 )
      {
        if ( *(_DWORD *)v73 == 2 )
        {
          if ( v100 )
          {
            v107[0] = (unsigned __int8 *)a2;
            v114 = "function must have a single !prof attachment";
            LOWORD(v116) = 259;
            sub_165A900((__int64 *)v3, (__int64)&v114, (__int64 *)v107, (unsigned __int8 **)(v73 + 8));
            goto LABEL_115;
          }
          v100 = 1;
        }
      }
      else
      {
        if ( v74 )
        {
          v107[0] = (unsigned __int8 *)a2;
          v93 = "function must have a single !dbg attachment";
          BYTE1(v116) = 1;
          goto LABEL_228;
        }
        v76 = *(_BYTE **)(v73 + 8);
        if ( *v76 != 17 )
        {
          v107[0] = (unsigned __int8 *)a2;
          v93 = "function !dbg attachment must be a subprogram";
          BYTE1(v116) = 1;
LABEL_228:
          v114 = v93;
          LOBYTE(v116) = 3;
          sub_165AB20((__int64 *)v3, (__int64)&v114, (__int64 *)v107, (unsigned __int8 **)(v73 + 8));
          goto LABEL_115;
        }
        v77 = *(_DWORD *)(v3 + 648);
        v106 = *(unsigned __int8 **)(v73 + 8);
        if ( v77 )
        {
          v78 = 1;
          v79 = 0;
          for ( i = (v77 - 1) & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4)); ; i = (v77 - 1) & v92 )
          {
            v81 = (_QWORD *)(*(_QWORD *)(v3 + 632) + 16LL * i);
            v82 = (_BYTE *)*v81;
            if ( v76 == (_BYTE *)*v81 )
            {
              v83 = v81[1];
              if ( v83 && v83 != a2 )
              {
                v107[0] = (unsigned __int8 *)a2;
                v114 = "DISubprogram attached to more than one function";
                LOWORD(v116) = 259;
                sub_165AC70(v3, (__int64)&v114, &v106, (__int64 *)v107);
                goto LABEL_115;
              }
              goto LABEL_189;
            }
            if ( v82 == (_BYTE *)-8LL )
              break;
            if ( v79 || v82 != (_BYTE *)-16LL )
              v81 = v79;
            v92 = v78 + i;
            v79 = v81;
            ++v78;
          }
          v89 = *(_DWORD *)(v3 + 640);
          if ( v79 )
            v81 = v79;
          ++*(_QWORD *)(v3 + 624);
          if ( 4 * (v89 + 1) >= 3 * v77 )
            goto LABEL_213;
          if ( v77 - *(_DWORD *)(v3 + 644) - (v89 + 1) <= v77 >> 3 )
            goto LABEL_214;
        }
        else
        {
          ++*(_QWORD *)(v3 + 624);
LABEL_213:
          v77 *= 2;
LABEL_214:
          sub_165E2C0(v3 + 624, v77);
          sub_165C730(v3 + 624, (__int64 *)&v106, &v114);
          v81 = v114;
        }
        ++*(_DWORD *)(v3 + 640);
        if ( *v81 != -8 )
          --*(_DWORD *)(v3 + 644);
        v90 = v106;
        v81[1] = 0;
        *v81 = v90;
LABEL_189:
        v81[1] = a2;
        v74 = 1;
      }
      v75 = *(_QWORD *)(v73 + 8);
      v73 += 16;
      sub_1656110((_QWORD *)v3, v75);
    }
    while ( v103 != (_BYTE *)v73 );
  }
LABEL_76:
  if ( *(_DWORD *)(a2 + 36)
    && !*(_QWORD *)(*(_QWORD *)(a2 + 40) + 168LL)
    && (unsigned __int8)sub_15E3650(a2, (__int64 *)v107) )
  {
    v85 = *(_QWORD *)v3;
    v114 = "Invalid user of intrinsic instruction!";
    LOWORD(v116) = 259;
    if ( v85 )
    {
      sub_16E2CE0(&v114, v85);
      v86 = *(_BYTE **)(v85 + 24);
      if ( (unsigned __int64)v86 >= *(_QWORD *)(v85 + 16) )
      {
        sub_16E7DE0(v85, 10);
      }
      else
      {
        *(_QWORD *)(v85 + 24) = v86 + 1;
        *v86 = 10;
      }
    }
    v87 = *(_QWORD *)v3 == 0;
    *(_BYTE *)(v3 + 72) = 1;
    if ( !v87 )
      sub_164FA80((__int64 *)v3, (__int64)v107[0]);
    goto LABEL_115;
  }
  v94 = (unsigned __int8 *)sub_1626D20(a2);
  *(_BYTE *)(v3 + 721) = v94 != 0;
  if ( !v94 )
    goto LABEL_115;
  v114 = 0;
  v115 = v119;
  v116 = v119;
  v33 = *(_QWORD *)(a2 + 80);
  v117 = 32;
  v118 = 0;
  v97 = v33;
  v95 = a2 + 72;
  if ( a2 + 72 == v33 )
    goto LABEL_115;
  v99 = a2;
  while ( 1 )
  {
    if ( !v97 )
      BUG();
    v34 = *(_QWORD *)(v97 + 24);
    if ( v97 + 16 != v34 )
      break;
LABEL_118:
    v97 = *(_QWORD *)(v97 + 8);
    if ( v95 == v97 )
      goto LABEL_113;
  }
  while ( 2 )
  {
    if ( !v34 )
      BUG();
    v36 = *(_QWORD *)(v34 + 24);
    if ( !v36 || *(_BYTE *)v36 != 5 )
    {
LABEL_85:
      v34 = *(_QWORD *)(v34 + 8);
      if ( v97 + 16 == v34 )
        goto LABEL_118;
      continue;
    }
    break;
  }
  v37 = v115;
  if ( v116 != v115 )
    goto LABEL_84;
  v38 = &v115[8 * HIDWORD(v117)];
  if ( v115 != (_BYTE *)v38 )
  {
    v39 = 0;
    while ( v36 != *v37 )
    {
      if ( *v37 == -2 )
        v39 = v37;
      if ( v38 == ++v37 )
      {
        if ( !v39 )
          goto LABEL_132;
        *v39 = v36;
        --v118;
        ++v114;
        goto LABEL_98;
      }
    }
    goto LABEL_85;
  }
LABEL_132:
  if ( HIDWORD(v117) < (unsigned int)v117 )
  {
    ++HIDWORD(v117);
    *v38 = v36;
    ++v114;
  }
  else
  {
LABEL_84:
    sub_16CCBA0(&v114, *(_QWORD *)(v34 + 24));
    if ( !v35 )
      goto LABEL_85;
  }
LABEL_98:
  v40 = v36;
  v41 = *(unsigned int *)(v36 + 8);
  if ( (_DWORD)v41 == 2 )
  {
    while ( 1 )
    {
      v42 = *(_QWORD *)(v40 - 8);
      if ( !v42 )
        break;
      v40 = *(_QWORD *)(v40 - 8);
      v41 = *(unsigned int *)(v42 + 8);
      if ( (_DWORD)v41 != 2 )
        goto LABEL_101;
    }
    v43 = -16;
  }
  else
  {
LABEL_101:
    v43 = -8 * v41;
  }
  v44 = *(unsigned __int8 **)(v40 + v43);
  if ( v44 )
  {
    sub_165ADB0((__int64)v107, (__int64)&v114, *(_QWORD *)(v40 + v43));
    if ( !v110 )
      goto LABEL_85;
    v45 = sub_15B1000(v44);
    v46 = (__int64)v45;
    if ( v45 )
    {
      if ( v45 != v44 )
      {
        sub_165ADB0((__int64)v107, (__int64)&v114, (__int64)v45);
        if ( !v110 )
          goto LABEL_85;
      }
    }
  }
  else
  {
    v46 = 0;
  }
  if ( sub_15B1050(v46, v99) )
    goto LABEL_85;
  v109 = 1;
  v107[0] = "!dbg attachment points at wrong subprogram for function";
  v108 = 3;
  sub_16521E0((__int64 *)v3, (__int64)v107);
  if ( *(_QWORD *)v3 )
  {
    sub_164ED40((__int64 *)v3, v94);
    sub_164FA80((__int64 *)v3, v99);
    sub_164FA80((__int64 *)v3, v34 - 24);
    sub_164ED40((__int64 *)v3, (unsigned __int8 *)v36);
    v47 = (unsigned __int8 *)v46;
    if ( v44 )
    {
      sub_164ED40((__int64 *)v3, v44);
      v47 = (unsigned __int8 *)v46;
    }
    if ( v47 )
      sub_164ED40((__int64 *)v3, v47);
  }
LABEL_113:
  if ( v116 != v115 )
    _libc_free((unsigned __int64)v116);
LABEL_115:
  if ( v111 != v113 )
    _libc_free((unsigned __int64)v111);
}
