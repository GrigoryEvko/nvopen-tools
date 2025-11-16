// Function: sub_1663F80
// Address: 0x1663f80
//
void __fastcall sub_1663F80(__int64 a1, __int64 a2)
{
  __int64 v4; // rbx
  __int64 i; // r15
  __int64 v6; // rax
  __int64 v7; // rcx
  unsigned int v8; // edi
  __int64 *v9; // rdx
  __int64 v10; // rsi
  const char *v11; // rax
  char v12; // al
  __int64 *j; // r14
  int v14; // r11d
  __int64 k; // r9
  char v16; // si
  __int64 v17; // rdx
  __int64 v18; // r15
  __int64 v19; // rcx
  __int64 v20; // r10
  char v21; // di
  bool v22; // al
  const char *v23; // rax
  _QWORD *v24; // rax
  unsigned __int8 v25; // dl
  char v26; // cl
  int v27; // edx
  int v28; // esi
  __int64 v29; // r10
  __int64 v30; // rax
  char v31; // dl
  __int64 v32; // rax
  __int64 v33; // rbx
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r15
  __int64 v37; // rbx
  unsigned __int8 v38; // dl
  int v39; // edx
  int v40; // r9d
  __int64 v41; // r8
  __int64 v42; // r15
  __int64 v43; // r8
  __int64 v44; // r15
  __int64 v45; // rax
  int v46; // ecx
  unsigned __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // rbx
  __int64 v50; // rax
  __int64 v51; // rax
  char v52; // r13
  unsigned int v53; // eax
  __int64 v54; // rcx
  __int64 v55; // rsi
  const char *v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rdx
  unsigned __int8 *v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  unsigned int v62; // r9d
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rbx
  unsigned __int64 v66; // rax
  unsigned __int8 *v67; // r15
  _BOOL4 v68; // eax
  __int64 v69; // rdx
  char v70; // al
  __int64 v71; // r8
  _BYTE *v72; // rax
  bool v73; // zf
  const char *v74; // rax
  __int64 v75; // rbx
  _BYTE *v76; // rax
  __int64 v77; // rdi
  _BYTE *v78; // rax
  unsigned __int8 *v79; // rax
  __int64 v80; // rdx
  __int64 v81; // rdx
  __int64 v82; // rax
  __int64 v83; // rax
  char v84; // r15
  __int64 v85; // rdi
  const char *v86; // rax
  __int64 v87; // rax
  int v88; // eax
  __int64 v89; // rdx
  unsigned __int8 *v90; // r15
  __int64 v91; // rax
  __int64 v92; // rbx
  unsigned int v93; // ebx
  __int64 v94; // rbx
  _BYTE *v95; // rax
  const char *v96; // rax
  __int64 v97; // [rsp+8h] [rbp-128h]
  __int64 v98; // [rsp+18h] [rbp-118h]
  __int64 v99; // [rsp+18h] [rbp-118h]
  __int64 v100; // [rsp+20h] [rbp-110h]
  unsigned int v101; // [rsp+20h] [rbp-110h]
  int v102; // [rsp+20h] [rbp-110h]
  __int64 v103; // [rsp+20h] [rbp-110h]
  int v104; // [rsp+20h] [rbp-110h]
  __int64 v105; // [rsp+20h] [rbp-110h]
  __int64 v106; // [rsp+28h] [rbp-108h]
  int v107; // [rsp+28h] [rbp-108h]
  __int64 v108; // [rsp+28h] [rbp-108h]
  __int64 v109; // [rsp+28h] [rbp-108h]
  int v110; // [rsp+28h] [rbp-108h]
  __int64 v111; // [rsp+28h] [rbp-108h]
  int v112; // [rsp+28h] [rbp-108h]
  __int64 v113; // [rsp+30h] [rbp-100h]
  char v114; // [rsp+38h] [rbp-F8h]
  __int64 v115; // [rsp+38h] [rbp-F8h]
  __int64 v116; // [rsp+38h] [rbp-F8h]
  __int64 v117; // [rsp+38h] [rbp-F8h]
  __int64 v118; // [rsp+38h] [rbp-F8h]
  _BYTE *v119; // [rsp+38h] [rbp-F8h]
  int v120; // [rsp+38h] [rbp-F8h]
  unsigned __int8 *v121; // [rsp+48h] [rbp-E8h] BYREF
  const char *v122; // [rsp+50h] [rbp-E0h] BYREF
  unsigned int v123; // [rsp+58h] [rbp-D8h]
  __int64 v124; // [rsp+60h] [rbp-D0h] BYREF
  unsigned int v125; // [rsp+68h] [rbp-C8h]
  const char *v126; // [rsp+70h] [rbp-C0h] BYREF
  unsigned int v127; // [rsp+78h] [rbp-B8h]
  __int64 v128; // [rsp+80h] [rbp-B0h] BYREF
  unsigned int v129; // [rsp+88h] [rbp-A8h]
  const char *v130; // [rsp+90h] [rbp-A0h] BYREF
  unsigned int v131; // [rsp+98h] [rbp-98h]
  __int64 v132; // [rsp+A0h] [rbp-90h]
  unsigned int v133; // [rsp+A8h] [rbp-88h]
  __int64 v134; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v135; // [rsp+B8h] [rbp-78h]
  __int64 v136; // [rsp+C0h] [rbp-70h]
  unsigned int v137; // [rsp+C8h] [rbp-68h]
  const char *v138; // [rsp+D0h] [rbp-60h] BYREF
  unsigned int v139; // [rsp+D8h] [rbp-58h]
  __int64 v140; // [rsp+E0h] [rbp-50h]
  unsigned int v141; // [rsp+E8h] [rbp-48h]

  v4 = *(_QWORD *)(a2 + 40);
  if ( !v4 )
  {
    v134 = a2;
    v11 = "Instruction not embedded in basic block!";
    BYTE1(v140) = 1;
    goto LABEL_12;
  }
  v114 = *(_BYTE *)(a2 + 16);
  if ( v114 != 77 )
  {
    for ( i = *(_QWORD *)(a2 + 8); i; i = *(_QWORD *)(i + 8) )
    {
      if ( (_QWORD *)a2 == sub_1648700(i) )
      {
        v6 = *(unsigned int *)(a1 + 128);
        if ( (_DWORD)v6 )
        {
          v7 = *(_QWORD *)(a1 + 112);
          v8 = (v6 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
          v9 = (__int64 *)(v7 + 16LL * v8);
          v10 = *v9;
          if ( v4 == *v9 )
          {
LABEL_9:
            if ( v9 != (__int64 *)(v7 + 16 * v6) && v9[1] )
            {
              v134 = a2;
              v11 = "Only PHI nodes may reference their own value!";
              BYTE1(v140) = 1;
LABEL_12:
              v138 = v11;
              LOBYTE(v140) = 3;
              sub_1654980((_BYTE *)a1, (__int64)&v138, &v134);
              return;
            }
          }
          else
          {
            v39 = 1;
            while ( v10 != -8 )
            {
              v40 = v39 + 1;
              v8 = (v6 - 1) & (v39 + v8);
              v9 = (__int64 *)(v7 + 16LL * v8);
              v10 = *v9;
              if ( v4 == *v9 )
                goto LABEL_9;
              v39 = v40;
            }
          }
        }
      }
    }
  }
  v12 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
  if ( v12 )
  {
    if ( v12 == 12 )
    {
      v134 = a2;
      v11 = "Instruction returns a non-scalar type!";
      BYTE1(v140) = 1;
      goto LABEL_12;
    }
    if ( v114 != 78 && v114 != 29 && v12 == 8 )
    {
      v134 = a2;
      v11 = "Invalid use of metadata!";
      BYTE1(v140) = 1;
      goto LABEL_12;
    }
  }
  else if ( (*(_BYTE *)(a2 + 23) & 0x20) != 0 )
  {
    v134 = a2;
    v11 = "Instruction has a name, but provides a void value!";
    BYTE1(v140) = 1;
    goto LABEL_12;
  }
  for ( j = *(__int64 **)(a2 + 8); j; j = (__int64 *)j[1] )
  {
    v24 = sub_1648700((__int64)j);
    if ( *((_BYTE *)v24 + 16) <= 0x17u )
    {
      v138 = "Use of instruction is not an instruction!";
      LOWORD(v140) = 259;
      sub_164FF40((__int64 *)a1, (__int64)&v138);
      if ( *(_QWORD *)a1 )
        sub_164FA80((__int64 *)a1, *j);
      return;
    }
    if ( !v24[5] )
    {
      v117 = (__int64)v24;
      v138 = "Instruction referencing instruction not embedded in a basic block!";
      LOWORD(v140) = 259;
      sub_164FF40((__int64 *)a1, (__int64)&v138);
      if ( *(_QWORD *)a1 )
      {
        sub_164FA80((__int64 *)a1, a2);
        sub_164FA80((__int64 *)a1, v117);
      }
      return;
    }
  }
  v14 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( v14 )
  {
    for ( k = 0; v14 != (_DWORD)k; ++k )
    {
      v16 = *(_BYTE *)(a2 + 23);
      if ( (v16 & 0x40) != 0 )
        v17 = *(_QWORD *)(a2 - 8);
      else
        v17 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v18 = *(_QWORD *)(v17 + 24 * k);
      v19 = 24 * k;
      if ( !v18 )
      {
        v134 = a2;
        v23 = "Instruction has null operand!";
        BYTE1(v140) = 1;
        goto LABEL_23;
      }
      v20 = *(_QWORD *)v18;
      v21 = *(_BYTE *)(*(_QWORD *)v18 + 8LL);
      v22 = v21 != 12 && v21 != 0;
      if ( !v22 )
      {
        v134 = a2;
        v23 = "Instruction operands must be first-class values!";
        BYTE1(v140) = 1;
        goto LABEL_23;
      }
      v25 = *(_BYTE *)(v18 + 16);
      if ( v25 )
      {
        if ( v25 == 18 )
        {
          if ( *(_QWORD *)(v4 + 56) != *(_QWORD *)(v18 + 56) )
          {
            v134 = a2;
            v23 = "Referring to a basic block in another function!";
            BYTE1(v140) = 1;
            goto LABEL_23;
          }
          continue;
        }
        if ( v25 == 17 )
        {
          if ( *(_QWORD *)(v4 + 56) != *(_QWORD *)(v18 + 24) )
          {
            v134 = a2;
            v23 = "Referring to an argument in another function!";
            BYTE1(v140) = 1;
            goto LABEL_23;
          }
          continue;
        }
        if ( (unsigned __int8)(v25 - 1) <= 2u )
        {
          v106 = *(_QWORD *)(a1 + 8);
          if ( v106 != *(_QWORD *)(v18 + 40) )
          {
            v41 = v18;
            v42 = *(_QWORD *)(v18 + 40);
            v116 = v41;
            v138 = "Referencing global in another module!";
            LOWORD(v140) = 259;
            sub_164FF40((__int64 *)a1, (__int64)&v138);
            if ( *(_QWORD *)a1 )
            {
              sub_164FA80((__int64 *)a1, a2);
              sub_164EDD0(*(_QWORD *)a1, v106);
              sub_164FA80((__int64 *)a1, v116);
              sub_164EDD0(*(_QWORD *)a1, v42);
            }
            return;
          }
          continue;
        }
        if ( v25 > 0x17u )
        {
          if ( v25 != 29 || *(_QWORD *)(v18 - 48) != *(_QWORD *)(v18 - 24) )
          {
            if ( *(_BYTE *)(a2 + 16) == 77 )
              goto LABEL_219;
            v98 = k;
            v102 = v14;
            v109 = 24 * k;
            v68 = sub_13A0E30(a1 + 160, v18);
            v14 = v102;
            k = v98;
            if ( !v68 )
            {
              v16 = *(_BYTE *)(a2 + 23);
              v19 = v109;
LABEL_219:
              if ( (v16 & 0x40) != 0 )
                v69 = *(_QWORD *)(a2 - 8);
              else
                v69 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
              v103 = k;
              v110 = v14;
              v70 = sub_15CD0F0(a1 + 80, v18, v19 + v69);
              v14 = v110;
              k = v103;
              if ( !v70 )
              {
                v71 = *(_QWORD *)a1;
                v138 = "Instruction does not dominate all uses!";
                LOWORD(v140) = 259;
                if ( v71 )
                {
                  v99 = v103;
                  v104 = v110;
                  v111 = v71;
                  sub_16E2CE0(&v138, v71);
                  v14 = v104;
                  k = v99;
                  v72 = *(_BYTE **)(v111 + 24);
                  if ( (unsigned __int64)v72 >= *(_QWORD *)(v111 + 16) )
                  {
                    sub_16E7DE0(v111, 10);
                    v14 = v104;
                    k = v99;
                  }
                  else
                  {
                    *(_QWORD *)(v111 + 24) = v72 + 1;
                    *v72 = 10;
                  }
                }
                v73 = *(_QWORD *)a1 == 0;
                *(_BYTE *)(a1 + 72) = 1;
                if ( !v73 )
                {
                  v105 = k;
                  v112 = v14;
                  sub_164FA80((__int64 *)a1, v18);
                  sub_164FA80((__int64 *)a1, a2);
                  v14 = v112;
                  k = v105;
                }
              }
            }
          }
        }
        else if ( v25 == 20 )
        {
          if ( v14 == (_DWORD)k + 1 && *(_BYTE *)(a2 + 16) == 78 )
            break;
          if ( v14 != (_DWORD)k + 3 || *(_BYTE *)(a2 + 16) != 29 )
          {
            v134 = a2;
            v23 = "Cannot take the address of an inline asm!";
            BYTE1(v140) = 1;
            goto LABEL_23;
          }
        }
        else if ( v25 == 5 )
        {
          if ( v21 == 16 )
            v20 = **(_QWORD **)(v20 + 16);
          if ( *(_BYTE *)(v20 + 8) == 15 || *(_DWORD *)(*(_QWORD *)(a1 + 56) + 416LL) )
          {
            v100 = k;
            v107 = v14;
            sub_16501E0(a1, v18);
            k = v100;
            v14 = v107;
          }
        }
      }
      else
      {
        if ( (*(_BYTE *)(v18 + 33) & 0x20) != 0 )
        {
          v26 = *(_BYTE *)(a2 + 16);
          if ( v26 == 78 )
          {
            if ( v14 - 1 != (_DWORD)k )
              goto LABEL_77;
          }
          else
          {
            v27 = v14 - 3;
            if ( v26 != 29 )
              v27 = 0;
            if ( v27 != (_DWORD)k )
            {
LABEL_77:
              v134 = a2;
              v23 = "Cannot take the address of an intrinsic!";
              BYTE1(v140) = 1;
              goto LABEL_23;
            }
            v28 = *(_DWORD *)(v18 + 36);
            if ( (unsigned int)(v28 - 16) <= 0x3E )
              v22 = ((0x4000000001000201uLL >> ((unsigned __int8)v28 - 16)) & 1) == 0;
            if ( (unsigned int)(v28 - 80) > 1 && v22 )
            {
              v134 = a2;
              v23 = "Cannot invoke an intrinsic other than donothing, patchpoint, statepoint, coro_resume or coro_destroy";
              BYTE1(v140) = 1;
              goto LABEL_23;
            }
          }
        }
        v29 = *(_QWORD *)(v18 + 40);
        if ( *(_QWORD *)(a1 + 8) != v29 )
        {
          v43 = v18;
          v44 = *(_QWORD *)(a1 + 8);
          v113 = v29;
          v118 = v43;
          v138 = "Referencing function in another module!";
          LOWORD(v140) = 259;
          sub_164FF40((__int64 *)a1, (__int64)&v138);
          if ( *(_QWORD *)a1 )
          {
            sub_164FA80((__int64 *)a1, a2);
            sub_164EDD0(*(_QWORD *)a1, v44);
            sub_164FA80((__int64 *)a1, v118);
            sub_164EDD0(*(_QWORD *)a1, v113);
          }
          return;
        }
      }
    }
  }
  if ( !*(_QWORD *)(a2 + 48) && *(__int16 *)(a2 + 18) >= 0 )
    goto LABEL_95;
  v30 = sub_1625790(a2, 3);
  if ( v30 )
  {
    v31 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
    if ( v31 == 16 )
      v31 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a2 + 16LL) + 8LL);
    if ( (unsigned __int8)(v31 - 1) > 5u )
    {
      v134 = a2;
      v23 = "fpmath requires a floating point result!";
      BYTE1(v140) = 1;
      goto LABEL_23;
    }
    if ( *(_DWORD *)(v30 + 8) != 1 )
    {
      v134 = a2;
      v23 = "fpmath takes one operand!";
      BYTE1(v140) = 1;
      goto LABEL_23;
    }
    v32 = *(_QWORD *)(v30 - 8);
    if ( !v32 || *(_BYTE *)v32 != 1 || (v33 = *(_QWORD *)(v32 + 136), *(_BYTE *)(v33 + 16) != 14) )
    {
      v134 = a2;
      v23 = "invalid fpmath accuracy!";
      BYTE1(v140) = 1;
      goto LABEL_23;
    }
    v115 = *(_QWORD *)(v33 + 32);
    v36 = sub_1698270(a2, 3);
    if ( v36 != v115 )
    {
      v134 = a2;
      v23 = "fpmath accuracy must have float type";
      BYTE1(v140) = 1;
      goto LABEL_23;
    }
    if ( v36 == sub_16982C0(a2, 3, v34, v35) )
      v37 = *(_QWORD *)(v33 + 40) + 8LL;
    else
      v37 = v33 + 32;
    v38 = *(_BYTE *)(v37 + 18) & 7;
    if ( v38 <= 1u || v38 == 3 || (*(_BYTE *)(v37 + 18) & 8) != 0 )
    {
      v134 = a2;
      v23 = "fpmath accuracy not a positive number!";
      BYTE1(v140) = 1;
      goto LABEL_23;
    }
  }
  if ( !*(_QWORD *)(a2 + 48) && *(__int16 *)(a2 + 18) >= 0 )
  {
LABEL_95:
    if ( *(__int16 *)(a2 + 18) < 0 )
      goto LABEL_183;
LABEL_96:
    if ( *(_BYTE *)(a2 + 16) != 78 )
      goto LABEL_97;
    v87 = *(_QWORD *)(a2 - 24);
    if ( *(_BYTE *)(v87 + 16) )
      goto LABEL_97;
    if ( (*(_BYTE *)(v87 + 33) & 0x20) == 0 )
      goto LABEL_97;
    v88 = *(_DWORD *)(v87 + 36);
    if ( (unsigned int)(v88 - 35) > 3 || v88 == 37 )
      goto LABEL_97;
    v89 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v90 = *(unsigned __int8 **)(*(_QWORD *)(a2 + 24 * (1 - v89)) + 24LL);
    if ( v90 && *v90 != 25 )
      v90 = 0;
    v91 = *(_QWORD *)(a2 + 24 * (2 - v89));
    v92 = *(_QWORD *)(v91 + 24);
    if ( !v92 )
      goto LABEL_97;
    if ( *(_BYTE *)v92 != 6 )
      goto LABEL_97;
    if ( !v90 )
      goto LABEL_97;
    if ( !(unsigned __int8)sub_15B1200(*(_QWORD *)(v91 + 24)) )
      goto LABEL_97;
    sub_15B1350((__int64)&v134, *(unsigned __int64 **)(v92 + 24), *(unsigned __int64 **)(v92 + 32));
    if ( !(_BYTE)v136 )
      goto LABEL_97;
    if ( (v90[36] & 0x40) != 0 )
      goto LABEL_97;
    v93 = v134;
    v120 = v135;
    sub_15B1130((__int64)&v130, (__int64)v90);
    if ( !(_BYTE)v131 )
      goto LABEL_97;
    if ( v93 + v120 > (unsigned __int64)v130 )
    {
      v94 = *(_QWORD *)a1;
      v138 = "fragment is larger than or outside of variable";
      LOWORD(v140) = 259;
      if ( v94 )
      {
        sub_16E2CE0(&v138, v94);
        v95 = *(_BYTE **)(v94 + 24);
        if ( (unsigned __int64)v95 >= *(_QWORD *)(v94 + 16) )
        {
          sub_16E7DE0(v94, 10);
        }
        else
        {
          *(_QWORD *)(v94 + 24) = v95 + 1;
          *v95 = 10;
        }
      }
      *(_BYTE *)(a1 + 72) |= *(_BYTE *)(a1 + 74);
      v73 = *(_QWORD *)a1 == 0;
      *(_BYTE *)(a1 + 73) = 1;
      if ( v73 )
        goto LABEL_97;
    }
    else
    {
      if ( v130 != (const char *)v93 )
        goto LABEL_97;
      v138 = "fragment covers entire variable";
      LOWORD(v140) = 259;
      sub_16521E0((__int64 *)a1, (__int64)&v138);
      if ( !*(_QWORD *)a1 )
        goto LABEL_97;
    }
    sub_164FA80((__int64 *)a1, a2);
    sub_164ED40((__int64 *)a1, v90);
LABEL_97:
    sub_165A590((__int64)&v138, a1 + 160, a2);
    return;
  }
  v45 = sub_1625790(a2, 4);
  if ( !v45 )
    goto LABEL_182;
  v46 = *(unsigned __int8 *)(a2 + 16);
  v47 = (unsigned int)(v46 - 29);
  if ( (unsigned __int8)(v46 - 29) > 0x31u || (v48 = 0x2000002000001LL, !_bittest64(&v48, v47)) )
  {
    v134 = a2;
    v23 = "Ranges are only for loads, calls and invokes!";
    BYTE1(v140) = 1;
    goto LABEL_23;
  }
  v121 = (unsigned __int8 *)v45;
  v97 = *(_QWORD *)a2;
  v101 = *(_DWORD *)(v45 + 8);
  if ( (v101 & 1) != 0 )
  {
    BYTE1(v140) = 1;
    v74 = "Unfinished range!";
    goto LABEL_237;
  }
  if ( v101 <= 1 )
  {
    BYTE1(v140) = 1;
    v74 = "It should have at least one range!";
LABEL_237:
    v75 = *(_QWORD *)a1;
    v138 = v74;
    LOBYTE(v140) = 3;
    if ( v75 )
    {
      sub_16E2CE0(&v138, v75);
      v76 = *(_BYTE **)(v75 + 24);
      if ( (unsigned __int64)v76 >= *(_QWORD *)(v75 + 16) )
      {
        sub_16E7DE0(v75, 10);
      }
      else
      {
        *(_QWORD *)(v75 + 24) = v76 + 1;
        *v76 = 10;
      }
      v75 = *(_QWORD *)a1;
    }
    *(_BYTE *)(a1 + 72) = 1;
    if ( v75 && v121 )
    {
      sub_15562E0(v121, v75, a1 + 16, *(_QWORD *)(a1 + 8));
      v77 = *(_QWORD *)a1;
      v78 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
      if ( (unsigned __int64)v78 >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
      {
        sub_16E7DE0(v77, 10);
      }
      else
      {
        *(_QWORD *)(v77 + 24) = v78 + 1;
        *v78 = 10;
      }
    }
    goto LABEL_182;
  }
  sub_15897D0((__int64)&v130, 1u, 1);
  v108 = 0;
  v119 = (_BYTE *)a1;
  while ( 1 )
  {
    v54 = *((unsigned int *)v121 + 2);
    v55 = *(_QWORD *)&v121[8 * (2 * v108 - v54)];
    if ( *(_BYTE *)v55 != 1 || (v49 = *(_QWORD *)(v55 + 136), *(_BYTE *)(v49 + 16) != 13) )
    {
      BYTE1(v140) = 1;
      a1 = (__int64)v119;
      v56 = "The lower limit must be an integer!";
      goto LABEL_174;
    }
    v50 = *(_QWORD *)&v121[8 * (2 * v108 + 1 - v54)];
    if ( *(_BYTE *)v50 != 1 || (v51 = *(_QWORD *)(v50 + 136), *(_BYTE *)(v51 + 16) != 13) )
    {
      BYTE1(v140) = 1;
      a1 = (__int64)v119;
      v56 = "The upper limit must be an integer!";
LABEL_174:
      v138 = v56;
      LOBYTE(v140) = 3;
      sub_164FF40((__int64 *)a1, (__int64)&v138);
      if ( *(_QWORD *)a1 )
        sub_164FA80((__int64 *)a1, 0);
      goto LABEL_176;
    }
    if ( *(_QWORD *)v49 != *(_QWORD *)v51 || v97 != *(_QWORD *)v51 )
    {
      a1 = (__int64)v119;
      LOWORD(v140) = 259;
      v138 = "Range types must match instruction type!";
      v134 = a2;
      sub_1654980(v119, (__int64)&v138, &v134);
      goto LABEL_176;
    }
    v123 = *(_DWORD *)(v51 + 32);
    if ( v123 > 0x40 )
      sub_16A4FD0(&v122, v51 + 24);
    else
      v122 = *(const char **)(v51 + 24);
    v125 = *(_DWORD *)(v49 + 32);
    if ( v125 > 0x40 )
      sub_16A4FD0(&v124, v49 + 24);
    else
      v124 = *(_QWORD *)(v49 + 24);
    v139 = v123;
    if ( v123 > 0x40 )
      sub_16A4FD0(&v138, &v122);
    else
      v138 = v122;
    v129 = v125;
    if ( v125 > 0x40 )
      sub_16A4FD0(&v128, &v124);
    else
      v128 = v124;
    sub_15898E0((__int64)&v134, (__int64)&v128, (__int64 *)&v138);
    if ( v129 > 0x40 && v128 )
      j_j___libc_free_0_0(v128);
    if ( v139 > 0x40 && v138 )
      j_j___libc_free_0_0(v138);
    if ( sub_158A120((__int64)&v134) || sub_158A0B0((__int64)&v134) )
      break;
    if ( v108 )
    {
      sub_158BE00((__int64)&v138, (__int64)&v134, (__int64)&v130);
      v52 = sub_158A120((__int64)&v138);
      if ( v141 > 0x40 && v140 )
        j_j___libc_free_0_0(v140);
      if ( v139 > 0x40 && v138 )
        j_j___libc_free_0_0(v138);
      if ( !v52 )
      {
        BYTE1(v140) = 1;
        a1 = (__int64)v119;
        v96 = "Intervals are overlapping";
        goto LABEL_348;
      }
      if ( (int)sub_16AEA10(&v124, &v130) <= 0 )
      {
        BYTE1(v140) = 1;
        a1 = (__int64)v119;
        v96 = "Intervals are not in order";
        goto LABEL_348;
      }
      if ( sub_164E6A0((__int64)&v134, &v130) )
      {
        BYTE1(v140) = 1;
        a1 = (__int64)v119;
        v96 = "Intervals are contiguous";
        goto LABEL_348;
      }
    }
    v129 = v123;
    if ( v123 > 0x40 )
      sub_16A4FD0(&v128, &v122);
    else
      v128 = (__int64)v122;
    v127 = v125;
    if ( v125 > 0x40 )
      sub_16A4FD0(&v126, &v124);
    else
      v126 = (const char *)v124;
    sub_15898E0((__int64)&v138, (__int64)&v126, &v128);
    if ( v131 > 0x40 && v130 )
      j_j___libc_free_0_0(v130);
    v130 = v138;
    v53 = v139;
    v139 = 0;
    v131 = v53;
    if ( v133 > 0x40 && v132 )
    {
      j_j___libc_free_0_0(v132);
      v132 = v140;
      v133 = v141;
      if ( v139 > 0x40 && v138 )
        j_j___libc_free_0_0(v138);
    }
    else
    {
      v132 = v140;
      v133 = v141;
    }
    if ( v127 > 0x40 && v126 )
      j_j___libc_free_0_0(v126);
    if ( v129 > 0x40 && v128 )
      j_j___libc_free_0_0(v128);
    if ( v137 > 0x40 && v136 )
      j_j___libc_free_0_0(v136);
    if ( (unsigned int)v135 > 0x40 && v134 )
      j_j___libc_free_0_0(v134);
    if ( v125 > 0x40 && v124 )
      j_j___libc_free_0_0(v124);
    if ( v123 > 0x40 && v122 )
      j_j___libc_free_0_0(v122);
    if ( v101 >> 1 == ++v108 )
    {
      a1 = (__int64)v119;
      if ( v101 <= 5 )
        goto LABEL_176;
      v79 = v121;
      v80 = *(_QWORD *)&v121[-8 * *((unsigned int *)v121 + 2)];
      if ( *(_BYTE *)v80 != 1 || (v81 = *(_QWORD *)(v80 + 136), *(_BYTE *)(v81 + 16) != 13) )
        BUG();
      v125 = *(_DWORD *)(v81 + 32);
      if ( v125 > 0x40 )
      {
        sub_16A4FD0(&v124, v81 + 24);
        v79 = v121;
      }
      else
      {
        v124 = *(_QWORD *)(v81 + 24);
      }
      v82 = *(_QWORD *)&v79[8 * (1LL - *((unsigned int *)v79 + 2))];
      if ( *(_BYTE *)v82 != 1 || (v83 = *(_QWORD *)(v82 + 136), *(_BYTE *)(v83 + 16) != 13) )
        BUG();
      v127 = *(_DWORD *)(v83 + 32);
      if ( v127 > 0x40 )
        sub_16A4FD0(&v126, v83 + 24);
      else
        v126 = *(const char **)(v83 + 24);
      v139 = v127;
      if ( v127 > 0x40 )
        sub_16A4FD0(&v138, &v126);
      else
        v138 = v126;
      v129 = v125;
      if ( v125 > 0x40 )
        sub_16A4FD0(&v128, &v124);
      else
        v128 = v124;
      sub_15898E0((__int64)&v134, (__int64)&v128, (__int64 *)&v138);
      if ( v129 > 0x40 && v128 )
        j_j___libc_free_0_0(v128);
      if ( v139 > 0x40 && v138 )
        j_j___libc_free_0_0(v138);
      sub_158BE00((__int64)&v138, (__int64)&v134, (__int64)&v130);
      v84 = sub_158A120((__int64)&v138);
      if ( v141 > 0x40 && v140 )
        j_j___libc_free_0_0(v140);
      if ( v139 > 0x40 && v138 )
        j_j___libc_free_0_0(v138);
      if ( v84 )
      {
        if ( !sub_164E6A0((__int64)&v134, &v130) )
        {
LABEL_288:
          if ( v137 > 0x40 && v136 )
            j_j___libc_free_0_0(v136);
          if ( (unsigned int)v135 > 0x40 && v134 )
            j_j___libc_free_0_0(v134);
          if ( v127 > 0x40 && v126 )
            j_j___libc_free_0_0(v126);
          if ( v125 > 0x40 )
          {
            v85 = v124;
            if ( v124 )
              goto LABEL_299;
          }
          goto LABEL_176;
        }
        BYTE1(v140) = 1;
        v86 = "Intervals are contiguous";
      }
      else
      {
        BYTE1(v140) = 1;
        v86 = "Intervals are overlapping";
      }
      v138 = v86;
      LOBYTE(v140) = 3;
      sub_1659E40(v119, (__int64)&v138, &v121);
      goto LABEL_288;
    }
  }
  BYTE1(v140) = 1;
  a1 = (__int64)v119;
  v96 = "Range must not be empty!";
LABEL_348:
  v138 = v96;
  LOBYTE(v140) = 3;
  sub_1659E40((_BYTE *)a1, (__int64)&v138, &v121);
  if ( v137 > 0x40 && v136 )
    j_j___libc_free_0_0(v136);
  if ( (unsigned int)v135 > 0x40 && v134 )
    j_j___libc_free_0_0(v134);
  if ( v125 > 0x40 && v124 )
    j_j___libc_free_0_0(v124);
  if ( v123 > 0x40 )
  {
    v85 = (__int64)v122;
    if ( v122 )
LABEL_299:
      j_j___libc_free_0_0(v85);
  }
LABEL_176:
  if ( v133 > 0x40 && v132 )
    j_j___libc_free_0_0(v132);
  if ( v131 > 0x40 && v130 )
    j_j___libc_free_0_0(v130);
LABEL_182:
  if ( !*(_QWORD *)(a2 + 48) )
    goto LABEL_95;
LABEL_183:
  if ( !sub_1625790(a2, 11) )
    goto LABEL_186;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 15 )
  {
    v134 = a2;
    v23 = "nonnull applies only to pointer types";
    BYTE1(v140) = 1;
    goto LABEL_23;
  }
  if ( *(_BYTE *)(a2 + 16) != 54 )
  {
    v134 = a2;
    v23 = "nonnull applies only to load instructions, use attributes for calls or invokes";
    BYTE1(v140) = 1;
    goto LABEL_23;
  }
LABEL_186:
  if ( !*(_QWORD *)(a2 + 48) && *(__int16 *)(a2 + 18) >= 0 )
    goto LABEL_96;
  v57 = sub_1625790(a2, 12);
  if ( v57 )
    sub_1654A70((_BYTE *)a1, a2, v57);
  if ( !*(_QWORD *)(a2 + 48) && *(__int16 *)(a2 + 18) >= 0 )
    goto LABEL_96;
  v58 = sub_1625790(a2, 13);
  if ( v58 )
    sub_1654A70((_BYTE *)a1, a2, v58);
  if ( !*(_QWORD *)(a2 + 48) && *(__int16 *)(a2 + 18) >= 0 )
    goto LABEL_96;
  v59 = (unsigned __int8 *)sub_1625790(a2, 1);
  if ( v59 )
    sub_16635B0((__int64 **)(a1 + 1600), a2, v59, v60, v61, v62);
  if ( !*(_QWORD *)(a2 + 48) && *(__int16 *)(a2 + 18) >= 0 )
    goto LABEL_96;
  v63 = sub_1625790(a2, 17);
  if ( v63 )
  {
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
    {
      if ( *(_BYTE *)(a2 + 16) == 54 )
      {
        if ( *(_DWORD *)(v63 + 8) == 1 )
        {
          v64 = *(_QWORD *)(v63 - 8);
          if ( *(_BYTE *)v64 == 1
            && (v65 = *(_QWORD *)(v64 + 136), *(_BYTE *)(v65 + 16) == 13)
            && sub_1642F90(*(_QWORD *)v65, 64) )
          {
            v66 = *(_QWORD *)(v65 + 24);
            if ( *(_DWORD *)(v65 + 32) > 0x40u )
              v66 = *(_QWORD *)v66;
            if ( !v66 || (v66 & (v66 - 1)) != 0 )
            {
              v134 = a2;
              v23 = "align metadata value must be a power of 2!";
              BYTE1(v140) = 1;
            }
            else
            {
              if ( v66 <= 0x20000000 )
                goto LABEL_208;
              v134 = a2;
              v23 = "alignment is larger that implementation defined limit";
              BYTE1(v140) = 1;
            }
          }
          else
          {
            v134 = a2;
            v23 = "align metadata value must be an i64!";
            BYTE1(v140) = 1;
          }
        }
        else
        {
          v134 = a2;
          v23 = "align takes one operand!";
          BYTE1(v140) = 1;
        }
      }
      else
      {
        v134 = a2;
        v23 = "align applies only to load instructions, use attributes for calls or invokes";
        BYTE1(v140) = 1;
      }
    }
    else
    {
      v134 = a2;
      v23 = "align applies only to pointer types";
      BYTE1(v140) = 1;
    }
LABEL_23:
    v138 = v23;
    LOBYTE(v140) = 3;
    sub_1654980((_BYTE *)a1, (__int64)&v138, &v134);
  }
  else
  {
LABEL_208:
    v67 = *(unsigned __int8 **)(a2 + 48);
    if ( !v67 )
      goto LABEL_96;
    if ( *v67 == 5 )
    {
      sub_1656110((_QWORD *)a1, *(_QWORD *)(a2 + 48));
      goto LABEL_96;
    }
    v138 = "invalid !dbg metadata attachment";
    LOWORD(v140) = 259;
    sub_16521E0((__int64 *)a1, (__int64)&v138);
    if ( *(_QWORD *)a1 )
    {
      sub_164FA80((__int64 *)a1, a2);
      sub_164ED40((__int64 *)a1, v67);
    }
  }
}
