// Function: sub_165D700
// Address: 0x165d700
//
__int64 __fastcall sub_165D700(__int64 a1)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 i; // r13
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // rsi
  __int64 v9; // rbx
  __int64 v10; // rsi
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  unsigned int v14; // r14d
  int v15; // r13d
  unsigned __int8 *v16; // r15
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r13
  _BYTE *v20; // rax
  __int64 v21; // rsi
  char v22; // al
  __int64 v23; // rdi
  _BYTE *v24; // rax
  int v25; // ecx
  _QWORD *v26; // rsi
  __int64 *v27; // rax
  __int64 v28; // rdx
  __int64 *v29; // r14
  __int64 *j; // rbx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // r15
  __int64 v35; // rax
  __int64 *v36; // rdx
  __int64 *v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rdi
  __int64 v40; // rax
  __int64 v41; // r14
  int v42; // r15d
  unsigned int v43; // ebx
  __int64 v44; // rax
  unsigned __int8 *v45; // r13
  _BYTE *v46; // rax
  __int64 v47; // rax
  __int64 *v48; // rcx
  __int64 *v49; // rdx
  __int64 *v50; // rsi
  __int64 v51; // r14
  __int64 v52; // r13
  __int64 v53; // rbx
  _BYTE *v54; // rax
  __int64 v55; // rsi
  __int64 v56; // rdi
  _BYTE *v57; // rax
  int v58; // eax
  __int64 v59; // rdx
  _QWORD *v60; // rax
  _QWORD *k; // rdx
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 v65; // r13
  _BYTE *v66; // rax
  __int64 v67; // rsi
  char v68; // al
  __int64 v69; // rdi
  _BYTE *v70; // rax
  __int64 v71; // r14
  _BYTE *v72; // rax
  __int64 v73; // rsi
  unsigned __int8 *v74; // rdi
  __int64 v75; // rdi
  _BYTE *v76; // rax
  unsigned int v77; // ecx
  _QWORD *v78; // rdi
  unsigned int v79; // eax
  __int64 v80; // rax
  unsigned __int64 v81; // rax
  unsigned __int64 v82; // rax
  int v83; // ebx
  __int64 v84; // r13
  _QWORD *v85; // rax
  __int64 v86; // rdx
  _QWORD *m; // rdx
  __int64 v88; // r9
  _BYTE *v89; // rax
  __int64 v90; // rax
  __int64 v91; // rsi
  __int64 v92; // rcx
  __int64 v93; // rdx
  __int64 v94; // rax
  __int64 v95; // r13
  _BYTE *v96; // rax
  __int64 v97; // rdx
  __int64 v98; // rax
  _QWORD *v99; // rax
  __int64 v100; // [rsp+0h] [rbp-60h]
  __int64 v101; // [rsp+0h] [rbp-60h]
  __int64 v102; // [rsp+0h] [rbp-60h]
  _QWORD v103[2]; // [rsp+10h] [rbp-50h] BYREF
  char v104; // [rsp+20h] [rbp-40h]
  char v105; // [rsp+21h] [rbp-3Fh]

  v2 = *(_QWORD *)(a1 + 8);
  *(_BYTE *)(a1 + 72) = 0;
  v3 = *(_QWORD *)(v2 + 32);
  for ( i = v2 + 24; i != v3; v3 = *(_QWORD *)(v3 + 8) )
  {
    while ( 1 )
    {
      if ( !v3 )
        BUG();
      if ( *(_DWORD *)(v3 - 20) == 75 )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( i == v3 )
        goto LABEL_9;
    }
    v5 = *(unsigned int *)(a1 + 1120);
    if ( (unsigned int)v5 >= *(_DWORD *)(a1 + 1124) )
    {
      sub_16CD150(a1 + 1112, a1 + 1128, 0, 8);
      v5 = *(unsigned int *)(a1 + 1120);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 1112) + 8 * v5) = v3 - 56;
    ++*(_DWORD *)(a1 + 1120);
  }
LABEL_9:
  if ( *(_DWORD *)(a1 + 744) )
  {
    v91 = *(_QWORD *)(a1 + 736);
    v92 = v91 + 16LL * *(unsigned int *)(a1 + 752);
    if ( v91 != v92 )
    {
      while ( 1 )
      {
        v93 = *(_QWORD *)v91;
        v94 = v91;
        if ( *(_QWORD *)v91 != -8 && v93 != -16 )
          break;
        v91 += 16;
        if ( v92 == v91 )
          goto LABEL_10;
      }
      if ( v92 != v91 )
      {
        if ( *(_DWORD *)(v91 + 8) < *(_DWORD *)(v91 + 12) )
        {
LABEL_144:
          v95 = *(_QWORD *)a1;
          v102 = v93;
          v105 = 1;
          v103[0] = "all indices passed to llvm.localrecover must be less than the number of arguments passed ot llvm.loc"
                    "alescape in the parent function";
          v104 = 3;
          if ( v95 )
          {
            sub_16E2CE0(v103, v95);
            v96 = *(_BYTE **)(v95 + 24);
            v97 = v102;
            if ( (unsigned __int64)v96 >= *(_QWORD *)(v95 + 16) )
            {
              sub_16E7DE0(v95, 10);
              v98 = *(_QWORD *)a1;
              v97 = v102;
            }
            else
            {
              *(_QWORD *)(v95 + 24) = v96 + 1;
              *v96 = 10;
              v98 = *(_QWORD *)a1;
            }
            *(_BYTE *)(a1 + 72) = 1;
            if ( v98 )
              sub_164FA80((__int64 *)a1, v97);
          }
          else
          {
            *(_BYTE *)(a1 + 72) = 1;
          }
        }
        else
        {
          while ( 1 )
          {
            v94 += 16;
            if ( v94 == v92 )
              break;
            while ( 1 )
            {
              v93 = *(_QWORD *)v94;
              if ( *(_QWORD *)v94 != -8 && v93 != -16 )
                break;
              v94 += 16;
              if ( v92 == v94 )
                goto LABEL_10;
            }
            if ( v92 == v94 )
              break;
            if ( *(_DWORD *)(v94 + 8) < *(_DWORD *)(v94 + 12) )
              goto LABEL_144;
          }
        }
      }
    }
  }
LABEL_10:
  v6 = *(_QWORD *)(a1 + 8);
  v7 = *(_QWORD *)(v6 + 16);
  if ( v6 + 8 != v7 )
  {
    do
    {
      v8 = v7 - 56;
      if ( !v7 )
        v8 = 0;
      sub_1655620(a1, v8);
      v7 = *(_QWORD *)(v7 + 8);
    }
    while ( v6 + 8 != v7 );
    v6 = *(_QWORD *)(a1 + 8);
  }
  v9 = *(_QWORD *)(v6 + 48);
  if ( v6 + 40 != v9 )
  {
    do
    {
      v10 = v9 - 48;
      if ( !v9 )
        v10 = 0;
      sub_1652030(a1, v10);
      v9 = *(_QWORD *)(v9 + 8);
    }
    while ( v6 + 40 != v9 );
    v6 = *(_QWORD *)(a1 + 8);
  }
  v11 = *(_QWORD *)(v6 + 80);
  v100 = v6 + 72;
  if ( v6 + 72 != v11 )
  {
    while ( 2 )
    {
      while ( 2 )
      {
        v12 = sub_161F640(v11);
        if ( v13 > 8 && *(_QWORD *)v12 == 0x6762642E6D766C6CLL && *(_BYTE *)(v12 + 8) == 46 )
        {
          v63 = sub_161F640(v11);
          if ( v64 != 11
            || *(_QWORD *)v63 != 0x6762642E6D766C6CLL
            || *(_WORD *)(v63 + 8) != 25390
            || *(_BYTE *)(v63 + 10) != 117 )
          {
            v65 = *(_QWORD *)a1;
            v105 = 1;
            v103[0] = "unrecognized named metadata node in the llvm.dbg namespace";
            v104 = 3;
            if ( v65 )
            {
              sub_16E2CE0(v103, v65);
              v66 = *(_BYTE **)(v65 + 24);
              if ( (unsigned __int64)v66 >= *(_QWORD *)(v65 + 16) )
              {
                sub_16E7DE0(v65, 10);
              }
              else
              {
                *(_QWORD *)(v65 + 24) = v66 + 1;
                *v66 = 10;
              }
              v67 = *(_QWORD *)a1;
              v68 = *(_BYTE *)(a1 + 74);
              *(_BYTE *)(a1 + 73) = 1;
              *(_BYTE *)(a1 + 72) |= v68;
              if ( !v11 || !v67 )
                goto LABEL_43;
              sub_1556C90(v11, v67, a1 + 16, 0);
              v69 = *(_QWORD *)a1;
              v70 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
              if ( (unsigned __int64)v70 >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
              {
                sub_16E7DE0(v69, 10);
                v11 = *(_QWORD *)(v11 + 8);
                if ( v100 == v11 )
                  goto LABEL_44;
              }
              else
              {
                *(_QWORD *)(v69 + 24) = v70 + 1;
                *v70 = 10;
                v11 = *(_QWORD *)(v11 + 8);
                if ( v100 == v11 )
                  goto LABEL_44;
              }
              continue;
            }
            goto LABEL_87;
          }
        }
        break;
      }
      v14 = 0;
      v15 = sub_161F520(v11);
      if ( !v15 )
        goto LABEL_43;
      while ( 1 )
      {
        v16 = (unsigned __int8 *)sub_161F530(v11, v14);
        v17 = sub_161F640(v11);
        if ( v18 == 11
          && *(_QWORD *)v17 == 0x6762642E6D766C6CLL
          && *(_WORD *)(v17 + 8) == 25390
          && *(_BYTE *)(v17 + 10) == 117 )
        {
          break;
        }
        if ( v16 )
          goto LABEL_26;
LABEL_27:
        if ( v15 == ++v14 )
          goto LABEL_43;
      }
      if ( !v16 || *v16 != 16 )
      {
        v19 = *(_QWORD *)a1;
        v105 = 1;
        v103[0] = "invalid compile unit";
        v104 = 3;
        if ( v19 )
        {
          sub_16E2CE0(v103, v19);
          v20 = *(_BYTE **)(v19 + 24);
          if ( (unsigned __int64)v20 >= *(_QWORD *)(v19 + 16) )
          {
            sub_16E7DE0(v19, 10);
          }
          else
          {
            *(_QWORD *)(v19 + 24) = v20 + 1;
            *v20 = 10;
          }
          v21 = *(_QWORD *)a1;
          v22 = *(_BYTE *)(a1 + 74);
          *(_BYTE *)(a1 + 73) = 1;
          *(_BYTE *)(a1 + 72) |= v22;
          if ( v21 )
          {
            if ( v11 )
            {
              sub_1556C90(v11, v21, a1 + 16, 0);
              v23 = *(_QWORD *)a1;
              v24 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
              if ( (unsigned __int64)v24 >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
              {
                sub_16E7DE0(v23, 10);
              }
              else
              {
                *(_QWORD *)(v23 + 24) = v24 + 1;
                *v24 = 10;
              }
            }
            if ( v16 )
              sub_164ED40((__int64 *)a1, v16);
          }
LABEL_43:
          v11 = *(_QWORD *)(v11 + 8);
          if ( v100 == v11 )
          {
LABEL_44:
            v6 = *(_QWORD *)(a1 + 8);
            goto LABEL_45;
          }
          continue;
        }
LABEL_87:
        *(_BYTE *)(a1 + 73) = 1;
        *(_BYTE *)(a1 + 72) |= *(_BYTE *)(a1 + 74);
        v11 = *(_QWORD *)(v11 + 8);
        if ( v100 == v11 )
          goto LABEL_44;
        continue;
      }
      break;
    }
LABEL_26:
    sub_1656110((_QWORD *)a1, (__int64)v16);
    goto LABEL_27;
  }
LABEL_45:
  v25 = *(_DWORD *)(v6 + 136);
  if ( v25 )
  {
    v26 = *(_QWORD **)(v6 + 128);
    if ( *v26 != -8 && *v26 )
    {
      v29 = *(__int64 **)(v6 + 128);
    }
    else
    {
      v27 = v26 + 1;
      do
      {
        do
        {
          v28 = *v27;
          v29 = v27++;
        }
        while ( !v28 );
      }
      while ( v28 == -8 );
    }
    for ( j = &v26[v25]; v29 != j; v6 = *(_QWORD *)(a1 + 8) )
    {
      while ( 1 )
      {
        v31 = sub_1580C70((_QWORD *)(*v29 + 8));
        v33 = sub_1632000(v6, v31, v32);
        v34 = v33;
        if ( v33 && (*(_BYTE *)(v33 + 32) & 0xF) == 8 )
        {
          v88 = *(_QWORD *)a1;
          v105 = 1;
          v103[0] = "comdat global value has private linkage";
          v104 = 3;
          if ( v88 )
          {
            v101 = v88;
            sub_16E2CE0(v103, v88);
            v89 = *(_BYTE **)(v101 + 24);
            if ( (unsigned __int64)v89 >= *(_QWORD *)(v101 + 16) )
            {
              sub_16E7DE0(v101, 10);
            }
            else
            {
              *(_QWORD *)(v101 + 24) = v89 + 1;
              *v89 = 10;
            }
            v90 = *(_QWORD *)a1;
            *(_BYTE *)(a1 + 72) = 1;
            if ( v90 )
              sub_164FA80((__int64 *)a1, v34);
          }
          else
          {
            *(_BYTE *)(a1 + 72) = 1;
          }
        }
        v35 = v29[1];
        v36 = v29 + 1;
        if ( !v35 || v35 == -8 )
          break;
        ++v29;
        v6 = *(_QWORD *)(a1 + 8);
        if ( v36 == j )
          goto LABEL_60;
      }
      v37 = v29 + 2;
      do
      {
        do
        {
          v38 = *v37;
          v29 = v37++;
        }
        while ( v38 == -8 );
      }
      while ( !v38 );
    }
  }
LABEL_60:
  sub_165C9A0(a1, v6);
  v39 = *(_QWORD *)(a1 + 8);
  v105 = 1;
  v103[0] = "llvm.ident";
  v104 = 3;
  v40 = sub_1632310(v39, (__int64)v103);
  v41 = v40;
  if ( v40 )
  {
    v42 = sub_161F520(v40);
    if ( v42 )
    {
      v43 = 0;
      while ( 1 )
      {
        v44 = sub_161F530(v41, v43);
        v45 = (unsigned __int8 *)v44;
        if ( *(_DWORD *)(v44 + 8) != 1 )
          break;
        v46 = *(_BYTE **)(v44 - 8);
        if ( !v46 || *v46 )
        {
          v71 = *(_QWORD *)a1;
          v105 = 1;
          v103[0] = "invalid value for llvm.ident metadata entry operand(the operand should be a string)";
          v104 = 3;
          if ( v71 )
          {
            sub_16E2CE0(v103, v71);
            v72 = *(_BYTE **)(v71 + 24);
            if ( (unsigned __int64)v72 >= *(_QWORD *)(v71 + 16) )
            {
              sub_16E7DE0(v71, 10);
            }
            else
            {
              *(_QWORD *)(v71 + 24) = v72 + 1;
              *v72 = 10;
            }
            v73 = *(_QWORD *)a1;
            *(_BYTE *)(a1 + 72) = 1;
            if ( v73 )
            {
              v74 = (unsigned __int8 *)*((_QWORD *)v45 - 1);
              if ( v74 )
              {
                sub_15562E0(v74, v73, a1 + 16, *(_QWORD *)(a1 + 8));
                v75 = *(_QWORD *)a1;
                v76 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
                if ( (unsigned __int64)v76 >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
                {
                  sub_16E7DE0(v75, 10);
                }
                else
                {
                  *(_QWORD *)(v75 + 24) = v76 + 1;
                  *v76 = 10;
                }
              }
            }
          }
          else
          {
            *(_BYTE *)(a1 + 72) = 1;
          }
          goto LABEL_67;
        }
        if ( v42 == ++v43 )
          goto LABEL_67;
      }
      v105 = 1;
      v103[0] = "incorrect number of operands in llvm.ident metadata";
      v104 = 3;
      sub_164FF40((__int64 *)a1, (__int64)v103);
      if ( *(_QWORD *)a1 )
        sub_164ED40((__int64 *)a1, v45);
    }
  }
LABEL_67:
  sub_164F440(a1);
  v47 = *(unsigned int *)(a1 + 1120);
  if ( (_DWORD)v47 )
  {
    v48 = *(__int64 **)(a1 + 1112);
    v49 = v48 + 1;
    v50 = &v48[v47];
    v51 = *v48;
    if ( v48 + 1 != v50 )
    {
      while ( 1 )
      {
        v52 = *v49;
        if ( ((*(_WORD *)(v51 + 18) >> 4) & 0x3FF) != ((*(_WORD *)(*v49 + 18) >> 4) & 0x3FF) )
          break;
        if ( v50 == ++v49 )
          goto LABEL_80;
      }
      v53 = *(_QWORD *)a1;
      v105 = 1;
      v103[0] = "All llvm.experimental.deoptimize declarations must have the same calling convention";
      v104 = 3;
      if ( v53 )
      {
        sub_16E2CE0(v103, v53);
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
        v55 = *(_QWORD *)a1;
        *(_BYTE *)(a1 + 72) = 1;
        if ( v55 )
        {
          if ( *(_BYTE *)(v51 + 16) <= 0x17u )
          {
            sub_1553920((__int64 *)v51, v55, 1, a1 + 16);
            v56 = *(_QWORD *)a1;
            v57 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
            if ( (unsigned __int64)v57 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
              goto LABEL_78;
          }
          else
          {
            sub_155BD40(v51, v55, a1 + 16, 0);
            v56 = *(_QWORD *)a1;
            v57 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
            if ( (unsigned __int64)v57 < *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
            {
LABEL_78:
              *(_QWORD *)(v56 + 24) = v57 + 1;
              *v57 = 10;
LABEL_79:
              sub_164FA80((__int64 *)a1, v52);
              goto LABEL_80;
            }
          }
          sub_16E7DE0(v56, 10);
          goto LABEL_79;
        }
      }
      else
      {
        *(_BYTE *)(a1 + 72) = 1;
      }
    }
  }
LABEL_80:
  v58 = *(_DWORD *)(a1 + 640);
  ++*(_QWORD *)(a1 + 624);
  if ( !v58 )
  {
    if ( !*(_DWORD *)(a1 + 644) )
      return *(unsigned __int8 *)(a1 + 72) ^ 1u;
    v59 = *(unsigned int *)(a1 + 648);
    if ( (unsigned int)v59 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 632));
      *(_QWORD *)(a1 + 632) = 0;
      *(_QWORD *)(a1 + 640) = 0;
      *(_DWORD *)(a1 + 648) = 0;
      return *(unsigned __int8 *)(a1 + 72) ^ 1u;
    }
    goto LABEL_83;
  }
  v77 = 4 * v58;
  v59 = *(unsigned int *)(a1 + 648);
  if ( (unsigned int)(4 * v58) < 0x40 )
    v77 = 64;
  if ( (unsigned int)v59 <= v77 )
  {
LABEL_83:
    v60 = *(_QWORD **)(a1 + 632);
    for ( k = &v60[2 * v59]; k != v60; v60 += 2 )
      *v60 = -8;
    *(_QWORD *)(a1 + 640) = 0;
    return *(unsigned __int8 *)(a1 + 72) ^ 1u;
  }
  v78 = *(_QWORD **)(a1 + 632);
  v79 = v58 - 1;
  if ( !v79 )
  {
    v84 = 2048;
    v83 = 128;
LABEL_114:
    j___libc_free_0(v78);
    *(_DWORD *)(a1 + 648) = v83;
    v85 = (_QWORD *)sub_22077B0(v84);
    v86 = *(unsigned int *)(a1 + 648);
    *(_QWORD *)(a1 + 640) = 0;
    *(_QWORD *)(a1 + 632) = v85;
    for ( m = &v85[2 * v86]; m != v85; v85 += 2 )
    {
      if ( v85 )
        *v85 = -8;
    }
    return *(unsigned __int8 *)(a1 + 72) ^ 1u;
  }
  _BitScanReverse(&v79, v79);
  v80 = (unsigned int)(1 << (33 - (v79 ^ 0x1F)));
  if ( (int)v80 < 64 )
    v80 = 64;
  if ( (_DWORD)v80 != (_DWORD)v59 )
  {
    v81 = (4 * (int)v80 / 3u + 1) | ((unsigned __int64)(4 * (int)v80 / 3u + 1) >> 1);
    v82 = ((v81 | (v81 >> 2)) >> 4) | v81 | (v81 >> 2) | ((((v81 | (v81 >> 2)) >> 4) | v81 | (v81 >> 2)) >> 8);
    v83 = (v82 | (v82 >> 16)) + 1;
    v84 = 16 * ((v82 | (v82 >> 16)) + 1);
    goto LABEL_114;
  }
  *(_QWORD *)(a1 + 640) = 0;
  v99 = &v78[2 * v80];
  do
  {
    if ( v78 )
      *v78 = -8;
    v78 += 2;
  }
  while ( v99 != v78 );
  return *(unsigned __int8 *)(a1 + 72) ^ 1u;
}
