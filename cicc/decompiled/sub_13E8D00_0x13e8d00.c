// Function: sub_13E8D00
// Address: 0x13e8d00
//
__int64 __fastcall sub_13E8D00(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  bool v14; // zf
  __int64 v15; // rax
  __int64 v16; // r13
  int v17; // eax
  __int64 v18; // rdx
  int v19; // r15d
  _QWORD *v20; // rbx
  unsigned int v21; // eax
  __int64 v22; // rdx
  _QWORD *v23; // r14
  __int64 v24; // r15
  __int64 v25; // rax
  __int64 v26; // r12
  __int64 v27; // rdx
  __int64 v28; // rax
  int v29; // r14d
  _QWORD *v30; // r12
  unsigned int v31; // eax
  _QWORD *v32; // rbx
  unsigned __int64 v33; // rdi
  __int64 v35; // r8
  __int64 v36; // rax
  __int64 v37; // r12
  __int64 v38; // rdx
  __int64 v39; // rax
  unsigned __int64 v40; // rdi
  __int64 v41; // rdi
  __int64 v42; // rdi
  unsigned int v43; // ecx
  _QWORD *v44; // rax
  _QWORD *i; // rdx
  unsigned int v46; // edx
  __int64 v47; // rdx
  __int64 v48; // rdi
  __int64 v49; // rdi
  int v50; // ebx
  unsigned int v51; // r15d
  unsigned int v52; // eax
  _QWORD *v53; // rdi
  unsigned __int64 v54; // rax
  unsigned __int64 v55; // rdi
  _QWORD *v56; // rax
  __int64 v57; // rdx
  _QWORD *k; // rdx
  int v59; // ebx
  unsigned int v60; // r14d
  unsigned int v61; // eax
  _QWORD *v62; // rdi
  unsigned __int64 v63; // rdx
  unsigned __int64 v64; // rax
  _QWORD *v65; // rax
  __int64 v66; // rdx
  _QWORD *m; // rdx
  _QWORD *v68; // rdi
  unsigned int v69; // eax
  int v70; // eax
  unsigned __int64 v71; // rax
  unsigned __int64 v72; // rax
  int v73; // ebx
  __int64 v74; // r12
  _QWORD *v75; // rax
  __int64 v76; // rdx
  _QWORD *j; // rdx
  _QWORD *v78; // rax
  _QWORD *v79; // rax
  _QWORD *v80; // rax
  __int64 v81; // [rsp+0h] [rbp-40h]
  __int64 v82; // [rsp+0h] [rbp-40h]
  __int64 v83; // [rsp+8h] [rbp-38h]
  __int64 v84; // [rsp+8h] [rbp-38h]
  __int64 v85; // [rsp+8h] [rbp-38h]
  __int64 v86; // [rsp+8h] [rbp-38h]
  __int64 v87; // [rsp+8h] [rbp-38h]
  __int64 v88; // [rsp+8h] [rbp-38h]

  v2 = (__int64 *)a1[1];
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_156:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F9D764 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_156;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F9D764);
  a1[20] = sub_14CF090(v5, a2);
  v6 = sub_1632FA0(*(_QWORD *)(a2 + 40));
  v7 = sub_160F9A0(a1[1], &unk_4F9E06C, 1);
  if ( v7 && (v8 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v7 + 104LL))(v7, &unk_4F9E06C)) != 0 )
    v9 = v8 + 160;
  else
    v9 = 0;
  v10 = (__int64 *)a1[1];
  a1[23] = v9;
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_155:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F9B6E8 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_155;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F9B6E8);
  v14 = a1[24] == 0;
  a1[22] = v13 + 360;
  if ( v14 )
    return 0;
  v15 = sub_13E7A30(a1 + 24, a1[20], v6, a1[23]);
  ++*(_QWORD *)v15;
  v16 = v15;
  v17 = *(_DWORD *)(v15 + 16);
  if ( !v17 )
  {
    if ( !*(_DWORD *)(v16 + 20) )
      goto LABEL_17;
    v18 = *(unsigned int *)(v16 + 24);
    if ( (unsigned int)v18 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(v16 + 8));
      *(_QWORD *)(v16 + 8) = 0;
      *(_QWORD *)(v16 + 16) = 0;
      *(_DWORD *)(v16 + 24) = 0;
      goto LABEL_17;
    }
    goto LABEL_91;
  }
  v43 = 4 * v17;
  v18 = *(unsigned int *)(v16 + 24);
  if ( (unsigned int)(4 * v17) < 0x40 )
    v43 = 64;
  if ( (unsigned int)v18 <= v43 )
  {
LABEL_91:
    v44 = *(_QWORD **)(v16 + 8);
    for ( i = &v44[v18]; i != v44; ++v44 )
      *v44 = -8;
    *(_QWORD *)(v16 + 16) = 0;
    goto LABEL_17;
  }
  v68 = *(_QWORD **)(v16 + 8);
  v69 = v17 - 1;
  if ( !v69 )
  {
    v74 = 1024;
    v73 = 128;
LABEL_133:
    j___libc_free_0(v68);
    *(_DWORD *)(v16 + 24) = v73;
    v75 = (_QWORD *)sub_22077B0(v74);
    v76 = *(unsigned int *)(v16 + 24);
    *(_QWORD *)(v16 + 16) = 0;
    *(_QWORD *)(v16 + 8) = v75;
    for ( j = &v75[v76]; j != v75; ++v75 )
    {
      if ( v75 )
        *v75 = -8;
    }
    goto LABEL_17;
  }
  _BitScanReverse(&v69, v69);
  v70 = 1 << (33 - (v69 ^ 0x1F));
  if ( v70 < 64 )
    v70 = 64;
  if ( (_DWORD)v18 != v70 )
  {
    v71 = (4 * v70 / 3u + 1) | ((unsigned __int64)(4 * v70 / 3u + 1) >> 1);
    v72 = ((v71 | (v71 >> 2)) >> 4) | v71 | (v71 >> 2) | ((((v71 | (v71 >> 2)) >> 4) | v71 | (v71 >> 2)) >> 8);
    v73 = (v72 | (v72 >> 16)) + 1;
    v74 = 8 * ((v72 | (v72 >> 16)) + 1);
    goto LABEL_133;
  }
  *(_QWORD *)(v16 + 16) = 0;
  v78 = &v68[v18];
  do
  {
    if ( v68 )
      *v68 = -8;
    ++v68;
  }
  while ( v78 != v68 );
LABEL_17:
  v19 = *(_DWORD *)(v16 + 48);
  ++*(_QWORD *)(v16 + 32);
  if ( !v19 && !*(_DWORD *)(v16 + 52) )
    goto LABEL_41;
  v20 = *(_QWORD **)(v16 + 40);
  v21 = 4 * v19;
  v22 = *(unsigned int *)(v16 + 56);
  v23 = &v20[2 * v22];
  if ( (unsigned int)(4 * v19) < 0x40 )
    v21 = 64;
  if ( (unsigned int)v22 <= v21 )
  {
    if ( v20 == v23 )
      goto LABEL_40;
    while ( *v20 == -8 )
    {
LABEL_39:
      v20 += 2;
      if ( v20 == v23 )
        goto LABEL_40;
    }
    if ( *v20 == -16 || (v24 = v20[1]) == 0 )
    {
LABEL_38:
      *v20 = -8;
      goto LABEL_39;
    }
    if ( (*(_BYTE *)(v24 + 48) & 1) != 0 )
    {
      v26 = v24 + 56;
      v27 = v24 + 248;
    }
    else
    {
      v25 = *(unsigned int *)(v24 + 64);
      v26 = *(_QWORD *)(v24 + 56);
      if ( !(_DWORD)v25 )
        goto LABEL_80;
      v27 = v26 + 48 * v25;
    }
    do
    {
      if ( *(_QWORD *)v26 != -8 && *(_QWORD *)v26 != -16 && *(_DWORD *)(v26 + 8) == 3 )
      {
        if ( *(_DWORD *)(v26 + 40) > 0x40u )
        {
          v41 = *(_QWORD *)(v26 + 32);
          if ( v41 )
          {
            v84 = v27;
            j_j___libc_free_0_0(v41);
            v27 = v84;
          }
        }
        if ( *(_DWORD *)(v26 + 24) > 0x40u )
        {
          v42 = *(_QWORD *)(v26 + 16);
          if ( v42 )
          {
            v85 = v27;
            j_j___libc_free_0_0(v42);
            v27 = v85;
          }
        }
      }
      v26 += 48;
    }
    while ( v26 != v27 );
    if ( (*(_BYTE *)(v24 + 48) & 1) != 0 )
      goto LABEL_34;
    v26 = *(_QWORD *)(v24 + 56);
LABEL_80:
    j___libc_free_0(v26);
LABEL_34:
    *(_QWORD *)v24 = &unk_49EE2B0;
    v28 = *(_QWORD *)(v24 + 24);
    if ( v28 != 0 && v28 != -8 && v28 != -16 )
      sub_1649B30(v24 + 8);
    j_j___libc_free_0(v24, 248);
    goto LABEL_38;
  }
  do
  {
    if ( *v20 == -16 )
      goto LABEL_70;
    if ( *v20 == -8 )
      goto LABEL_70;
    v35 = v20[1];
    if ( !v35 )
      goto LABEL_70;
    if ( (*(_BYTE *)(v35 + 48) & 1) != 0 )
    {
      v37 = v35 + 56;
      v38 = v35 + 248;
    }
    else
    {
      v36 = *(unsigned int *)(v35 + 64);
      v37 = *(_QWORD *)(v35 + 56);
      if ( !(_DWORD)v36 )
        goto LABEL_103;
      v38 = v37 + 48 * v36;
    }
    do
    {
      if ( *(_QWORD *)v37 != -16 && *(_QWORD *)v37 != -8 && *(_DWORD *)(v37 + 8) == 3 )
      {
        if ( *(_DWORD *)(v37 + 40) > 0x40u )
        {
          v48 = *(_QWORD *)(v37 + 32);
          if ( v48 )
          {
            v81 = v38;
            v87 = v35;
            j_j___libc_free_0_0(v48);
            v38 = v81;
            v35 = v87;
          }
        }
        if ( *(_DWORD *)(v37 + 24) > 0x40u )
        {
          v49 = *(_QWORD *)(v37 + 16);
          if ( v49 )
          {
            v82 = v38;
            v88 = v35;
            j_j___libc_free_0_0(v49);
            v38 = v82;
            v35 = v88;
          }
        }
      }
      v37 += 48;
    }
    while ( v37 != v38 );
    if ( (*(_BYTE *)(v35 + 48) & 1) != 0 )
      goto LABEL_66;
    v37 = *(_QWORD *)(v35 + 56);
LABEL_103:
    v86 = v35;
    j___libc_free_0(v37);
    v35 = v86;
LABEL_66:
    *(_QWORD *)v35 = &unk_49EE2B0;
    v39 = *(_QWORD *)(v35 + 24);
    if ( v39 != 0 && v39 != -8 && v39 != -16 )
    {
      v83 = v35;
      sub_1649B30(v35 + 8);
      v35 = v83;
    }
    j_j___libc_free_0(v35, 248);
LABEL_70:
    v20 += 2;
  }
  while ( v20 != v23 );
  v46 = *(_DWORD *)(v16 + 56);
  if ( v19 )
  {
    v50 = 64;
    v51 = v19 - 1;
    if ( v51 )
    {
      _BitScanReverse(&v52, v51);
      v50 = 1 << (33 - (v52 ^ 0x1F));
      if ( v50 < 64 )
        v50 = 64;
    }
    v53 = *(_QWORD **)(v16 + 40);
    if ( v46 == v50 )
    {
      *(_QWORD *)(v16 + 48) = 0;
      v80 = &v53[2 * v46];
      do
      {
        if ( v53 )
          *v53 = -8;
        v53 += 2;
      }
      while ( v80 != v53 );
    }
    else
    {
      j___libc_free_0(v53);
      v54 = ((((((((4 * v50 / 3u + 1) | ((unsigned __int64)(4 * v50 / 3u + 1) >> 1)) >> 2)
               | (4 * v50 / 3u + 1)
               | ((unsigned __int64)(4 * v50 / 3u + 1) >> 1)) >> 4)
             | (((4 * v50 / 3u + 1) | ((unsigned __int64)(4 * v50 / 3u + 1) >> 1)) >> 2)
             | (4 * v50 / 3u + 1)
             | ((unsigned __int64)(4 * v50 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v50 / 3u + 1) | ((unsigned __int64)(4 * v50 / 3u + 1) >> 1)) >> 2)
             | (4 * v50 / 3u + 1)
             | ((unsigned __int64)(4 * v50 / 3u + 1) >> 1)) >> 4)
           | (((4 * v50 / 3u + 1) | ((unsigned __int64)(4 * v50 / 3u + 1) >> 1)) >> 2)
           | (4 * v50 / 3u + 1)
           | ((unsigned __int64)(4 * v50 / 3u + 1) >> 1)) >> 16;
      v55 = (v54
           | (((((((4 * v50 / 3u + 1) | ((unsigned __int64)(4 * v50 / 3u + 1) >> 1)) >> 2)
               | (4 * v50 / 3u + 1)
               | ((unsigned __int64)(4 * v50 / 3u + 1) >> 1)) >> 4)
             | (((4 * v50 / 3u + 1) | ((unsigned __int64)(4 * v50 / 3u + 1) >> 1)) >> 2)
             | (4 * v50 / 3u + 1)
             | ((unsigned __int64)(4 * v50 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v50 / 3u + 1) | ((unsigned __int64)(4 * v50 / 3u + 1) >> 1)) >> 2)
             | (4 * v50 / 3u + 1)
             | ((unsigned __int64)(4 * v50 / 3u + 1) >> 1)) >> 4)
           | (((4 * v50 / 3u + 1) | ((unsigned __int64)(4 * v50 / 3u + 1) >> 1)) >> 2)
           | (4 * v50 / 3u + 1)
           | ((unsigned __int64)(4 * v50 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(v16 + 56) = v55;
      v56 = (_QWORD *)sub_22077B0(16 * v55);
      v57 = *(unsigned int *)(v16 + 56);
      *(_QWORD *)(v16 + 48) = 0;
      *(_QWORD *)(v16 + 40) = v56;
      for ( k = &v56[2 * v57]; k != v56; v56 += 2 )
      {
        if ( v56 )
          *v56 = -8;
      }
    }
  }
  else
  {
    if ( v46 )
    {
      j___libc_free_0(*(_QWORD *)(v16 + 40));
      *(_QWORD *)(v16 + 40) = 0;
      *(_QWORD *)(v16 + 48) = 0;
      *(_DWORD *)(v16 + 56) = 0;
      goto LABEL_41;
    }
LABEL_40:
    *(_QWORD *)(v16 + 48) = 0;
  }
LABEL_41:
  v29 = *(_DWORD *)(v16 + 80);
  ++*(_QWORD *)(v16 + 64);
  if ( v29 || *(_DWORD *)(v16 + 84) )
  {
    v30 = *(_QWORD **)(v16 + 72);
    v31 = 4 * v29;
    v32 = &v30[10 * *(unsigned int *)(v16 + 88)];
    if ( (unsigned int)(4 * v29) < 0x40 )
      v31 = 64;
    if ( *(_DWORD *)(v16 + 88) <= v31 )
    {
      while ( v30 != v32 )
      {
        if ( *v30 != -8 )
        {
          if ( *v30 != -16 )
          {
            v33 = v30[3];
            if ( v33 != v30[2] )
              _libc_free(v33);
          }
          *v30 = -8;
        }
        v30 += 10;
      }
      goto LABEL_54;
    }
    do
    {
      if ( *v30 != -8 && *v30 != -16 )
      {
        v40 = v30[3];
        if ( v40 != v30[2] )
          _libc_free(v40);
      }
      v30 += 10;
    }
    while ( v32 != v30 );
    v47 = *(unsigned int *)(v16 + 88);
    if ( v29 )
    {
      v59 = 64;
      v60 = v29 - 1;
      if ( v60 )
      {
        _BitScanReverse(&v61, v60);
        v59 = 1 << (33 - (v61 ^ 0x1F));
        if ( v59 < 64 )
          v59 = 64;
      }
      v62 = *(_QWORD **)(v16 + 72);
      if ( (_DWORD)v47 == v59 )
      {
        *(_QWORD *)(v16 + 80) = 0;
        v79 = &v62[10 * v47];
        do
        {
          if ( v62 )
            *v62 = -8;
          v62 += 10;
        }
        while ( v79 != v62 );
      }
      else
      {
        j___libc_free_0(v62);
        v63 = ((((((((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
                 | (4 * v59 / 3u + 1)
                 | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 4)
               | (((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
               | (4 * v59 / 3u + 1)
               | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 8)
             | (((((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
               | (4 * v59 / 3u + 1)
               | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 4)
             | (((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
             | (4 * v59 / 3u + 1)
             | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 16;
        v64 = (v63
             | (((((((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
                 | (4 * v59 / 3u + 1)
                 | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 4)
               | (((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
               | (4 * v59 / 3u + 1)
               | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 8)
             | (((((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
               | (4 * v59 / 3u + 1)
               | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 4)
             | (((4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1)) >> 2)
             | (4 * v59 / 3u + 1)
             | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1))
            + 1;
        *(_DWORD *)(v16 + 88) = v64;
        v65 = (_QWORD *)sub_22077B0(80 * v64);
        v66 = *(unsigned int *)(v16 + 88);
        *(_QWORD *)(v16 + 80) = 0;
        *(_QWORD *)(v16 + 72) = v65;
        for ( m = &v65[10 * v66]; m != v65; v65 += 10 )
        {
          if ( v65 )
            *v65 = -8;
        }
      }
    }
    else
    {
      if ( (_DWORD)v47 )
      {
        j___libc_free_0(*(_QWORD *)(v16 + 72));
        *(_QWORD *)(v16 + 72) = 0;
        *(_QWORD *)(v16 + 80) = 0;
        *(_DWORD *)(v16 + 88) = 0;
        return 0;
      }
LABEL_54:
      *(_QWORD *)(v16 + 80) = 0;
    }
  }
  return 0;
}
