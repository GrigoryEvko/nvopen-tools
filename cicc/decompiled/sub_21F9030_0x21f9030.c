// Function: sub_21F9030
// Address: 0x21f9030
//
__int64 __fastcall sub_21F9030(__int64 a1)
{
  int v2; // ecx
  __int64 v3; // rax
  _QWORD *v4; // rdi
  _QWORD *k; // rax
  int v6; // ecx
  __int64 result; // rax
  __int64 v8; // rax
  _QWORD *v9; // rdi
  _QWORD *v10; // r14
  _QWORD *v11; // rax
  _QWORD *v12; // rbx
  __int64 v13; // r15
  int v14; // eax
  __int64 v15; // rdx
  _DWORD *v16; // rax
  _DWORD *n; // rdx
  __int64 v18; // r15
  unsigned int v19; // edx
  _QWORD *v20; // r8
  unsigned int v21; // ecx
  unsigned int v22; // edx
  int v23; // edx
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rax
  int v26; // ebx
  __int64 v27; // r13
  __int64 v28; // rdx
  __int64 jj; // rdx
  unsigned int v30; // ecx
  _DWORD *v31; // r8
  unsigned int v32; // eax
  int v33; // eax
  unsigned __int64 v34; // rax
  unsigned __int64 v35; // rax
  __int64 v36; // r9
  _DWORD *v37; // rax
  __int64 v38; // rdx
  _DWORD *ii; // rdx
  _QWORD *v40; // r14
  _QWORD *v41; // rax
  _QWORD *v42; // rbx
  __int64 v43; // r15
  int v44; // eax
  __int64 v45; // rdx
  _DWORD *v46; // rax
  _DWORD *i; // rdx
  __int64 v48; // r15
  unsigned int v49; // edx
  _QWORD *v50; // r8
  unsigned int v51; // ecx
  unsigned int v52; // edx
  int v53; // edx
  unsigned __int64 v54; // rax
  unsigned __int64 v55; // rax
  int v56; // ebx
  __int64 v57; // r13
  _QWORD *v58; // rax
  __int64 v59; // rdx
  _QWORD *m; // rdx
  unsigned int v61; // ecx
  _DWORD *v62; // r8
  unsigned int v63; // eax
  int v64; // eax
  unsigned __int64 v65; // rax
  unsigned __int64 v66; // rax
  __int64 v67; // r9
  _DWORD *v68; // rax
  __int64 v69; // rdx
  _DWORD *j; // rdx
  _DWORD *v71; // rax
  _DWORD *v72; // rax
  _QWORD *v73; // rax
  int v74; // [rsp+4h] [rbp-3Ch]
  int v75; // [rsp+4h] [rbp-3Ch]
  __int64 v76; // [rsp+8h] [rbp-38h]
  __int64 v77; // [rsp+8h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 280);
  if ( !v2 )
  {
    ++*(_QWORD *)(a1 + 264);
    goto LABEL_3;
  }
  v4 = *(_QWORD **)(a1 + 272);
  v40 = &v4[2 * *(unsigned int *)(a1 + 288)];
  if ( v4 == v40 )
    goto LABEL_71;
  v41 = v4;
  while ( 1 )
  {
    v42 = v41;
    if ( *v41 != -8 && *v41 != -16 )
      break;
    v41 += 2;
    if ( v40 == v41 )
      goto LABEL_71;
  }
  if ( v40 == v41 )
  {
LABEL_71:
    ++*(_QWORD *)(a1 + 264);
    goto LABEL_87;
  }
  v43 = v41[1];
  v44 = *(_DWORD *)(v43 + 16);
  ++*(_QWORD *)v43;
  if ( v44 )
    goto LABEL_102;
LABEL_74:
  if ( *(_DWORD *)(v43 + 20) )
  {
    v45 = *(unsigned int *)(v43 + 24);
    if ( (unsigned int)v45 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(v43 + 8));
      *(_QWORD *)(v43 + 8) = 0;
      *(_QWORD *)(v43 + 16) = 0;
      *(_DWORD *)(v43 + 24) = 0;
      goto LABEL_79;
    }
LABEL_76:
    v46 = *(_DWORD **)(v43 + 8);
    for ( i = &v46[2 * v45]; i != v46; v46 += 2 )
      *v46 = -1;
    *(_QWORD *)(v43 + 16) = 0;
    goto LABEL_79;
  }
  while ( 1 )
  {
LABEL_79:
    v48 = v42[1];
    if ( v48 )
    {
      j___libc_free_0(*(_QWORD *)(v48 + 8));
      j_j___libc_free_0(v48, 32);
    }
    v42 += 2;
    if ( v42 == v40 )
      break;
    while ( *v42 == -8 || *v42 == -16 )
    {
      v42 += 2;
      if ( v40 == v42 )
        goto LABEL_85;
    }
    if ( v40 == v42 )
      break;
    v43 = v42[1];
    v44 = *(_DWORD *)(v43 + 16);
    ++*(_QWORD *)v43;
    if ( !v44 )
      goto LABEL_74;
LABEL_102:
    v61 = 4 * v44;
    v45 = *(unsigned int *)(v43 + 24);
    if ( (unsigned int)(4 * v44) < 0x40 )
      v61 = 64;
    if ( v61 >= (unsigned int)v45 )
      goto LABEL_76;
    v62 = *(_DWORD **)(v43 + 8);
    v63 = v44 - 1;
    if ( !v63 )
    {
      v75 = 128;
      v67 = 1024;
      goto LABEL_110;
    }
    _BitScanReverse(&v63, v63);
    v64 = 1 << (33 - (v63 ^ 0x1F));
    if ( v64 < 64 )
      v64 = 64;
    if ( (_DWORD)v45 == v64 )
    {
      *(_QWORD *)(v43 + 16) = 0;
      v71 = &v62[2 * v45];
      do
      {
        if ( v62 )
          *v62 = -1;
        v62 += 2;
      }
      while ( v71 != v62 );
    }
    else
    {
      v65 = (4 * v64 / 3u + 1) | ((unsigned __int64)(4 * v64 / 3u + 1) >> 1);
      v66 = ((v65 | (v65 >> 2)) >> 4) | v65 | (v65 >> 2) | ((((v65 | (v65 >> 2)) >> 4) | v65 | (v65 >> 2)) >> 8);
      v75 = (v66 | (v66 >> 16)) + 1;
      v67 = 8 * ((v66 | (v66 >> 16)) + 1);
LABEL_110:
      v77 = v67;
      j___libc_free_0(v62);
      *(_DWORD *)(v43 + 24) = v75;
      v68 = (_DWORD *)sub_22077B0(v77);
      v69 = *(unsigned int *)(v43 + 24);
      *(_QWORD *)(v43 + 16) = 0;
      *(_QWORD *)(v43 + 8) = v68;
      for ( j = &v68[2 * v69]; j != v68; v68 += 2 )
      {
        if ( v68 )
          *v68 = -1;
      }
    }
  }
LABEL_85:
  v2 = *(_DWORD *)(a1 + 280);
  ++*(_QWORD *)(a1 + 264);
  if ( !v2 )
  {
LABEL_3:
    if ( !*(_DWORD *)(a1 + 284) )
      goto LABEL_8;
    v3 = *(unsigned int *)(a1 + 288);
    v4 = *(_QWORD **)(a1 + 272);
    if ( (unsigned int)v3 > 0x40 )
    {
      j___libc_free_0(v4);
      *(_QWORD *)(a1 + 272) = 0;
      *(_QWORD *)(a1 + 280) = 0;
      *(_DWORD *)(a1 + 288) = 0;
      goto LABEL_8;
    }
    goto LABEL_5;
  }
  v4 = *(_QWORD **)(a1 + 272);
LABEL_87:
  v49 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 288);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v49 = 64;
  if ( (unsigned int)v3 <= v49 )
  {
LABEL_5:
    for ( k = &v4[2 * v3]; k != v4; v4 += 2 )
      *v4 = -8;
    *(_QWORD *)(a1 + 280) = 0;
    goto LABEL_8;
  }
  v50 = v4;
  v51 = v2 - 1;
  if ( !v51 )
  {
    v57 = 2048;
    v56 = 128;
    goto LABEL_95;
  }
  _BitScanReverse(&v52, v51);
  v53 = 1 << (33 - (v52 ^ 0x1F));
  if ( v53 < 64 )
    v53 = 64;
  if ( (_DWORD)v3 == v53 )
  {
    *(_QWORD *)(a1 + 280) = 0;
    v73 = &v4[2 * v3];
    do
    {
      if ( v50 )
        *v50 = -8;
      v50 += 2;
    }
    while ( v73 != v50 );
  }
  else
  {
    v54 = (4 * v53 / 3u + 1) | ((unsigned __int64)(4 * v53 / 3u + 1) >> 1);
    v55 = ((v54 | (v54 >> 2)) >> 4) | v54 | (v54 >> 2) | ((((v54 | (v54 >> 2)) >> 4) | v54 | (v54 >> 2)) >> 8);
    v56 = (v55 | (v55 >> 16)) + 1;
    v57 = 16 * ((v55 | (v55 >> 16)) + 1);
LABEL_95:
    j___libc_free_0(v4);
    *(_DWORD *)(a1 + 288) = v56;
    v58 = (_QWORD *)sub_22077B0(v57);
    v59 = *(unsigned int *)(a1 + 288);
    *(_QWORD *)(a1 + 280) = 0;
    *(_QWORD *)(a1 + 272) = v58;
    for ( m = &v58[2 * v59]; m != v58; v58 += 2 )
    {
      if ( v58 )
        *v58 = -8;
    }
  }
LABEL_8:
  v6 = *(_DWORD *)(a1 + 312);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 296);
    goto LABEL_10;
  }
  v9 = *(_QWORD **)(a1 + 304);
  v10 = &v9[2 * *(unsigned int *)(a1 + 320)];
  if ( v9 == v10 )
    goto LABEL_21;
  v11 = *(_QWORD **)(a1 + 304);
  while ( 1 )
  {
    v12 = v11;
    if ( *v11 != -16 && *v11 != -8 )
      break;
    v11 += 2;
    if ( v10 == v11 )
      goto LABEL_21;
  }
  if ( v10 == v11 )
  {
LABEL_21:
    ++*(_QWORD *)(a1 + 296);
    goto LABEL_37;
  }
  v13 = v11[1];
  v14 = *(_DWORD *)(v13 + 16);
  ++*(_QWORD *)v13;
  if ( v14 )
    goto LABEL_52;
LABEL_24:
  if ( *(_DWORD *)(v13 + 20) )
  {
    v15 = *(unsigned int *)(v13 + 24);
    if ( (unsigned int)v15 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(v13 + 8));
      *(_QWORD *)(v13 + 8) = 0;
      *(_QWORD *)(v13 + 16) = 0;
      *(_DWORD *)(v13 + 24) = 0;
      goto LABEL_29;
    }
LABEL_26:
    v16 = *(_DWORD **)(v13 + 8);
    for ( n = &v16[2 * v15]; n != v16; v16 += 2 )
      *v16 = -1;
    *(_QWORD *)(v13 + 16) = 0;
    goto LABEL_29;
  }
  while ( 1 )
  {
LABEL_29:
    v18 = v12[1];
    if ( v18 )
    {
      j___libc_free_0(*(_QWORD *)(v18 + 8));
      j_j___libc_free_0(v18, 32);
    }
    v12 += 2;
    if ( v12 == v10 )
      break;
    while ( *v12 == -8 || *v12 == -16 )
    {
      v12 += 2;
      if ( v10 == v12 )
        goto LABEL_35;
    }
    if ( v12 == v10 )
      break;
    v13 = v12[1];
    v14 = *(_DWORD *)(v13 + 16);
    ++*(_QWORD *)v13;
    if ( !v14 )
      goto LABEL_24;
LABEL_52:
    v30 = 4 * v14;
    v15 = *(unsigned int *)(v13 + 24);
    if ( (unsigned int)(4 * v14) < 0x40 )
      v30 = 64;
    if ( (unsigned int)v15 <= v30 )
      goto LABEL_26;
    v31 = *(_DWORD **)(v13 + 8);
    v32 = v14 - 1;
    if ( v32 )
    {
      _BitScanReverse(&v32, v32);
      v33 = 1 << (33 - (v32 ^ 0x1F));
      if ( v33 < 64 )
        v33 = 64;
      if ( (_DWORD)v15 == v33 )
      {
        *(_QWORD *)(v13 + 16) = 0;
        v72 = &v31[2 * v15];
        do
        {
          if ( v31 )
            *v31 = -1;
          v31 += 2;
        }
        while ( v72 != v31 );
        continue;
      }
      v34 = (4 * v33 / 3u + 1) | ((unsigned __int64)(4 * v33 / 3u + 1) >> 1);
      v35 = ((v34 | (v34 >> 2)) >> 4) | v34 | (v34 >> 2) | ((((v34 | (v34 >> 2)) >> 4) | v34 | (v34 >> 2)) >> 8);
      v74 = (v35 | (v35 >> 16)) + 1;
      v36 = 8 * ((v35 | (v35 >> 16)) + 1);
    }
    else
    {
      v74 = 128;
      v36 = 1024;
    }
    v76 = v36;
    j___libc_free_0(v31);
    *(_DWORD *)(v13 + 24) = v74;
    v37 = (_DWORD *)sub_22077B0(v76);
    v38 = *(unsigned int *)(v13 + 24);
    *(_QWORD *)(v13 + 16) = 0;
    *(_QWORD *)(v13 + 8) = v37;
    for ( ii = &v37[2 * v38]; ii != v37; v37 += 2 )
    {
      if ( v37 )
        *v37 = -1;
    }
  }
LABEL_35:
  v6 = *(_DWORD *)(a1 + 312);
  ++*(_QWORD *)(a1 + 296);
  if ( !v6 )
  {
LABEL_10:
    result = *(unsigned int *)(a1 + 316);
    if ( !(_DWORD)result )
      return result;
    v8 = *(unsigned int *)(a1 + 320);
    v9 = *(_QWORD **)(a1 + 304);
    if ( (unsigned int)v8 > 0x40 )
    {
      result = j___libc_free_0(v9);
      *(_QWORD *)(a1 + 304) = 0;
      *(_QWORD *)(a1 + 312) = 0;
      *(_DWORD *)(a1 + 320) = 0;
      return result;
    }
    goto LABEL_12;
  }
  v9 = *(_QWORD **)(a1 + 304);
LABEL_37:
  v19 = 4 * v6;
  v8 = *(unsigned int *)(a1 + 320);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v19 = 64;
  if ( (unsigned int)v8 <= v19 )
  {
LABEL_12:
    for ( result = (__int64)&v9[2 * v8]; (_QWORD *)result != v9; v9 += 2 )
      *v9 = -8;
    *(_QWORD *)(a1 + 312) = 0;
    return result;
  }
  v20 = v9;
  v21 = v6 - 1;
  if ( !v21 )
  {
    v27 = 2048;
    v26 = 128;
    goto LABEL_45;
  }
  _BitScanReverse(&v22, v21);
  v23 = 1 << (33 - (v22 ^ 0x1F));
  if ( v23 < 64 )
    v23 = 64;
  if ( (_DWORD)v8 == v23 )
  {
    *(_QWORD *)(a1 + 312) = 0;
    result = (__int64)&v9[2 * v8];
    do
    {
      if ( v20 )
        *v20 = -8;
      v20 += 2;
    }
    while ( (_QWORD *)result != v20 );
  }
  else
  {
    v24 = (4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1);
    v25 = ((v24 | (v24 >> 2)) >> 4) | v24 | (v24 >> 2) | ((((v24 | (v24 >> 2)) >> 4) | v24 | (v24 >> 2)) >> 8);
    v26 = (v25 | (v25 >> 16)) + 1;
    v27 = 16 * ((v25 | (v25 >> 16)) + 1);
LABEL_45:
    j___libc_free_0(v9);
    *(_DWORD *)(a1 + 320) = v26;
    result = sub_22077B0(v27);
    v28 = *(unsigned int *)(a1 + 320);
    *(_QWORD *)(a1 + 312) = 0;
    *(_QWORD *)(a1 + 304) = result;
    for ( jj = result + 16 * v28; jj != result; result += 16 )
    {
      if ( result )
        *(_QWORD *)result = -8;
    }
  }
  return result;
}
