// Function: sub_2DF8870
// Address: 0x2df8870
//
__int64 __fastcall sub_2DF8870(__int64 a1)
{
  int v2; // r15d
  unsigned int v3; // eax
  __int64 v4; // r12
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 v7; // r13
  unsigned __int64 v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // r13
  unsigned __int64 v12; // r12
  __int64 v13; // rcx
  unsigned __int64 v14; // r8
  unsigned __int64 v15; // r9
  unsigned __int64 v16; // rdi
  unsigned __int64 *v17; // r14
  unsigned __int64 v18; // rdi
  __int64 v19; // rsi
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 j; // r12
  unsigned __int64 v23; // r13
  __int64 v24; // rsi
  int v25; // eax
  __int64 v26; // rdx
  _DWORD *v27; // rax
  _DWORD *k; // rdx
  int v29; // eax
  __int64 v30; // rdx
  _QWORD *v31; // rax
  _QWORD *ii; // rdx
  _QWORD *v34; // rdi
  _QWORD *v35; // rax
  unsigned int v36; // ecx
  unsigned int v37; // eax
  __int64 v38; // rdi
  int v39; // r12d
  __int64 v40; // rax
  unsigned int v41; // ecx
  unsigned int v42; // eax
  _DWORD *v43; // rdi
  int v44; // r12d
  _DWORD *v45; // rax
  unsigned __int64 v46; // rdi
  int v47; // edx
  int v48; // r12d
  unsigned int v49; // r15d
  unsigned int v50; // eax
  _DWORD *v51; // rdi
  unsigned int v52; // eax
  _DWORD *v53; // rax
  __int64 v54; // rdx
  _DWORD *i; // rdx
  unsigned int v56; // eax
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 n; // rdx
  unsigned int v60; // eax
  _DWORD *v61; // rax
  __int64 v62; // rdx
  _DWORD *m; // rdx
  _DWORD *v64; // rax
  __int64 v65; // [rsp+8h] [rbp-38h]

  *(_QWORD *)(a1 + 104) = 0;
  sub_2DF5A20(*(_QWORD *)(a1 + 144));
  v2 = *(_DWORD *)(a1 + 192);
  ++*(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = a1 + 136;
  *(_QWORD *)(a1 + 160) = a1 + 136;
  *(_QWORD *)(a1 + 168) = 0;
  if ( !v2 && !*(_DWORD *)(a1 + 196) )
    goto LABEL_14;
  v3 = 4 * v2;
  v4 = *(_QWORD *)(a1 + 184);
  v5 = *(unsigned int *)(a1 + 200);
  v6 = 32 * v5;
  if ( (unsigned int)(4 * v2) < 0x40 )
    v3 = 64;
  v7 = v4 + v6;
  if ( (unsigned int)v5 <= v3 )
  {
    for ( ; v4 != v7; v4 += 32 )
    {
      if ( *(_DWORD *)v4 != -1 )
      {
        if ( *(_DWORD *)v4 != -2 )
        {
          v8 = *(_QWORD *)(v4 + 8);
          if ( v8 )
            j_j___libc_free_0(v8);
        }
        *(_DWORD *)v4 = -1;
      }
    }
    goto LABEL_13;
  }
  do
  {
    if ( *(_DWORD *)v4 <= 0xFFFFFFFD )
    {
      v46 = *(_QWORD *)(v4 + 8);
      if ( v46 )
        j_j___libc_free_0(v46);
    }
    v4 += 32;
  }
  while ( v4 != v7 );
  v47 = *(_DWORD *)(a1 + 200);
  if ( !v2 )
  {
    if ( !v47 )
    {
LABEL_13:
      *(_QWORD *)(a1 + 192) = 0;
      goto LABEL_14;
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 184), v6, 8);
    *(_DWORD *)(a1 + 200) = 0;
    goto LABEL_94;
  }
  v48 = 64;
  v49 = v2 - 1;
  if ( v49 )
  {
    _BitScanReverse(&v50, v49);
    v48 = 1 << (33 - (v50 ^ 0x1F));
    if ( v48 < 64 )
      v48 = 64;
  }
  v51 = *(_DWORD **)(a1 + 184);
  if ( v48 != v47 )
  {
    sub_C7D6A0((__int64)v51, v6, 8);
    v52 = sub_AF1560(4 * v48 / 3u + 1);
    *(_DWORD *)(a1 + 200) = v52;
    if ( v52 )
    {
      v53 = (_DWORD *)sub_C7D670(32LL * v52, 8);
      v54 = *(unsigned int *)(a1 + 200);
      *(_QWORD *)(a1 + 192) = 0;
      *(_QWORD *)(a1 + 184) = v53;
      for ( i = &v53[8 * v54]; i != v53; v53 += 8 )
      {
        if ( v53 )
          *v53 = -1;
      }
      goto LABEL_14;
    }
LABEL_94:
    *(_QWORD *)(a1 + 184) = 0;
    *(_QWORD *)(a1 + 192) = 0;
    goto LABEL_14;
  }
  *(_QWORD *)(a1 + 192) = 0;
  v64 = &v51[8 * v48];
  do
  {
    if ( v51 )
      *v51 = -1;
    v51 += 8;
  }
  while ( v64 != v51 );
LABEL_14:
  v9 = *(_QWORD *)(a1 + 1000);
  v10 = *(unsigned int *)(a1 + 1008);
  *(_DWORD *)(a1 + 216) = 0;
  v11 = v9 + 8 * v10;
  v65 = v9;
  while ( v65 != v11 )
  {
    v12 = *(_QWORD *)(v11 - 8);
    v11 -= 8;
    if ( v12 )
    {
      sub_2DF5850(*(_QWORD *)(v12 + 456));
      v16 = *(_QWORD *)(v12 + 408);
      if ( v16 != v12 + 424 )
        _libc_free(v16);
      if ( *(_DWORD *)(v12 + 392) )
      {
        sub_2DF5350(v12 + 232, (__int64)sub_2DF57F0, 0, v13, v14, v15);
        *(_DWORD *)(v12 + 392) = 0;
        memset((void *)(v12 + 232), 0, 0xA0u);
        v34 = (_QWORD *)(v12 + 232);
        v35 = (_QWORD *)(v12 + 296);
        do
        {
          *v34 = 0;
          v34 += 2;
          *(v34 - 1) = 0;
        }
        while ( v35 != v34 );
        do
        {
          *v35 = 0;
          v35 += 3;
          *((_BYTE *)v35 - 16) = 0;
          *(v35 - 1) = 0;
        }
        while ( v35 != (_QWORD *)(v12 + 392) );
      }
      v17 = (unsigned __int64 *)(v12 + 368);
      *(_DWORD *)(v12 + 396) = 0;
      do
      {
        if ( *v17 )
          j_j___libc_free_0_0(*v17);
        v17 -= 3;
      }
      while ( v17 != (unsigned __int64 *)(v12 + 272) );
      v18 = *(_QWORD *)(v12 + 56);
      if ( v18 != v12 + 72 )
        _libc_free(v18);
      v19 = *(_QWORD *)(v12 + 32);
      if ( v19 )
        sub_B91220(v12 + 32, v19);
      j_j___libc_free_0(v12);
    }
  }
  v20 = *(_QWORD *)(a1 + 1080);
  v21 = *(unsigned int *)(a1 + 1088);
  *(_DWORD *)(a1 + 1008) = 0;
  for ( j = v20 + 8 * v21; v20 != j; j -= 8 )
  {
    v23 = *(_QWORD *)(j - 8);
    if ( v23 )
    {
      v24 = *(_QWORD *)(v23 + 8);
      if ( v24 )
        sub_B91220(v23 + 8, v24);
      j_j___libc_free_0(v23);
    }
  }
  v25 = *(_DWORD *)(a1 + 1128);
  ++*(_QWORD *)(a1 + 1112);
  *(_DWORD *)(a1 + 1088) = 0;
  if ( !v25 )
  {
    if ( !*(_DWORD *)(a1 + 1132) )
      goto LABEL_41;
    v26 = *(unsigned int *)(a1 + 1136);
    if ( (unsigned int)v26 <= 0x40 )
      goto LABEL_38;
    sub_C7D6A0(*(_QWORD *)(a1 + 1120), 16LL * (unsigned int)v26, 8);
    *(_DWORD *)(a1 + 1136) = 0;
LABEL_98:
    *(_QWORD *)(a1 + 1120) = 0;
LABEL_40:
    *(_QWORD *)(a1 + 1128) = 0;
    goto LABEL_41;
  }
  v41 = 4 * v25;
  v26 = *(unsigned int *)(a1 + 1136);
  if ( (unsigned int)(4 * v25) < 0x40 )
    v41 = 64;
  if ( v41 >= (unsigned int)v26 )
  {
LABEL_38:
    v27 = *(_DWORD **)(a1 + 1120);
    for ( k = &v27[4 * v26]; k != v27; v27 += 4 )
      *v27 = -1;
    goto LABEL_40;
  }
  v42 = v25 - 1;
  if ( v42 )
  {
    _BitScanReverse(&v42, v42);
    v43 = *(_DWORD **)(a1 + 1120);
    v44 = 1 << (33 - (v42 ^ 0x1F));
    if ( v44 < 64 )
      v44 = 64;
    if ( (_DWORD)v26 == v44 )
    {
      *(_QWORD *)(a1 + 1128) = 0;
      v45 = &v43[4 * (unsigned int)v26];
      do
      {
        if ( v43 )
          *v43 = -1;
        v43 += 4;
      }
      while ( v45 != v43 );
      goto LABEL_41;
    }
  }
  else
  {
    v43 = *(_DWORD **)(a1 + 1120);
    v44 = 64;
  }
  sub_C7D6A0((__int64)v43, 16LL * (unsigned int)v26, 8);
  v60 = sub_AF1560(4 * v44 / 3u + 1);
  *(_DWORD *)(a1 + 1136) = v60;
  if ( !v60 )
    goto LABEL_98;
  v61 = (_DWORD *)sub_C7D670(16LL * v60, 8);
  v62 = *(unsigned int *)(a1 + 1136);
  *(_QWORD *)(a1 + 1128) = 0;
  *(_QWORD *)(a1 + 1120) = v61;
  for ( m = &v61[4 * v62]; m != v61; v61 += 4 )
  {
    if ( v61 )
      *v61 = -1;
  }
LABEL_41:
  v29 = *(_DWORD *)(a1 + 1160);
  ++*(_QWORD *)(a1 + 1144);
  if ( v29 )
  {
    v36 = 4 * v29;
    v30 = *(unsigned int *)(a1 + 1168);
    if ( (unsigned int)(4 * v29) < 0x40 )
      v36 = 64;
    if ( v36 >= (unsigned int)v30 )
      goto LABEL_44;
    v37 = v29 - 1;
    if ( v37 )
    {
      _BitScanReverse(&v37, v37);
      v38 = *(_QWORD *)(a1 + 1152);
      v39 = 1 << (33 - (v37 ^ 0x1F));
      if ( v39 < 64 )
        v39 = 64;
      if ( (_DWORD)v30 == v39 )
      {
        *(_QWORD *)(a1 + 1160) = 0;
        v40 = v38 + 48 * v30;
        do
        {
          if ( v38 )
          {
            *(_QWORD *)v38 = 0;
            *(_BYTE *)(v38 + 24) = 0;
            *(_QWORD *)(v38 + 32) = 0;
          }
          v38 += 48;
        }
        while ( v40 != v38 );
        goto LABEL_47;
      }
    }
    else
    {
      v38 = *(_QWORD *)(a1 + 1152);
      v39 = 64;
    }
    sub_C7D6A0(v38, 48 * v30, 8);
    v56 = sub_AF1560(4 * v39 / 3u + 1);
    *(_DWORD *)(a1 + 1168) = v56;
    if ( !v56 )
      goto LABEL_96;
    v57 = sub_C7D670(48LL * v56, 8);
    v58 = *(unsigned int *)(a1 + 1168);
    *(_QWORD *)(a1 + 1160) = 0;
    *(_QWORD *)(a1 + 1152) = v57;
    for ( n = v57 + 48 * v58; n != v57; v57 += 48 )
    {
      if ( v57 )
      {
        *(_QWORD *)v57 = 0;
        *(_BYTE *)(v57 + 24) = 0;
        *(_QWORD *)(v57 + 32) = 0;
      }
    }
  }
  else if ( *(_DWORD *)(a1 + 1164) )
  {
    v30 = *(unsigned int *)(a1 + 1168);
    if ( (unsigned int)v30 <= 0x40 )
    {
LABEL_44:
      v31 = *(_QWORD **)(a1 + 1152);
      for ( ii = &v31[6 * v30]; ii != v31; *(v31 - 2) = 0 )
      {
        *v31 = 0;
        v31 += 6;
        *((_BYTE *)v31 - 24) = 0;
      }
      goto LABEL_46;
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 1152), 48 * v30, 8);
    *(_DWORD *)(a1 + 1168) = 0;
LABEL_96:
    *(_QWORD *)(a1 + 1152) = 0;
LABEL_46:
    *(_QWORD *)(a1 + 1160) = 0;
  }
LABEL_47:
  *(_WORD *)(a1 + 992) = 0;
  return 0;
}
