// Function: sub_190DF50
// Address: 0x190df50
//
__int64 __fastcall sub_190DF50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  int v7; // eax
  __int64 v8; // rdx
  _DWORD *v9; // rax
  _DWORD *i; // rdx
  int v11; // eax
  __int64 v12; // rdx
  _QWORD *v13; // rax
  _QWORD *k; // rdx
  unsigned __int64 *v15; // rbx
  unsigned __int64 *v16; // r13
  unsigned __int64 v17; // rdi
  __int64 v18; // rax
  int v19; // eax
  __int64 result; // rax
  __int64 v21; // rdx
  __int64 n; // rdx
  unsigned int v23; // ecx
  _QWORD *v24; // rdi
  unsigned int v25; // eax
  int v26; // eax
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rax
  int v29; // ebx
  __int64 v30; // r13
  __int64 v31; // rdx
  __int64 ii; // rdx
  unsigned int v33; // ecx
  _QWORD *v34; // rdi
  unsigned int v35; // eax
  int v36; // eax
  unsigned __int64 v37; // r13
  unsigned int v38; // eax
  _QWORD *v39; // rax
  __int64 v40; // rdx
  _QWORD *m; // rdx
  unsigned int v42; // ecx
  _DWORD *v43; // rdi
  unsigned int v44; // eax
  int v45; // eax
  unsigned __int64 v46; // rax
  __int64 v47; // rax
  int v48; // ebx
  __int64 v49; // r13
  _DWORD *v50; // rax
  __int64 v51; // rdx
  _DWORD *j; // rdx
  _QWORD *v53; // rbx
  __int64 v54; // rdx
  unsigned __int64 *v55; // r13
  unsigned __int64 *v56; // rbx
  unsigned __int64 v57; // rdi
  _QWORD *v58; // rax
  _DWORD *v59; // rax

  sub_190D990(a1 + 152, a2, a3, a4, a5, a6);
  v7 = *(_DWORD *)(a1 + 392);
  ++*(_QWORD *)(a1 + 376);
  if ( !v7 )
  {
    if ( !*(_DWORD *)(a1 + 396) )
      goto LABEL_7;
    v8 = *(unsigned int *)(a1 + 400);
    if ( (unsigned int)v8 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 384));
      *(_QWORD *)(a1 + 384) = 0;
      *(_QWORD *)(a1 + 392) = 0;
      *(_DWORD *)(a1 + 400) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v42 = 4 * v7;
  v8 = *(unsigned int *)(a1 + 400);
  if ( (unsigned int)(4 * v7) < 0x40 )
    v42 = 64;
  if ( v42 >= (unsigned int)v8 )
  {
LABEL_4:
    v9 = *(_DWORD **)(a1 + 384);
    for ( i = &v9[10 * v8]; i != v9; v9 += 10 )
      *v9 = -1;
    *(_QWORD *)(a1 + 392) = 0;
    goto LABEL_7;
  }
  v43 = *(_DWORD **)(a1 + 384);
  v44 = v7 - 1;
  if ( !v44 )
  {
    v49 = 5120;
    v48 = 128;
LABEL_58:
    j___libc_free_0(v43);
    *(_DWORD *)(a1 + 400) = v48;
    v50 = (_DWORD *)sub_22077B0(v49);
    v51 = *(unsigned int *)(a1 + 400);
    *(_QWORD *)(a1 + 392) = 0;
    *(_QWORD *)(a1 + 384) = v50;
    for ( j = &v50[10 * v51]; j != v50; v50 += 10 )
    {
      if ( v50 )
        *v50 = -1;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v44, v44);
  v45 = 1 << (33 - (v44 ^ 0x1F));
  if ( v45 < 64 )
    v45 = 64;
  if ( (_DWORD)v8 != v45 )
  {
    v46 = (4 * v45 / 3u + 1) | ((unsigned __int64)(4 * v45 / 3u + 1) >> 1);
    v47 = ((((v46 >> 2) | v46 | (((v46 >> 2) | v46) >> 4)) >> 8)
         | (v46 >> 2)
         | v46
         | (((v46 >> 2) | v46) >> 4)
         | (((((v46 >> 2) | v46 | (((v46 >> 2) | v46) >> 4)) >> 8) | (v46 >> 2) | v46 | (((v46 >> 2) | v46) >> 4)) >> 16))
        + 1;
    v48 = v47;
    v49 = 40 * v47;
    goto LABEL_58;
  }
  *(_QWORD *)(a1 + 392) = 0;
  v59 = &v43[10 * v8];
  do
  {
    if ( v43 )
      *v43 = -1;
    v43 += 10;
  }
  while ( v59 != v43 );
LABEL_7:
  v11 = *(_DWORD *)(a1 + 768);
  ++*(_QWORD *)(a1 + 752);
  if ( !v11 )
  {
    if ( !*(_DWORD *)(a1 + 772) )
      goto LABEL_13;
    v12 = *(unsigned int *)(a1 + 776);
    if ( (unsigned int)v12 <= 0x40 )
      goto LABEL_10;
    j___libc_free_0(*(_QWORD *)(a1 + 760));
    *(_DWORD *)(a1 + 776) = 0;
LABEL_69:
    *(_QWORD *)(a1 + 760) = 0;
LABEL_12:
    *(_QWORD *)(a1 + 768) = 0;
    goto LABEL_13;
  }
  v33 = 4 * v11;
  v12 = *(unsigned int *)(a1 + 776);
  if ( (unsigned int)(4 * v11) < 0x40 )
    v33 = 64;
  if ( (unsigned int)v12 <= v33 )
  {
LABEL_10:
    v13 = *(_QWORD **)(a1 + 760);
    for ( k = &v13[2 * v12]; k != v13; v13 += 2 )
      *v13 = -8;
    goto LABEL_12;
  }
  v34 = *(_QWORD **)(a1 + 760);
  v35 = v11 - 1;
  if ( v35 )
  {
    _BitScanReverse(&v35, v35);
    v36 = 1 << (33 - (v35 ^ 0x1F));
    if ( v36 < 64 )
      v36 = 64;
    if ( (_DWORD)v12 == v36 )
    {
      *(_QWORD *)(a1 + 768) = 0;
      v58 = &v34[2 * (unsigned int)v12];
      do
      {
        if ( v34 )
          *v34 = -8;
        v34 += 2;
      }
      while ( v58 != v34 );
      goto LABEL_13;
    }
    v37 = 4 * v36 / 3u + 1;
  }
  else
  {
    v37 = 86;
  }
  j___libc_free_0(v34);
  v38 = sub_1454B60(v37);
  *(_DWORD *)(a1 + 776) = v38;
  if ( !v38 )
    goto LABEL_69;
  v39 = (_QWORD *)sub_22077B0(16LL * v38);
  v40 = *(unsigned int *)(a1 + 776);
  *(_QWORD *)(a1 + 768) = 0;
  *(_QWORD *)(a1 + 760) = v39;
  for ( m = &v39[2 * v40]; m != v39; v39 += 2 )
  {
    if ( v39 )
      *v39 = -8;
  }
LABEL_13:
  v15 = *(unsigned __int64 **)(a1 + 472);
  v16 = &v15[2 * *(unsigned int *)(a1 + 480)];
  while ( v16 != v15 )
  {
    v17 = *v15;
    v15 += 2;
    _libc_free(v17);
  }
  v18 = *(unsigned int *)(a1 + 432);
  *(_DWORD *)(a1 + 480) = 0;
  if ( (_DWORD)v18 )
  {
    v53 = *(_QWORD **)(a1 + 424);
    *(_QWORD *)(a1 + 488) = 0;
    v54 = *v53;
    v55 = &v53[v18];
    v56 = v53 + 1;
    *(_QWORD *)(a1 + 408) = v54;
    *(_QWORD *)(a1 + 416) = v54 + 4096;
    while ( v55 != v56 )
    {
      v57 = *v56++;
      _libc_free(v57);
    }
    *(_DWORD *)(a1 + 432) = 1;
  }
  v19 = *(_DWORD *)(a1 + 128);
  ++*(_QWORD *)(a1 + 112);
  if ( !v19 )
  {
    result = *(unsigned int *)(a1 + 132);
    if ( !(_DWORD)result )
      goto LABEL_22;
    v21 = *(unsigned int *)(a1 + 136);
    if ( (unsigned int)v21 > 0x40 )
    {
      result = j___libc_free_0(*(_QWORD *)(a1 + 120));
      *(_QWORD *)(a1 + 120) = 0;
      *(_QWORD *)(a1 + 128) = 0;
      *(_DWORD *)(a1 + 136) = 0;
      goto LABEL_22;
    }
    goto LABEL_19;
  }
  v23 = 4 * v19;
  v21 = *(unsigned int *)(a1 + 136);
  if ( (unsigned int)(4 * v19) < 0x40 )
    v23 = 64;
  if ( (unsigned int)v21 <= v23 )
  {
LABEL_19:
    result = *(_QWORD *)(a1 + 120);
    for ( n = result + 16 * v21; n != result; result += 16 )
      *(_QWORD *)result = -8;
    *(_QWORD *)(a1 + 128) = 0;
    goto LABEL_22;
  }
  v24 = *(_QWORD **)(a1 + 120);
  v25 = v19 - 1;
  if ( !v25 )
  {
    v30 = 2048;
    v29 = 128;
LABEL_31:
    j___libc_free_0(v24);
    *(_DWORD *)(a1 + 136) = v29;
    result = sub_22077B0(v30);
    v31 = *(unsigned int *)(a1 + 136);
    *(_QWORD *)(a1 + 128) = 0;
    *(_QWORD *)(a1 + 120) = result;
    for ( ii = result + 16 * v31; ii != result; result += 16 )
    {
      if ( result )
        *(_QWORD *)result = -8;
    }
    goto LABEL_22;
  }
  _BitScanReverse(&v25, v25);
  v26 = 1 << (33 - (v25 ^ 0x1F));
  if ( v26 < 64 )
    v26 = 64;
  if ( (_DWORD)v21 != v26 )
  {
    v27 = (((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
        | (4 * v26 / 3u + 1)
        | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)
        | (((((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
          | (4 * v26 / 3u + 1)
          | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 4);
    v28 = (v27 >> 8) | v27;
    v29 = (v28 | (v28 >> 16)) + 1;
    v30 = 16 * ((v28 | (v28 >> 16)) + 1);
    goto LABEL_31;
  }
  *(_QWORD *)(a1 + 128) = 0;
  result = (__int64)&v24[2 * (unsigned int)v21];
  do
  {
    if ( v24 )
      *v24 = -8;
    v24 += 2;
  }
  while ( (_QWORD *)result != v24 );
LABEL_22:
  *(_BYTE *)(a1 + 784) = 1;
  return result;
}
