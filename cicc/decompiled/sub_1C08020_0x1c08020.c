// Function: sub_1C08020
// Address: 0x1c08020
//
__int64 __fastcall sub_1C08020(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // rax
  int v6; // eax
  __int64 v7; // rdx
  _QWORD *v8; // rax
  _QWORD *i; // rdx
  int v10; // eax
  __int64 result; // rax
  __int64 v12; // rdx
  __int64 k; // rdx
  unsigned int v14; // ecx
  _QWORD *v15; // rdi
  unsigned int v16; // eax
  int v17; // eax
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rax
  int v20; // r13d
  __int64 v21; // r12
  __int64 v22; // rdx
  __int64 m; // rdx
  unsigned int v24; // ecx
  _QWORD *v25; // rdi
  unsigned int v26; // eax
  int v27; // eax
  unsigned __int64 v28; // rax
  __int64 v29; // rax
  int v30; // r13d
  __int64 v31; // r12
  _QWORD *v32; // rax
  __int64 v33; // rdx
  _QWORD *j; // rdx
  _QWORD *v35; // rax

  v4 = *(_QWORD *)(a1 + 104);
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 112) = a3;
  if ( v4 )
  {
    sub_1C07C70(v4);
    j_j___libc_free_0(v4, 376);
  }
  if ( a2 )
  {
    v5 = sub_22077B0(376);
    if ( v5 )
    {
      *(_QWORD *)(v5 + 8) = 0;
      *(_QWORD *)(v5 + 16) = 0;
      *(_QWORD *)(v5 + 24) = 0;
      *(_DWORD *)(v5 + 32) = 0;
      *(_QWORD *)(v5 + 40) = 0;
      *(_QWORD *)(v5 + 48) = 0;
      *(_QWORD *)(v5 + 56) = 0;
      *(_DWORD *)(v5 + 64) = 0;
      *(_QWORD *)(v5 + 72) = 0;
      *(_QWORD *)(v5 + 80) = 0;
      *(_QWORD *)(v5 + 88) = 0;
      *(_DWORD *)(v5 + 96) = 0;
      *(_QWORD *)(v5 + 104) = v5 + 120;
      *(_QWORD *)(v5 + 112) = 0x2000000000LL;
      *(_DWORD *)v5 = 0;
    }
    *(_QWORD *)(a1 + 104) = v5;
  }
  else
  {
    *(_QWORD *)(a1 + 104) = 0;
  }
  v6 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)(a1 + 8);
  if ( !v6 )
  {
    if ( !*(_DWORD *)(a1 + 28) )
      goto LABEL_13;
    v7 = *(unsigned int *)(a1 + 32);
    if ( (unsigned int)v7 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 16));
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_DWORD *)(a1 + 32) = 0;
      goto LABEL_13;
    }
    goto LABEL_10;
  }
  v24 = 4 * v6;
  v7 = *(unsigned int *)(a1 + 32);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v24 = 64;
  if ( v24 >= (unsigned int)v7 )
  {
LABEL_10:
    v8 = *(_QWORD **)(a1 + 16);
    for ( i = &v8[3 * v7]; i != v8; v8 += 3 )
      *v8 = -8;
    *(_QWORD *)(a1 + 24) = 0;
    goto LABEL_13;
  }
  v25 = *(_QWORD **)(a1 + 16);
  v26 = v6 - 1;
  if ( !v26 )
  {
    v31 = 3072;
    v30 = 128;
LABEL_41:
    j___libc_free_0(v25);
    *(_DWORD *)(a1 + 32) = v30;
    v32 = (_QWORD *)sub_22077B0(v31);
    v33 = *(unsigned int *)(a1 + 32);
    *(_QWORD *)(a1 + 24) = 0;
    *(_QWORD *)(a1 + 16) = v32;
    for ( j = &v32[3 * v33]; j != v32; v32 += 3 )
    {
      if ( v32 )
        *v32 = -8;
    }
    goto LABEL_13;
  }
  _BitScanReverse(&v26, v26);
  v27 = 1 << (33 - (v26 ^ 0x1F));
  if ( v27 < 64 )
    v27 = 64;
  if ( (_DWORD)v7 != v27 )
  {
    v28 = (4 * v27 / 3u + 1) | ((unsigned __int64)(4 * v27 / 3u + 1) >> 1);
    v29 = ((((v28 >> 2) | v28 | (((v28 >> 2) | v28) >> 4)) >> 8)
         | (v28 >> 2)
         | v28
         | (((v28 >> 2) | v28) >> 4)
         | (((((v28 >> 2) | v28 | (((v28 >> 2) | v28) >> 4)) >> 8) | (v28 >> 2) | v28 | (((v28 >> 2) | v28) >> 4)) >> 16))
        + 1;
    v30 = v29;
    v31 = 24 * v29;
    goto LABEL_41;
  }
  *(_QWORD *)(a1 + 24) = 0;
  v35 = &v25[3 * v7];
  do
  {
    if ( v25 )
      *v25 = -8;
    v25 += 3;
  }
  while ( v35 != v25 );
LABEL_13:
  v10 = *(_DWORD *)(a1 + 56);
  ++*(_QWORD *)(a1 + 40);
  if ( !v10 )
  {
    result = *(unsigned int *)(a1 + 60);
    if ( !(_DWORD)result )
      goto LABEL_19;
    v12 = *(unsigned int *)(a1 + 64);
    if ( (unsigned int)v12 > 0x40 )
    {
      result = j___libc_free_0(*(_QWORD *)(a1 + 48));
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 56) = 0;
      *(_DWORD *)(a1 + 64) = 0;
      goto LABEL_19;
    }
    goto LABEL_16;
  }
  v14 = 4 * v10;
  v12 = *(unsigned int *)(a1 + 64);
  if ( (unsigned int)(4 * v10) < 0x40 )
    v14 = 64;
  if ( (unsigned int)v12 <= v14 )
  {
LABEL_16:
    result = *(_QWORD *)(a1 + 48);
    for ( k = result + 8 * v12; k != result; result += 8 )
      *(_QWORD *)result = -8;
    *(_QWORD *)(a1 + 56) = 0;
    goto LABEL_19;
  }
  v15 = *(_QWORD **)(a1 + 48);
  v16 = v10 - 1;
  if ( !v16 )
  {
    v21 = 1024;
    v20 = 128;
LABEL_28:
    j___libc_free_0(v15);
    *(_DWORD *)(a1 + 64) = v20;
    result = sub_22077B0(v21);
    v22 = *(unsigned int *)(a1 + 64);
    *(_QWORD *)(a1 + 56) = 0;
    *(_QWORD *)(a1 + 48) = result;
    for ( m = result + 8 * v22; m != result; result += 8 )
    {
      if ( result )
        *(_QWORD *)result = -8;
    }
    goto LABEL_19;
  }
  _BitScanReverse(&v16, v16);
  v17 = 1 << (33 - (v16 ^ 0x1F));
  if ( v17 < 64 )
    v17 = 64;
  if ( (_DWORD)v12 != v17 )
  {
    v18 = (((4 * v17 / 3u + 1) | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 2)
        | (4 * v17 / 3u + 1)
        | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)
        | (((((4 * v17 / 3u + 1) | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 2)
          | (4 * v17 / 3u + 1)
          | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 4);
    v19 = (v18 >> 8) | v18;
    v20 = (v19 | (v19 >> 16)) + 1;
    v21 = 8 * ((v19 | (v19 >> 16)) + 1);
    goto LABEL_28;
  }
  *(_QWORD *)(a1 + 56) = 0;
  result = (__int64)&v15[v12];
  do
  {
    if ( v15 )
      *v15 = -8;
    ++v15;
  }
  while ( (_QWORD *)result != v15 );
LABEL_19:
  *(_DWORD *)(a1 + 128) = 7;
  return result;
}
