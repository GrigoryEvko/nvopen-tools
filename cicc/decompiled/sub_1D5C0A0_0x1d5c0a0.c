// Function: sub_1D5C0A0
// Address: 0x1d5c0a0
//
char *__fastcall sub_1D5C0A0(
        __int64 a1,
        __int64 **a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rax
  double v12; // xmm4_8
  double v13; // xmm5_8
  _QWORD **v14; // r13
  __int64 v15; // r12
  _QWORD **v16; // r14
  _QWORD *v17; // r15
  unsigned int v18; // eax
  unsigned int v19; // eax
  unsigned int v20; // ecx
  __int64 v21; // rdx
  _QWORD *v22; // rax
  __int64 v23; // rdx
  _QWORD *i; // rdx
  char *v25; // rdi
  char *v26; // rcx
  __int64 v27; // rax
  char *v28; // r15
  char *v29; // rax
  _QWORD *v30; // r14
  char *v31; // r13
  char *result; // rax
  unsigned int v33; // eax
  __int64 v34; // rdx
  bool v35; // zf
  _QWORD *v36; // rax
  __int64 v37; // rdx
  _QWORD *j; // rdx
  unsigned int v39; // eax
  unsigned int v40; // r13d
  char v41; // al
  __int64 v42; // rdi
  __int64 v43; // rax
  _QWORD *v44; // rax
  __int64 v45; // rdx
  _QWORD *v46; // rdx
  __int64 v47; // [rsp+8h] [rbp-38h]

  v11 = sub_1599EF0(a2);
  v14 = *(_QWORD ***)(a1 + 312);
  v15 = v11;
  v16 = &v14[*(unsigned int *)(a1 + 320)];
  while ( v16 != v14 )
  {
    v17 = *v14++;
    sub_164D160((__int64)v17, v15, a3, a4, a5, a6, v12, v13, a9, a10);
    sub_15F20C0(v17);
  }
  v18 = *(_DWORD *)(a1 + 48);
  ++*(_QWORD *)(a1 + 40);
  v19 = v18 >> 1;
  if ( v19 )
  {
    if ( (*(_BYTE *)(a1 + 48) & 1) == 0 )
    {
      v20 = 4 * v19;
      goto LABEL_6;
    }
LABEL_34:
    v22 = (_QWORD *)(a1 + 56);
    v23 = 32;
    goto LABEL_9;
  }
  if ( !*(_DWORD *)(a1 + 52) )
    goto LABEL_12;
  v20 = 0;
  if ( (*(_BYTE *)(a1 + 48) & 1) != 0 )
    goto LABEL_34;
LABEL_6:
  v21 = *(unsigned int *)(a1 + 64);
  if ( (unsigned int)v21 <= v20 || (unsigned int)v21 <= 0x40 )
  {
    v22 = *(_QWORD **)(a1 + 56);
    v23 = v21;
LABEL_9:
    for ( i = &v22[v23]; i != v22; ++v22 )
      *v22 = -8;
    *(_QWORD *)(a1 + 48) &= 1uLL;
    goto LABEL_12;
  }
  if ( !v19 || (v39 = v19 - 1) == 0 )
  {
    j___libc_free_0(*(_QWORD *)(a1 + 56));
    *(_BYTE *)(a1 + 48) |= 1u;
    goto LABEL_39;
  }
  _BitScanReverse(&v39, v39);
  v40 = 1 << (33 - (v39 ^ 0x1F));
  if ( v40 - 33 <= 0x1E )
  {
    v40 = 64;
    j___libc_free_0(*(_QWORD *)(a1 + 56));
    v41 = *(_BYTE *)(a1 + 48);
    v42 = 512;
    goto LABEL_51;
  }
  if ( (_DWORD)v21 != v40 )
  {
    j___libc_free_0(*(_QWORD *)(a1 + 56));
    v41 = *(_BYTE *)(a1 + 48) | 1;
    *(_BYTE *)(a1 + 48) = v41;
    if ( v40 <= 0x20 )
      goto LABEL_39;
    v42 = 8LL * v40;
LABEL_51:
    *(_BYTE *)(a1 + 48) = v41 & 0xFE;
    v43 = sub_22077B0(v42);
    *(_DWORD *)(a1 + 64) = v40;
    *(_QWORD *)(a1 + 56) = v43;
LABEL_39:
    v35 = (*(_QWORD *)(a1 + 48) & 1LL) == 0;
    *(_QWORD *)(a1 + 48) &= 1uLL;
    if ( v35 )
    {
      v36 = *(_QWORD **)(a1 + 56);
      v37 = *(unsigned int *)(a1 + 64);
    }
    else
    {
      v36 = (_QWORD *)(a1 + 56);
      v37 = 32;
    }
    for ( j = &v36[v37]; j != v36; ++v36 )
    {
      if ( v36 )
        *v36 = -8;
    }
    goto LABEL_12;
  }
  v35 = (*(_QWORD *)(a1 + 48) & 1LL) == 0;
  *(_QWORD *)(a1 + 48) &= 1uLL;
  if ( v35 )
  {
    v44 = *(_QWORD **)(a1 + 56);
    v45 = v21;
  }
  else
  {
    v44 = (_QWORD *)(a1 + 56);
    v45 = 32;
  }
  v46 = &v44[v45];
  do
  {
    if ( v44 )
      *v44 = -8;
    ++v44;
  }
  while ( v46 != v44 );
LABEL_12:
  v25 = *(char **)(a1 + 600);
  v26 = *(char **)(a1 + 592);
  *(_DWORD *)(a1 + 320) = 0;
  if ( v25 == v26 )
    v27 = *(unsigned int *)(a1 + 612);
  else
    v27 = *(unsigned int *)(a1 + 608);
  v28 = &v25[8 * v27];
  if ( v25 == v28 )
  {
LABEL_18:
    result = (char *)(a1 + 584);
    v47 = a1 + 584;
  }
  else
  {
    v29 = v25;
    while ( 1 )
    {
      v30 = *(_QWORD **)v29;
      v31 = v29;
      if ( *(_QWORD *)v29 < 0xFFFFFFFFFFFFFFFELL )
        break;
      v29 += 8;
      if ( v28 == v29 )
        goto LABEL_18;
    }
    result = (char *)(a1 + 584);
    v47 = a1 + 584;
    if ( v28 != v31 )
    {
      do
      {
        sub_164D160((__int64)v30, v15, a3, a4, a5, a6, v12, v13, a9, a10);
        sub_15F20C0(v30);
        result = v31 + 8;
        if ( v31 + 8 == v28 )
          break;
        while ( 1 )
        {
          v30 = *(_QWORD **)result;
          v31 = result;
          if ( *(_QWORD *)result < 0xFFFFFFFFFFFFFFFELL )
            break;
          result += 8;
          if ( v28 == result )
            goto LABEL_23;
        }
      }
      while ( v28 != result );
LABEL_23:
      v25 = *(char **)(a1 + 600);
      v26 = *(char **)(a1 + 592);
    }
  }
  ++*(_QWORD *)(a1 + 584);
  if ( v26 == v25 )
    goto LABEL_29;
  v33 = 4 * (*(_DWORD *)(a1 + 612) - *(_DWORD *)(a1 + 616));
  v34 = *(unsigned int *)(a1 + 608);
  if ( v33 < 0x20 )
    v33 = 32;
  if ( (unsigned int)v34 <= v33 )
  {
    result = (char *)memset(v25, -1, 8 * v34);
LABEL_29:
    *(_QWORD *)(a1 + 612) = 0;
    return result;
  }
  return (char *)sub_16CC920(v47);
}
