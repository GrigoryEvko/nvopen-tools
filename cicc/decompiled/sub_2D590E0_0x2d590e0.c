// Function: sub_2D590E0
// Address: 0x2d590e0
//
_QWORD **__fastcall sub_2D590E0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi
  __int64 v4; // r12
  unsigned __int64 v5; // rax
  __int64 v6; // r13
  _QWORD *v7; // r15
  unsigned int v8; // eax
  unsigned int v9; // eax
  unsigned int v10; // ecx
  __int64 v11; // rdx
  _QWORD *v12; // rax
  __int64 v13; // rdx
  _QWORD *i; // rdx
  char v15; // dl
  _QWORD **v16; // rax
  __int64 v17; // rcx
  _QWORD **v18; // r15
  _QWORD *v19; // r14
  _QWORD **v20; // r13
  _QWORD **result; // rax
  unsigned int v22; // eax
  __int64 v23; // rdx
  bool v24; // zf
  _QWORD *v25; // rax
  __int64 v26; // rdx
  _QWORD *j; // rdx
  unsigned int v28; // eax
  unsigned int v29; // r13d
  char v30; // al
  __int64 v31; // rdi
  __int64 v32; // rax
  _QWORD *v33; // rax
  __int64 v34; // rdx
  _QWORD *v35; // rdx
  __int64 v36; // [rsp+8h] [rbp-48h]
  unsigned __int64 v37[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = (_QWORD *)(a1 + 40);
  v4 = sub_ACADE0((__int64 **)a2);
  v5 = *(_QWORD *)(a1 + 840);
  if ( !v5 )
  {
    a2 = a1 + 840;
    sub_2D579F0((__int64)v3, (unsigned __int64 *)(a1 + 840));
    v5 = *(_QWORD *)(a1 + 840);
  }
  v6 = *(unsigned int *)(a1 + 48);
  v37[0] = v5;
  if ( v6 != v5 )
  {
    while ( 1 )
    {
      v7 = *(_QWORD **)(*v3 + 8 * v5);
      sub_BD84D0((__int64)v7, v4);
      sub_B43D60(v7);
      a2 = (__int64)v37;
      ++v37[0];
      sub_2D579F0(a1 + 40, v37);
      v5 = v37[0];
      if ( v6 == v37[0] )
        break;
      v3 = (_QWORD *)(a1 + 40);
    }
  }
  v8 = *(_DWORD *)(a1 + 320);
  ++*(_QWORD *)(a1 + 312);
  v9 = v8 >> 1;
  if ( v9 )
  {
    if ( (*(_BYTE *)(a1 + 320) & 1) == 0 )
    {
      v10 = 4 * v9;
      goto LABEL_10;
    }
LABEL_37:
    v12 = (_QWORD *)(a1 + 328);
    v13 = 64;
    goto LABEL_13;
  }
  if ( !*(_DWORD *)(a1 + 324) )
    goto LABEL_16;
  v10 = 0;
  if ( (*(_BYTE *)(a1 + 320) & 1) != 0 )
    goto LABEL_37;
LABEL_10:
  v11 = *(unsigned int *)(a1 + 336);
  if ( v10 >= (unsigned int)v11 || (unsigned int)v11 <= 0x40 )
  {
    v12 = *(_QWORD **)(a1 + 328);
    v13 = 2 * v11;
LABEL_13:
    for ( i = &v12[v13]; i != v12; v12 += 2 )
      *v12 = -4096;
    *(_QWORD *)(a1 + 320) &= 1uLL;
    goto LABEL_16;
  }
  if ( !v9 || (v28 = v9 - 1) == 0 )
  {
    a2 = 16LL * *(unsigned int *)(a1 + 336);
    sub_C7D6A0(*(_QWORD *)(a1 + 328), a2, 8);
    *(_BYTE *)(a1 + 320) |= 1u;
    goto LABEL_42;
  }
  _BitScanReverse(&v28, v28);
  v29 = 1 << (33 - (v28 ^ 0x1F));
  if ( v29 - 33 <= 0x1E )
  {
    v29 = 64;
    sub_C7D6A0(*(_QWORD *)(a1 + 328), 16LL * *(unsigned int *)(a1 + 336), 8);
    v30 = *(_BYTE *)(a1 + 320);
    v31 = 1024;
    goto LABEL_52;
  }
  if ( (_DWORD)v11 != v29 )
  {
    a2 = 16LL * *(unsigned int *)(a1 + 336);
    sub_C7D6A0(*(_QWORD *)(a1 + 328), a2, 8);
    v30 = *(_BYTE *)(a1 + 320) | 1;
    *(_BYTE *)(a1 + 320) = v30;
    if ( v29 <= 0x20 )
    {
LABEL_42:
      v24 = (*(_QWORD *)(a1 + 320) & 1LL) == 0;
      *(_QWORD *)(a1 + 320) &= 1uLL;
      if ( v24 )
      {
        v25 = *(_QWORD **)(a1 + 328);
        v26 = 2LL * *(unsigned int *)(a1 + 336);
      }
      else
      {
        v25 = (_QWORD *)(a1 + 328);
        v26 = 64;
      }
      for ( j = &v25[v26]; j != v25; v25 += 2 )
      {
        if ( v25 )
          *v25 = -4096;
      }
      goto LABEL_16;
    }
    v31 = 16LL * v29;
LABEL_52:
    a2 = 8;
    *(_BYTE *)(a1 + 320) = v30 & 0xFE;
    v32 = sub_C7D670(v31, 8);
    *(_DWORD *)(a1 + 336) = v29;
    *(_QWORD *)(a1 + 328) = v32;
    goto LABEL_42;
  }
  v24 = (*(_QWORD *)(a1 + 320) & 1LL) == 0;
  *(_QWORD *)(a1 + 320) &= 1uLL;
  if ( v24 )
  {
    v33 = *(_QWORD **)(a1 + 328);
    v34 = 2 * v11;
  }
  else
  {
    v33 = (_QWORD *)(a1 + 328);
    v34 = 64;
  }
  v35 = &v33[v34];
  do
  {
    if ( v33 )
      *v33 = -4096;
    v33 += 2;
  }
  while ( v35 != v33 );
LABEL_16:
  v15 = *(_BYTE *)(a1 + 876);
  *(_DWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 840) = 0;
  v16 = *(_QWORD ***)(a1 + 856);
  if ( v15 )
    v17 = *(unsigned int *)(a1 + 868);
  else
    v17 = *(unsigned int *)(a1 + 864);
  v18 = &v16[v17];
  if ( v16 == v18 )
  {
LABEL_21:
    result = (_QWORD **)(a1 + 848);
    ++*(_QWORD *)(a1 + 848);
    v36 = a1 + 848;
    if ( v15 )
    {
LABEL_22:
      *(_QWORD *)(a1 + 868) = 0;
      return result;
    }
  }
  else
  {
    while ( 1 )
    {
      v19 = *v16;
      v20 = v16;
      if ( (unsigned __int64)*v16 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v18 == ++v16 )
        goto LABEL_21;
    }
    result = (_QWORD **)(a1 + 848);
    v36 = a1 + 848;
    if ( v20 != v18 )
    {
      do
      {
        a2 = v4;
        sub_BD84D0((__int64)v19, v4);
        sub_B43D60(v19);
        result = v20 + 1;
        if ( v20 + 1 == v18 )
          break;
        while ( 1 )
        {
          v19 = *result;
          v20 = result;
          if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v18 == ++result )
            goto LABEL_27;
        }
      }
      while ( result != v18 );
LABEL_27:
      v15 = *(_BYTE *)(a1 + 876);
    }
    ++*(_QWORD *)(a1 + 848);
    if ( v15 )
      goto LABEL_22;
  }
  v22 = 4 * (*(_DWORD *)(a1 + 868) - *(_DWORD *)(a1 + 872));
  v23 = *(unsigned int *)(a1 + 864);
  if ( v22 < 0x20 )
    v22 = 32;
  if ( v22 >= (unsigned int)v23 )
  {
    result = (_QWORD **)memset(*(void **)(a1 + 856), -1, 8 * v23);
    goto LABEL_22;
  }
  return (_QWORD **)sub_C8C990(v36, a2);
}
