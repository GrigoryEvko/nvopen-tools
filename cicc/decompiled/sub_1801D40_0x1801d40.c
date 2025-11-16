// Function: sub_1801D40
// Address: 0x1801d40
//
__int64 __fastcall sub_1801D40(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  unsigned int v7; // ecx
  _QWORD *v8; // rdi
  unsigned int v9; // eax
  int v10; // eax
  unsigned __int64 v11; // rax
  __int64 v12; // rax
  int v13; // r13d
  __int64 v14; // r12
  _QWORD *v15; // rax
  __int64 v16; // rcx
  _QWORD *j; // rdx
  _QWORD *v18; // rax

  v2 = *(_DWORD *)(a1 + 752);
  ++*(_QWORD *)(a1 + 736);
  *(_BYTE *)(a1 + 728) = 0;
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 756) )
      return 0;
    v3 = *(unsigned int *)(a1 + 760);
    if ( (unsigned int)v3 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 744));
      *(_QWORD *)(a1 + 744) = 0;
      *(_QWORD *)(a1 + 752) = 0;
      *(_DWORD *)(a1 + 760) = 0;
      return 0;
    }
    goto LABEL_4;
  }
  v7 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 760);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v7 = 64;
  if ( (unsigned int)v3 <= v7 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 744);
    for ( i = &v4[7 * v3]; i != v4; v4 += 7 )
      *v4 = -8;
    *(_QWORD *)(a1 + 752) = 0;
    return 0;
  }
  v8 = *(_QWORD **)(a1 + 744);
  v9 = v2 - 1;
  if ( !v9 )
  {
    v14 = 7168;
    v13 = 128;
LABEL_16:
    j___libc_free_0(v8);
    *(_DWORD *)(a1 + 760) = v13;
    v15 = (_QWORD *)sub_22077B0(v14);
    v16 = *(unsigned int *)(a1 + 760);
    *(_QWORD *)(a1 + 752) = 0;
    *(_QWORD *)(a1 + 744) = v15;
    for ( j = &v15[7 * v16]; j != v15; v15 += 7 )
    {
      if ( v15 )
        *v15 = -8;
    }
    return 0;
  }
  _BitScanReverse(&v9, v9);
  v10 = 1 << (33 - (v9 ^ 0x1F));
  if ( v10 < 64 )
    v10 = 64;
  if ( (_DWORD)v3 != v10 )
  {
    v11 = (4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1);
    v12 = ((((v11 >> 2) | v11 | (((v11 >> 2) | v11) >> 4)) >> 8)
         | (v11 >> 2)
         | v11
         | (((v11 >> 2) | v11) >> 4)
         | (((((v11 >> 2) | v11 | (((v11 >> 2) | v11) >> 4)) >> 8) | (v11 >> 2) | v11 | (((v11 >> 2) | v11) >> 4)) >> 16))
        + 1;
    v13 = v12;
    v14 = 56 * v12;
    goto LABEL_16;
  }
  *(_QWORD *)(a1 + 752) = 0;
  v18 = &v8[7 * v3];
  do
  {
    if ( v8 )
      *v8 = -8;
    v8 += 7;
  }
  while ( v18 != v8 );
  return 0;
}
