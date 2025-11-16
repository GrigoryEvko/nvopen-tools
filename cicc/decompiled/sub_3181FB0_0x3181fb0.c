// Function: sub_3181FB0
// Address: 0x3181fb0
//
__int64 __fastcall sub_3181FB0(__int64 a1)
{
  int v2; // edx
  unsigned int v3; // r8d
  __int64 *v4; // rdi
  __int64 v5; // rsi
  __int64 v6; // rsi
  __int64 *v7; // rax
  __int64 *v9; // r13
  __int64 *v10; // rax
  __int64 v11; // r12
  __int64 *v12; // rbx
  __int64 v13; // rax
  unsigned int v14; // eax
  unsigned int v15; // eax
  int v16; // ebx
  __int64 *v17; // rax
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // rax
  __int64 *v20; // rax
  __int64 v21; // rsi
  __int64 *i; // rdx
  char *v23; // [rsp+8h] [rbp-58h]
  _BYTE v24[16]; // [rsp+10h] [rbp-50h] BYREF
  void (__fastcall *v25)(_BYTE *, _BYTE *, __int64); // [rsp+20h] [rbp-40h]

  v2 = *(_DWORD *)(a1 + 16);
  v3 = *(_DWORD *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
  if ( !v2 )
  {
    ++*(_QWORD *)a1;
LABEL_3:
    v5 = v3;
    if ( !*(_DWORD *)(a1 + 20) )
    {
      v6 = 2LL * v3;
      return sub_C7D6A0((__int64)v4, v6 * 8, 8);
    }
    if ( v3 > 0x40 )
    {
      sub_C7D6A0((__int64)v4, 16LL * v3, 8);
      *(_QWORD *)(a1 + 8) = 0;
      v4 = 0;
      v6 = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return sub_C7D6A0((__int64)v4, v6 * 8, 8);
    }
    goto LABEL_5;
  }
  v5 = v3;
  v9 = &v4[2 * v3];
  if ( v9 == v4 )
    goto LABEL_15;
  v10 = v4;
  while ( 1 )
  {
    v11 = *v10;
    v12 = v10;
    if ( *v10 != -4096 && v11 != -8192 )
      break;
    v10 += 2;
    if ( v9 == v10 )
      goto LABEL_15;
  }
  if ( v9 == v10 )
  {
LABEL_15:
    ++*(_QWORD *)a1;
  }
  else
  {
    while ( 1 )
    {
      if ( *(_BYTE *)(a1 + 32) )
      {
        v13 = v12[1];
        if ( *(_BYTE *)v13 == 85 )
          *(_WORD *)(v13 + 2) |= 3u;
      }
      if ( *(_QWORD *)(v11 + 16) )
      {
        sub_BD84D0(v11, *(_QWORD *)(v11 - 32LL * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF)));
        sub_B43D60((_QWORD *)v11);
      }
      else
      {
        v23 = *(char **)(v11 - 32LL * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF));
        sub_B43D60((_QWORD *)v11);
        v25 = 0;
        sub_F5CAB0(v23, 0, 0, (__int64)v24);
        if ( v25 )
          v25(v24, v24, 3);
      }
      v12 += 2;
      if ( v12 == v9 )
        break;
      while ( *v12 == -4096 || *v12 == -8192 )
      {
        v12 += 2;
        if ( v9 == v12 )
          goto LABEL_26;
      }
      if ( v9 == v12 )
        break;
      v11 = *v12;
    }
LABEL_26:
    v2 = *(_DWORD *)(a1 + 16);
    ++*(_QWORD *)a1;
    v4 = *(__int64 **)(a1 + 8);
    v3 = *(_DWORD *)(a1 + 24);
    if ( !v2 )
      goto LABEL_3;
    v5 = v3;
  }
  v14 = 4 * v2;
  if ( (unsigned int)(4 * v2) < 0x40 )
    v14 = 64;
  if ( v14 >= v3 )
  {
LABEL_5:
    v6 = 2 * v5;
    v7 = &v4[v6];
    if ( v4 != &v4[v6] )
    {
      do
      {
        *v4 = -4096;
        v4 += 2;
      }
      while ( v7 != v4 );
      v4 = *(__int64 **)(a1 + 8);
      v6 = 2LL * *(unsigned int *)(a1 + 24);
    }
    *(_QWORD *)(a1 + 16) = 0;
    return sub_C7D6A0((__int64)v4, v6 * 8, 8);
  }
  if ( v2 == 1 )
  {
    v16 = 64;
  }
  else
  {
    _BitScanReverse(&v15, v2 - 1);
    v16 = 1 << (33 - (v15 ^ 0x1F));
    if ( v16 < 64 )
      v16 = 64;
    if ( v3 == v16 )
    {
      *(_QWORD *)(a1 + 16) = 0;
      v17 = &v4[2 * v5];
      do
      {
        if ( v4 )
          *v4 = -4096;
        v4 += 2;
      }
      while ( v17 != v4 );
      v4 = *(__int64 **)(a1 + 8);
      v6 = 2LL * *(unsigned int *)(a1 + 24);
      return sub_C7D6A0((__int64)v4, v6 * 8, 8);
    }
  }
  sub_C7D6A0((__int64)v4, 16 * v5, 8);
  v18 = ((((((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
           | (4 * v16 / 3u + 1)
           | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
         | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
         | (4 * v16 / 3u + 1)
         | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
         | (4 * v16 / 3u + 1)
         | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
       | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
       | (4 * v16 / 3u + 1)
       | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 16;
  v19 = (v18
       | (((((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
           | (4 * v16 / 3u + 1)
           | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
         | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
         | (4 * v16 / 3u + 1)
         | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
         | (4 * v16 / 3u + 1)
         | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 4)
       | (((4 * v16 / 3u + 1) | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1)) >> 2)
       | (4 * v16 / 3u + 1)
       | ((unsigned __int64)(4 * v16 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 24) = v19;
  v20 = (__int64 *)sub_C7D670(16 * v19, 8);
  v21 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = v20;
  v4 = v20;
  v6 = 2 * v21;
  for ( i = &v20[v6]; i != v20; v20 += 2 )
  {
    if ( v20 )
      *v20 = -4096;
  }
  return sub_C7D6A0((__int64)v4, v6 * 8, 8);
}
