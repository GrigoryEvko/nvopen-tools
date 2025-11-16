// Function: sub_E34680
// Address: 0xe34680
//
__int64 __fastcall sub_E34680(__int64 a1)
{
  int v1; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  __int64 result; // rax
  unsigned int v7; // ecx
  unsigned int v8; // eax
  _QWORD *v9; // rdi
  int v10; // r12d
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rdi
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *j; // rdx
  _QWORD *v16; // rax

  v1 = *(_DWORD *)(a1 + 176);
  ++*(_QWORD *)(a1 + 160);
  if ( !v1 )
  {
    if ( !*(_DWORD *)(a1 + 180) )
      goto LABEL_7;
    v3 = *(unsigned int *)(a1 + 184);
    if ( (unsigned int)v3 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 168), 16LL * (unsigned int)v3, 8);
      *(_QWORD *)(a1 + 168) = 0;
      *(_QWORD *)(a1 + 176) = 0;
      *(_DWORD *)(a1 + 184) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v7 = 4 * v1;
  v3 = *(unsigned int *)(a1 + 184);
  if ( (unsigned int)(4 * v1) < 0x40 )
    v7 = 64;
  if ( v7 >= (unsigned int)v3 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 168);
    for ( i = &v4[2 * v3]; i != v4; v4 += 2 )
      *v4 = -4096;
    *(_QWORD *)(a1 + 176) = 0;
    goto LABEL_7;
  }
  v8 = v1 - 1;
  if ( !v8 )
  {
    v9 = *(_QWORD **)(a1 + 168);
    v10 = 64;
LABEL_15:
    sub_C7D6A0((__int64)v9, 16LL * (unsigned int)v3, 8);
    v11 = ((((((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
             | (4 * v10 / 3u + 1)
             | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
           | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
           | (4 * v10 / 3u + 1)
           | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
           | (4 * v10 / 3u + 1)
           | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
         | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
         | (4 * v10 / 3u + 1)
         | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 16;
    v12 = (v11
         | (((((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
             | (4 * v10 / 3u + 1)
             | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
           | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
           | (4 * v10 / 3u + 1)
           | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
           | (4 * v10 / 3u + 1)
           | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
         | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
         | (4 * v10 / 3u + 1)
         | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 184) = v12;
    v13 = (_QWORD *)sub_C7D670(16 * v12, 8);
    v14 = *(unsigned int *)(a1 + 184);
    *(_QWORD *)(a1 + 176) = 0;
    *(_QWORD *)(a1 + 168) = v13;
    for ( j = &v13[2 * v14]; j != v13; v13 += 2 )
    {
      if ( v13 )
        *v13 = -4096;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v8, v8);
  v9 = *(_QWORD **)(a1 + 168);
  v10 = 1 << (33 - (v8 ^ 0x1F));
  if ( v10 < 64 )
    v10 = 64;
  if ( v10 != (_DWORD)v3 )
    goto LABEL_15;
  *(_QWORD *)(a1 + 176) = 0;
  v16 = &v9[2 * (unsigned int)v10];
  do
  {
    if ( v9 )
      *v9 = -4096;
    v9 += 2;
  }
  while ( v16 != v9 );
LABEL_7:
  result = sub_E38360(a1 + 48);
  *(_DWORD *)(a1 + 152) = 2;
  return result;
}
