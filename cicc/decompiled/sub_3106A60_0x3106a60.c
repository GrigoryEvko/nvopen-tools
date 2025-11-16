// Function: sub_3106A60
// Address: 0x3106a60
//
__int64 __fastcall sub_3106A60(__int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // rdx
  _QWORD *v5; // rax
  _QWORD *i; // rdx
  unsigned int v8; // ecx
  unsigned int v9; // eax
  _QWORD *v10; // rdi
  __int64 v11; // rbx
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // rdx
  _QWORD *j; // rdx
  _QWORD *v17; // rax

  v3 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v3 )
  {
    if ( !*(_DWORD *)(a1 + 20) )
      return sub_3106620(a1, a2);
    v4 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v4 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 8), 8 * v4, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return sub_3106620(a1, a2);
    }
    goto LABEL_4;
  }
  v8 = 4 * v3;
  v4 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v3) < 0x40 )
    v8 = 64;
  if ( v8 >= (unsigned int)v4 )
  {
LABEL_4:
    v5 = *(_QWORD **)(a1 + 8);
    for ( i = &v5[v4]; i != v5; ++v5 )
      *v5 = -4;
    *(_QWORD *)(a1 + 16) = 0;
    return sub_3106620(a1, a2);
  }
  v9 = v3 - 1;
  if ( !v9 )
  {
    v10 = *(_QWORD **)(a1 + 8);
    LODWORD(v11) = 64;
LABEL_15:
    sub_C7D6A0((__int64)v10, 8 * v4, 8);
    v12 = ((((((((4 * (int)v11 / 3u + 1) | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v11 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v11 / 3u + 1) | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v11 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v11 / 3u + 1) | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v11 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v11 / 3u + 1) | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v11 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 16;
    v13 = (v12
         | (((((((4 * (int)v11 / 3u + 1) | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v11 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v11 / 3u + 1) | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v11 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v11 / 3u + 1) | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v11 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v11 / 3u + 1) | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v11 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v11 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 24) = v13;
    v14 = (_QWORD *)sub_C7D670(8 * v13, 8);
    v15 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v14;
    for ( j = &v14[v15]; j != v14; ++v14 )
    {
      if ( v14 )
        *v14 = -4;
    }
    return sub_3106620(a1, a2);
  }
  _BitScanReverse(&v9, v9);
  v10 = *(_QWORD **)(a1 + 8);
  v11 = (unsigned int)(1 << (33 - (v9 ^ 0x1F)));
  if ( (int)v11 < 64 )
    v11 = 64;
  if ( (_DWORD)v11 != (_DWORD)v4 )
    goto LABEL_15;
  *(_QWORD *)(a1 + 16) = 0;
  v17 = &v10[v11];
  do
  {
    if ( v10 )
      *v10 = -4;
    ++v10;
  }
  while ( v17 != v10 );
  return sub_3106620(a1, a2);
}
