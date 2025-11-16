// Function: sub_2FB3410
// Address: 0x2fb3410
//
__int64 __fastcall sub_2FB3410(__int64 a1, __int64 a2, int a3, __int64 a4, unsigned __int64 a5, unsigned __int64 a6)
{
  __int64 v6; // rcx
  int v8; // eax
  __int64 v9; // rdx
  _DWORD *v10; // rax
  _DWORD *i; // rdx
  __int64 v12; // r9
  unsigned int v14; // ecx
  unsigned int v15; // eax
  _DWORD *v16; // rdi
  int v17; // r12d
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rdi
  _DWORD *v20; // rax
  __int64 v21; // rdx
  _DWORD *j; // rdx
  _QWORD *v23; // rdi
  _DWORD *v24; // rax

  v6 = *(unsigned int *)(a1 + 376);
  *(_QWORD *)(a1 + 72) = a2;
  *(_DWORD *)(a1 + 84) = a3;
  *(_DWORD *)(a1 + 80) = 0;
  if ( (_DWORD)v6 )
  {
    sub_2FB2EF0(a1 + 192, (char *)sub_2F4C150, 0, v6, a5, a6);
    *(_DWORD *)(a1 + 376) = 0;
    memset((void *)(a1 + 192), 0, 0xB8u);
    v23 = (_QWORD *)(a1 + 192);
    do
    {
      *v23 = 0;
      v23 += 2;
      *(v23 - 1) = 0;
    }
    while ( (_QWORD *)(a1 + 336) != v23 );
  }
  v8 = *(_DWORD *)(a1 + 408);
  ++*(_QWORD *)(a1 + 392);
  *(_DWORD *)(a1 + 380) = 0;
  if ( !v8 )
  {
    if ( !*(_DWORD *)(a1 + 412) )
      goto LABEL_8;
    v9 = *(unsigned int *)(a1 + 416);
    if ( (unsigned int)v9 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 400), 16LL * (unsigned int)v9, 8);
      *(_QWORD *)(a1 + 400) = 0;
      *(_QWORD *)(a1 + 408) = 0;
      *(_DWORD *)(a1 + 416) = 0;
      goto LABEL_8;
    }
    goto LABEL_5;
  }
  v14 = 4 * v8;
  v9 = *(unsigned int *)(a1 + 416);
  if ( (unsigned int)(4 * v8) < 0x40 )
    v14 = 64;
  if ( v14 >= (unsigned int)v9 )
  {
LABEL_5:
    v10 = *(_DWORD **)(a1 + 400);
    for ( i = &v10[4 * v9]; i != v10; *(v10 - 3) = -1 )
    {
      *v10 = -1;
      v10 += 4;
    }
    *(_QWORD *)(a1 + 408) = 0;
    goto LABEL_8;
  }
  v15 = v8 - 1;
  if ( !v15 )
  {
    v16 = *(_DWORD **)(a1 + 400);
    v17 = 64;
LABEL_18:
    sub_C7D6A0((__int64)v16, 16LL * (unsigned int)v9, 8);
    v18 = ((((((((4 * v17 / 3u + 1) | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 2)
             | (4 * v17 / 3u + 1)
             | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 4)
           | (((4 * v17 / 3u + 1) | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 2)
           | (4 * v17 / 3u + 1)
           | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v17 / 3u + 1) | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 2)
           | (4 * v17 / 3u + 1)
           | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 4)
         | (((4 * v17 / 3u + 1) | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 2)
         | (4 * v17 / 3u + 1)
         | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 16;
    v19 = (v18
         | (((((((4 * v17 / 3u + 1) | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 2)
             | (4 * v17 / 3u + 1)
             | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 4)
           | (((4 * v17 / 3u + 1) | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 2)
           | (4 * v17 / 3u + 1)
           | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v17 / 3u + 1) | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 2)
           | (4 * v17 / 3u + 1)
           | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 4)
         | (((4 * v17 / 3u + 1) | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 2)
         | (4 * v17 / 3u + 1)
         | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 416) = v19;
    v20 = (_DWORD *)sub_C7D670(16 * v19, 8);
    v21 = *(unsigned int *)(a1 + 416);
    *(_QWORD *)(a1 + 408) = 0;
    *(_QWORD *)(a1 + 400) = v20;
    for ( j = &v20[4 * v21]; j != v20; v20 += 4 )
    {
      if ( v20 )
      {
        *v20 = -1;
        v20[1] = -1;
      }
    }
    goto LABEL_8;
  }
  _BitScanReverse(&v15, v15);
  v16 = *(_DWORD **)(a1 + 400);
  v17 = 1 << (33 - (v15 ^ 0x1F));
  if ( v17 < 64 )
    v17 = 64;
  if ( v17 != (_DWORD)v9 )
    goto LABEL_18;
  *(_QWORD *)(a1 + 408) = 0;
  v24 = &v16[4 * v17];
  do
  {
    if ( v16 )
    {
      *v16 = -1;
      v16[1] = -1;
    }
    v16 += 4;
  }
  while ( v24 != v16 );
LABEL_8:
  sub_2E1DCC0(
    a1 + 424,
    *(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL),
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL),
    *(_QWORD *)(a1 + 32),
    *(_QWORD *)(a1 + 8) + 56LL,
    a6);
  if ( *(_DWORD *)(a1 + 84) )
    sub_2E1DCC0(
      a1 + 1136,
      *(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL),
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL),
      *(_QWORD *)(a1 + 32),
      *(_QWORD *)(a1 + 8) + 56LL,
      v12);
  return sub_3509F80(*(_QWORD *)(a1 + 72));
}
