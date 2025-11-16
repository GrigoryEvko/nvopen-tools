// Function: sub_1F15B50
// Address: 0x1f15b50
//
__int64 __fastcall sub_1F15B50(__int64 a1, __int64 a2, int a3, __int64 a4, unsigned __int64 a5, unsigned __int64 a6)
{
  __int64 v7; // rcx
  int v8; // eax
  __int64 v9; // rdx
  _DWORD *v10; // rax
  _DWORD *i; // rdx
  int v12; // r9d
  unsigned int v14; // ecx
  _DWORD *v15; // rdi
  unsigned int v16; // eax
  int v17; // eax
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rax
  int v20; // r13d
  __int64 v21; // r12
  _DWORD *v22; // rax
  __int64 v23; // rdx
  _DWORD *j; // rdx
  _DWORD *v25; // rax

  v7 = *(unsigned int *)(a1 + 384);
  *(_QWORD *)(a1 + 72) = a2;
  *(_DWORD *)(a1 + 84) = a3;
  *(_DWORD *)(a1 + 80) = 0;
  if ( (_DWORD)v7 )
  {
    sub_1EC25D0(a1 + 200, (char *)sub_1EBAFD0, 0, v7, a5, a6);
    *(_DWORD *)(a1 + 384) = 0;
    memset((void *)(a1 + 200), 0, 0xB8u);
  }
  v8 = *(_DWORD *)(a1 + 416);
  ++*(_QWORD *)(a1 + 400);
  *(_DWORD *)(a1 + 388) = 0;
  if ( !v8 )
  {
    if ( !*(_DWORD *)(a1 + 420) )
      goto LABEL_9;
    v9 = *(unsigned int *)(a1 + 424);
    if ( (unsigned int)v9 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 408));
      *(_QWORD *)(a1 + 408) = 0;
      *(_QWORD *)(a1 + 416) = 0;
      *(_DWORD *)(a1 + 424) = 0;
      goto LABEL_9;
    }
    goto LABEL_6;
  }
  v14 = 4 * v8;
  v9 = *(unsigned int *)(a1 + 424);
  if ( (unsigned int)(4 * v8) < 0x40 )
    v14 = 64;
  if ( (unsigned int)v9 <= v14 )
  {
LABEL_6:
    v10 = *(_DWORD **)(a1 + 408);
    for ( i = &v10[4 * v9]; i != v10; *(v10 - 3) = -1 )
    {
      *v10 = -1;
      v10 += 4;
    }
    *(_QWORD *)(a1 + 416) = 0;
    goto LABEL_9;
  }
  v15 = *(_DWORD **)(a1 + 408);
  v16 = v8 - 1;
  if ( !v16 )
  {
    v21 = 2048;
    v20 = 128;
LABEL_20:
    j___libc_free_0(v15);
    *(_DWORD *)(a1 + 424) = v20;
    v22 = (_DWORD *)sub_22077B0(v21);
    v23 = *(unsigned int *)(a1 + 424);
    *(_QWORD *)(a1 + 416) = 0;
    *(_QWORD *)(a1 + 408) = v22;
    for ( j = &v22[4 * v23]; j != v22; v22 += 4 )
    {
      if ( v22 )
      {
        *v22 = -1;
        v22[1] = -1;
      }
    }
    goto LABEL_9;
  }
  _BitScanReverse(&v16, v16);
  v17 = 1 << (33 - (v16 ^ 0x1F));
  if ( v17 < 64 )
    v17 = 64;
  if ( (_DWORD)v9 != v17 )
  {
    v18 = (((4 * v17 / 3u + 1) | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 2)
        | (4 * v17 / 3u + 1)
        | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)
        | (((((4 * v17 / 3u + 1) | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 2)
          | (4 * v17 / 3u + 1)
          | ((unsigned __int64)(4 * v17 / 3u + 1) >> 1)) >> 4);
    v19 = (v18 >> 8) | v18;
    v20 = (v19 | (v19 >> 16)) + 1;
    v21 = 16 * ((v19 | (v19 >> 16)) + 1);
    goto LABEL_20;
  }
  *(_QWORD *)(a1 + 416) = 0;
  v25 = &v15[4 * (unsigned int)v9];
  do
  {
    if ( v15 )
    {
      *v15 = -1;
      v15[1] = -1;
    }
    v15 += 4;
  }
  while ( v25 != v15 );
LABEL_9:
  sub_1DC3BD0(
    (__m128i *)(a1 + 432),
    *(_QWORD *)(*(_QWORD *)(a1 + 24) + 256LL),
    *(_QWORD *)(*(_QWORD *)(a1 + 16) + 272LL),
    *(_QWORD *)(a1 + 40),
    *(_QWORD *)(a1 + 16) + 296LL,
    a6);
  if ( *(_DWORD *)(a1 + 84) )
    sub_1DC3BD0(
      (__m128i *)(a1 + 1096),
      *(_QWORD *)(*(_QWORD *)(a1 + 24) + 256LL),
      *(_QWORD *)(*(_QWORD *)(a1 + 16) + 272LL),
      *(_QWORD *)(a1 + 40),
      *(_QWORD *)(a1 + 16) + 296LL,
      v12);
  return sub_20FFFD0(*(_QWORD *)(a1 + 72), 0);
}
