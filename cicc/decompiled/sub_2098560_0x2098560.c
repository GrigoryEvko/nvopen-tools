// Function: sub_2098560
// Address: 0x2098560
//
__int64 __fastcall sub_2098560(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  int v6; // eax
  __int64 v8; // rdx
  _QWORD *v9; // rax
  _QWORD *i; // rdx
  unsigned __int64 *v11; // r13
  __int64 v13; // rdi
  unsigned int v14; // eax
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rax
  int v18; // r14d
  __int64 v19; // r13
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 j; // rdx
  __int64 v23; // rax

  v6 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v6 )
  {
    if ( !*(_DWORD *)(a1 + 20) )
      goto LABEL_7;
    v8 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v8 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 8));
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  a4 = (unsigned int)(4 * v6);
  v8 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)a4 < 0x40 )
    a4 = 64;
  if ( (unsigned int)a4 >= (unsigned int)v8 )
  {
LABEL_4:
    v9 = *(_QWORD **)(a1 + 8);
    for ( i = &v9[4 * v8]; i != v9; *((_DWORD *)v9 - 6) = -1 )
    {
      *v9 = 0;
      v9 += 4;
    }
    *(_QWORD *)(a1 + 16) = 0;
    goto LABEL_7;
  }
  v13 = *(_QWORD *)(a1 + 8);
  v14 = v6 - 1;
  if ( !v14 )
  {
    v19 = 4096;
    v18 = 128;
LABEL_19:
    j___libc_free_0(v13);
    *(_DWORD *)(a1 + 24) = v18;
    v20 = sub_22077B0(v19);
    v21 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v20;
    for ( j = v20 + 32 * v21; j != v20; v20 += 32 )
    {
      if ( v20 )
      {
        *(_QWORD *)v20 = 0;
        *(_DWORD *)(v20 + 8) = -1;
      }
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v14, v14);
  a4 = 33 - (v14 ^ 0x1F);
  v15 = (unsigned int)(1 << (33 - (v14 ^ 0x1F)));
  if ( (int)v15 < 64 )
    v15 = 64;
  if ( (_DWORD)v15 != (_DWORD)v8 )
  {
    v16 = (((4 * (int)v15 / 3u + 1) | ((unsigned __int64)(4 * (int)v15 / 3u + 1) >> 1)) >> 2)
        | (4 * (int)v15 / 3u + 1)
        | ((unsigned __int64)(4 * (int)v15 / 3u + 1) >> 1)
        | (((((4 * (int)v15 / 3u + 1) | ((unsigned __int64)(4 * (int)v15 / 3u + 1) >> 1)) >> 2)
          | (4 * (int)v15 / 3u + 1)
          | ((unsigned __int64)(4 * (int)v15 / 3u + 1) >> 1)) >> 4);
    v17 = (v16 >> 8) | v16;
    v18 = (v17 | (v17 >> 16)) + 1;
    v19 = 32 * ((v17 | (v17 >> 16)) + 1);
    goto LABEL_19;
  }
  *(_QWORD *)(a1 + 16) = 0;
  v23 = v13 + 32 * v15;
  do
  {
    if ( v13 )
    {
      *(_QWORD *)v13 = 0;
      *(_DWORD *)(v13 + 8) = -1;
    }
    v13 += 32;
  }
  while ( v23 != v13 );
LABEL_7:
  v11 = *(unsigned __int64 **)(a1 + 32);
  *(_DWORD *)(a1 + 40) = 0;
  if ( ((unsigned __int8)v11 & 1) == 0 && v11 )
  {
    _libc_free(*v11);
    j_j___libc_free_0(v11, 24);
  }
  *(_QWORD *)(a1 + 32) = 1;
  return sub_13A5100((unsigned __int64 *)(a1 + 32), *(_DWORD *)(*(_QWORD *)(a2 + 712) + 576LL), 0, a4, a5, a6);
}
