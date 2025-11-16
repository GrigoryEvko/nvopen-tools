// Function: sub_19904D0
// Address: 0x19904d0
//
void __fastcall sub_19904D0(__int64 a1)
{
  int v2; // r14d
  _QWORD *v3; // rbx
  __int64 v4; // rdx
  _QWORD *v5; // r13
  unsigned int v6; // eax
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  int v9; // edx
  __int64 v10; // rbx
  unsigned int v11; // r14d
  unsigned int v12; // eax
  _QWORD *v13; // rdi
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rax
  _QWORD *v16; // rax
  __int64 v17; // rdx
  _QWORD *i; // rdx
  _QWORD *v19; // rax

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 && !*(_DWORD *)(a1 + 20) )
    return;
  v3 = *(_QWORD **)(a1 + 8);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = &v3[5 * v4];
  v6 = 4 * v2;
  if ( (unsigned int)(4 * v2) < 0x40 )
    v6 = 64;
  if ( (unsigned int)v4 <= v6 )
  {
    while ( v3 != v5 )
    {
      if ( *v3 != -8 )
      {
        if ( *v3 != -16 )
        {
          v7 = v3[1];
          if ( (_QWORD *)v7 != v3 + 3 )
            _libc_free(v7);
        }
        *v3 = -8;
      }
      v3 += 5;
    }
LABEL_14:
    *(_QWORD *)(a1 + 16) = 0;
    return;
  }
  do
  {
    if ( *v3 != -16 && *v3 != -8 )
    {
      v8 = v3[1];
      if ( (_QWORD *)v8 != v3 + 3 )
        _libc_free(v8);
    }
    v3 += 5;
  }
  while ( v3 != v5 );
  v9 = *(_DWORD *)(a1 + 24);
  if ( !v2 )
  {
    if ( v9 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 8));
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return;
    }
    goto LABEL_14;
  }
  v10 = 64;
  v11 = v2 - 1;
  if ( v11 )
  {
    _BitScanReverse(&v12, v11);
    v10 = (unsigned int)(1 << (33 - (v12 ^ 0x1F)));
    if ( (int)v10 < 64 )
      v10 = 64;
  }
  v13 = *(_QWORD **)(a1 + 8);
  if ( (_DWORD)v10 == v9 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v19 = &v13[5 * v10];
    do
    {
      if ( v13 )
        *v13 = -8;
      v13 += 5;
    }
    while ( v19 != v13 );
  }
  else
  {
    j___libc_free_0(v13);
    v14 = ((((((((4 * (int)v10 / 3u + 1) | ((unsigned __int64)(4 * (int)v10 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v10 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v10 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v10 / 3u + 1) | ((unsigned __int64)(4 * (int)v10 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v10 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v10 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v10 / 3u + 1) | ((unsigned __int64)(4 * (int)v10 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v10 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v10 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v10 / 3u + 1) | ((unsigned __int64)(4 * (int)v10 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v10 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v10 / 3u + 1) >> 1)) >> 16;
    v15 = (v14
         | (((((((4 * (int)v10 / 3u + 1) | ((unsigned __int64)(4 * (int)v10 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v10 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v10 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v10 / 3u + 1) | ((unsigned __int64)(4 * (int)v10 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v10 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v10 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v10 / 3u + 1) | ((unsigned __int64)(4 * (int)v10 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v10 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v10 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v10 / 3u + 1) | ((unsigned __int64)(4 * (int)v10 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v10 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v10 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 24) = v15;
    v16 = (_QWORD *)sub_22077B0(40 * v15);
    v17 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v16;
    for ( i = &v16[5 * v17]; i != v16; v16 += 5 )
    {
      if ( v16 )
        *v16 = -8;
    }
  }
}
