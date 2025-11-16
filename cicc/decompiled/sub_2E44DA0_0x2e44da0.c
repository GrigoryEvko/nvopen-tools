// Function: sub_2E44DA0
// Address: 0x2e44da0
//
void __fastcall sub_2E44DA0(__int64 a1)
{
  int v2; // r15d
  __int64 v3; // rbx
  unsigned int v4; // eax
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 v7; // r13
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  int v10; // edx
  int v11; // ebx
  unsigned int v12; // r15d
  unsigned int v13; // eax
  _DWORD *v14; // rdi
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rdi
  _DWORD *v17; // rax
  __int64 v18; // rdx
  _DWORD *i; // rdx
  _DWORD *v20; // rax

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( v2 || *(_DWORD *)(a1 + 20) )
  {
    v3 = *(_QWORD *)(a1 + 8);
    v4 = 4 * v2;
    v5 = *(unsigned int *)(a1 + 24);
    v6 = v5 << 7;
    if ( (unsigned int)(4 * v2) < 0x40 )
      v4 = 64;
    v7 = v3 + v6;
    if ( (unsigned int)v5 <= v4 )
    {
      for ( ; v3 != v7; v3 += 128 )
      {
        if ( *(_DWORD *)v3 != -1 )
        {
          if ( *(_DWORD *)v3 != -2 )
          {
            v8 = *(_QWORD *)(v3 + 88);
            if ( v8 != v3 + 104 )
              _libc_free(v8);
            if ( !*(_BYTE *)(v3 + 52) )
              _libc_free(*(_QWORD *)(v3 + 32));
          }
          *(_DWORD *)v3 = -1;
        }
      }
LABEL_15:
      *(_QWORD *)(a1 + 16) = 0;
      return;
    }
    do
    {
      if ( *(_DWORD *)v3 <= 0xFFFFFFFD )
      {
        v9 = *(_QWORD *)(v3 + 88);
        if ( v9 != v3 + 104 )
          _libc_free(v9);
        if ( !*(_BYTE *)(v3 + 52) )
          _libc_free(*(_QWORD *)(v3 + 32));
      }
      v3 += 128;
    }
    while ( v7 != v3 );
    v10 = *(_DWORD *)(a1 + 24);
    if ( !v2 )
    {
      if ( v10 )
      {
        sub_C7D6A0(*(_QWORD *)(a1 + 8), v6, 8);
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 24) = 0;
        return;
      }
      goto LABEL_15;
    }
    v11 = 64;
    v12 = v2 - 1;
    if ( v12 )
    {
      _BitScanReverse(&v13, v12);
      v11 = 1 << (33 - (v13 ^ 0x1F));
      if ( v11 < 64 )
        v11 = 64;
    }
    v14 = *(_DWORD **)(a1 + 8);
    if ( v11 == v10 )
    {
      *(_QWORD *)(a1 + 16) = 0;
      v20 = &v14[32 * (unsigned __int64)(unsigned int)v11];
      do
      {
        if ( v14 )
          *v14 = -1;
        v14 += 32;
      }
      while ( v20 != v14 );
    }
    else
    {
      sub_C7D6A0((__int64)v14, v6, 8);
      v15 = ((((((((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
               | (4 * v11 / 3u + 1)
               | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 4)
             | (((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
             | (4 * v11 / 3u + 1)
             | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
             | (4 * v11 / 3u + 1)
             | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 4)
           | (((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
           | (4 * v11 / 3u + 1)
           | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 16;
      v16 = (v15
           | (((((((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
               | (4 * v11 / 3u + 1)
               | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 4)
             | (((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
             | (4 * v11 / 3u + 1)
             | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
             | (4 * v11 / 3u + 1)
             | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 4)
           | (((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
           | (4 * v11 / 3u + 1)
           | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 24) = v16;
      v17 = (_DWORD *)sub_C7D670(v16 << 7, 8);
      v18 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = v17;
      for ( i = &v17[32 * v18]; i != v17; v17 += 32 )
      {
        if ( v17 )
          *v17 = -1;
      }
    }
  }
}
