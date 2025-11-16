// Function: sub_3585AC0
// Address: 0x3585ac0
//
void __fastcall sub_3585AC0(__int64 a1)
{
  int v2; // r15d
  _QWORD *v3; // rbx
  unsigned int v4; // eax
  __int64 v5; // r14
  _QWORD *v6; // r13
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  __int64 v9; // rdx
  int v10; // ebx
  unsigned int v11; // r15d
  unsigned int v12; // eax
  _QWORD *v13; // rdi
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  _QWORD *v16; // rax
  __int64 v17; // rdx
  _QWORD *i; // rdx
  _QWORD *v19; // rax

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( v2 || *(_DWORD *)(a1 + 20) )
  {
    v3 = *(_QWORD **)(a1 + 8);
    v4 = 4 * v2;
    v5 = 88LL * *(unsigned int *)(a1 + 24);
    if ( (unsigned int)(4 * v2) < 0x40 )
      v4 = 64;
    v6 = &v3[(unsigned __int64)v5 / 8];
    if ( *(_DWORD *)(a1 + 24) <= v4 )
    {
      for ( ; v3 != v6; v3 += 11 )
      {
        if ( *v3 != -4096 )
        {
          if ( *v3 != -8192 )
          {
            v7 = v3[1];
            if ( (_QWORD *)v7 != v3 + 3 )
              _libc_free(v7);
          }
          *v3 = -4096;
        }
      }
LABEL_13:
      *(_QWORD *)(a1 + 16) = 0;
      return;
    }
    do
    {
      if ( *v3 != -8192 && *v3 != -4096 )
      {
        v8 = v3[1];
        if ( (_QWORD *)v8 != v3 + 3 )
          _libc_free(v8);
      }
      v3 += 11;
    }
    while ( v3 != v6 );
    v9 = *(unsigned int *)(a1 + 24);
    if ( !v2 )
    {
      if ( (_DWORD)v9 )
      {
        sub_C7D6A0(*(_QWORD *)(a1 + 8), v5, 8);
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 24) = 0;
        return;
      }
      goto LABEL_13;
    }
    v10 = 64;
    v11 = v2 - 1;
    if ( v11 )
    {
      _BitScanReverse(&v12, v11);
      v10 = 1 << (33 - (v12 ^ 0x1F));
      if ( v10 < 64 )
        v10 = 64;
    }
    v13 = *(_QWORD **)(a1 + 8);
    if ( (_DWORD)v9 == v10 )
    {
      *(_QWORD *)(a1 + 16) = 0;
      v19 = &v13[11 * v9];
      do
      {
        if ( v13 )
          *v13 = -4096;
        v13 += 11;
      }
      while ( v19 != v13 );
    }
    else
    {
      sub_C7D6A0((__int64)v13, v5, 8);
      v14 = (((((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
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
          | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1);
      v15 = ((v14 >> 16) | v14) + 1;
      *(_DWORD *)(a1 + 24) = v15;
      v16 = (_QWORD *)sub_C7D670(88 * v15, 8);
      v17 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = v16;
      for ( i = &v16[11 * v17]; i != v16; v16 += 11 )
      {
        if ( v16 )
          *v16 = -4096;
      }
    }
  }
}
