// Function: sub_26C3830
// Address: 0x26c3830
//
void __fastcall sub_26C3830(__int64 a1)
{
  int v2; // r15d
  _QWORD *v3; // rbx
  unsigned int v4; // eax
  __int64 v5; // r14
  _QWORD *v6; // r13
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  __int64 v9; // rdx
  int v10; // r13d
  unsigned int v11; // r15d
  unsigned int v12; // eax
  _QWORD *v13; // rdi
  unsigned int v14; // eax
  _QWORD *v15; // rax
  __int64 v16; // rdx
  _QWORD *i; // rdx
  _QWORD *v18; // rax

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 && !*(_DWORD *)(a1 + 20) )
    return;
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
    goto LABEL_13;
  }
  do
  {
    if ( *v3 != -4096 && *v3 != -8192 )
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
    if ( !(_DWORD)v9 )
    {
LABEL_13:
      *(_QWORD *)(a1 + 16) = 0;
      return;
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 8), v5, 8);
    *(_DWORD *)(a1 + 24) = 0;
    goto LABEL_34;
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
  if ( (_DWORD)v9 != v10 )
  {
    sub_C7D6A0((__int64)v13, v5, 8);
    v14 = sub_26BC060(v10);
    *(_DWORD *)(a1 + 24) = v14;
    if ( v14 )
    {
      v15 = (_QWORD *)sub_C7D670(88LL * v14, 8);
      v16 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = v15;
      for ( i = &v15[11 * v16]; i != v15; v15 += 11 )
      {
        if ( v15 )
          *v15 = -4096;
      }
      return;
    }
LABEL_34:
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    return;
  }
  *(_QWORD *)(a1 + 16) = 0;
  v18 = &v13[11 * v9];
  do
  {
    if ( v13 )
      *v13 = -4096;
    v13 += 11;
  }
  while ( v18 != v13 );
}
