// Function: sub_18A7BC0
// Address: 0x18a7bc0
//
void __fastcall sub_18A7BC0(__int64 a1)
{
  int v2; // r14d
  _QWORD *v3; // rbx
  _QWORD *v4; // r13
  unsigned int v5; // eax
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  int v8; // edx
  __int64 v9; // rbx
  unsigned int v10; // r14d
  unsigned int v11; // eax
  _QWORD *v12; // rdi
  unsigned int v13; // eax
  _QWORD *v14; // rax
  __int64 v15; // rdx
  _QWORD *i; // rdx
  _QWORD *v17; // rax

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 && !*(_DWORD *)(a1 + 20) )
    return;
  v3 = *(_QWORD **)(a1 + 8);
  v4 = &v3[11 * *(unsigned int *)(a1 + 24)];
  v5 = 4 * v2;
  if ( (unsigned int)(4 * v2) < 0x40 )
    v5 = 64;
  if ( *(_DWORD *)(a1 + 24) <= v5 )
  {
    for ( ; v3 != v4; v3 += 11 )
    {
      if ( *v3 != -8 )
      {
        if ( *v3 != -16 )
        {
          v6 = v3[1];
          if ( (_QWORD *)v6 != v3 + 3 )
            _libc_free(v6);
        }
        *v3 = -8;
      }
    }
    goto LABEL_13;
  }
  do
  {
    if ( *v3 != -16 && *v3 != -8 )
    {
      v7 = v3[1];
      if ( (_QWORD *)v7 != v3 + 3 )
        _libc_free(v7);
    }
    v3 += 11;
  }
  while ( v3 != v4 );
  v8 = *(_DWORD *)(a1 + 24);
  if ( !v2 )
  {
    if ( !v8 )
    {
LABEL_13:
      *(_QWORD *)(a1 + 16) = 0;
      return;
    }
    j___libc_free_0(*(_QWORD *)(a1 + 8));
    *(_DWORD *)(a1 + 24) = 0;
    goto LABEL_34;
  }
  v9 = 64;
  v10 = v2 - 1;
  if ( v10 )
  {
    _BitScanReverse(&v11, v10);
    v9 = (unsigned int)(1 << (33 - (v11 ^ 0x1F)));
    if ( (int)v9 < 64 )
      v9 = 64;
  }
  v12 = *(_QWORD **)(a1 + 8);
  if ( (_DWORD)v9 != v8 )
  {
    j___libc_free_0(v12);
    v13 = sub_1454B60(4 * (int)v9 / 3u + 1);
    *(_DWORD *)(a1 + 24) = v13;
    if ( v13 )
    {
      v14 = (_QWORD *)sub_22077B0(88LL * v13);
      v15 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = v14;
      for ( i = &v14[11 * v15]; i != v14; v14 += 11 )
      {
        if ( v14 )
          *v14 = -8;
      }
      return;
    }
LABEL_34:
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    return;
  }
  *(_QWORD *)(a1 + 16) = 0;
  v17 = &v12[11 * v9];
  do
  {
    if ( v12 )
      *v12 = -8;
    v12 += 11;
  }
  while ( v17 != v12 );
}
