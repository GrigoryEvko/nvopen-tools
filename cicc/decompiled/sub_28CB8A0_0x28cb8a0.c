// Function: sub_28CB8A0
// Address: 0x28cb8a0
//
void __fastcall sub_28CB8A0(__int64 a1)
{
  int v1; // ebx
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 i; // r13
  unsigned int v5; // eax
  __int64 v6; // r15
  unsigned int v7; // ebx
  unsigned int v8; // eax
  unsigned int v9; // eax
  _QWORD *v10; // rax
  __int64 v11; // rcx
  _QWORD *j; // rdx
  _QWORD *v13; // rax
  _QWORD *v14; // rdx

  v1 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v1 )
  {
    if ( !*(_DWORD *)(a1 + 20) )
      return;
    v2 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v2 <= 0x40 )
      goto LABEL_4;
    sub_28CB820(a1);
    if ( !*(_DWORD *)(a1 + 24) )
    {
LABEL_11:
      *(_QWORD *)(a1 + 16) = 0;
      return;
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 8), 56 * v2, 8);
    *(_DWORD *)(a1 + 24) = 0;
LABEL_28:
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    return;
  }
  v5 = 4 * v1;
  v2 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v1) < 0x40 )
    v5 = 64;
  if ( (unsigned int)v2 <= v5 )
  {
LABEL_4:
    v3 = *(_QWORD *)(a1 + 8);
    for ( i = v3 + 56 * v2; i != v3; v3 += 56 )
    {
      if ( *(_QWORD *)v3 != -8 )
      {
        if ( *(_QWORD *)v3 != 0x7FFFFFFF0LL && !*(_BYTE *)(v3 + 36) )
          _libc_free(*(_QWORD *)(v3 + 16));
        *(_QWORD *)v3 = -8;
      }
    }
    goto LABEL_11;
  }
  v6 = 64;
  sub_28CB820(a1);
  v7 = v1 - 1;
  if ( v7 )
  {
    _BitScanReverse(&v8, v7);
    v6 = (unsigned int)(1 << (33 - (v8 ^ 0x1F)));
    if ( (int)v6 < 64 )
      v6 = 64;
  }
  if ( *(_DWORD *)(a1 + 24) == (_DWORD)v6 )
  {
    v13 = *(_QWORD **)(a1 + 8);
    *(_QWORD *)(a1 + 16) = 0;
    v14 = &v13[7 * v6];
    do
    {
      if ( v13 )
        *v13 = -8;
      v13 += 7;
    }
    while ( v14 != v13 );
  }
  else
  {
    sub_C7D6A0(*(_QWORD *)(a1 + 8), 56 * v2, 8);
    v9 = sub_28C9170(v6);
    *(_DWORD *)(a1 + 24) = v9;
    if ( !v9 )
      goto LABEL_28;
    v10 = (_QWORD *)sub_C7D670(56LL * v9, 8);
    v11 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v10;
    for ( j = &v10[7 * v11]; j != v10; v10 += 7 )
    {
      if ( v10 )
        *v10 = -8;
    }
  }
}
