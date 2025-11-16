// Function: sub_19E69C0
// Address: 0x19e69c0
//
void __fastcall sub_19E69C0(__int64 a1)
{
  int v2; // ebx
  __int64 v3; // r12
  _QWORD *v4; // rbx
  _QWORD *i; // r12
  unsigned __int64 v6; // rdi
  unsigned int v7; // eax
  int v8; // r12d
  unsigned int v9; // ebx
  unsigned int v10; // eax
  _QWORD *v11; // rdi
  unsigned int v12; // eax
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *j; // rdx
  _QWORD *v16; // rax

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 20) )
      return;
    v3 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v3 <= 0x40 )
      goto LABEL_4;
    sub_19E6950(a1);
    if ( !*(_DWORD *)(a1 + 24) )
    {
LABEL_11:
      *(_QWORD *)(a1 + 16) = 0;
      return;
    }
    j___libc_free_0(*(_QWORD *)(a1 + 8));
    *(_DWORD *)(a1 + 24) = 0;
LABEL_28:
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    return;
  }
  v7 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v7 = 64;
  if ( (unsigned int)v3 <= v7 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 8);
    for ( i = &v4[8 * v3]; i != v4; v4 += 8 )
    {
      if ( *v4 != -8 )
      {
        if ( *v4 != 0x7FFFFFFF0LL )
        {
          v6 = v4[3];
          if ( v6 != v4[2] )
            _libc_free(v6);
        }
        *v4 = -8;
      }
    }
    goto LABEL_11;
  }
  v8 = 64;
  sub_19E6950(a1);
  v9 = v2 - 1;
  if ( v9 )
  {
    _BitScanReverse(&v10, v9);
    v8 = 1 << (33 - (v10 ^ 0x1F));
    if ( v8 < 64 )
      v8 = 64;
  }
  v11 = *(_QWORD **)(a1 + 8);
  if ( v8 == *(_DWORD *)(a1 + 24) )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v16 = &v11[8 * (unsigned __int64)(unsigned int)v8];
    do
    {
      if ( v11 )
        *v11 = -8;
      v11 += 8;
    }
    while ( v16 != v11 );
  }
  else
  {
    j___libc_free_0(v11);
    v12 = sub_19E3C70(v8);
    *(_DWORD *)(a1 + 24) = v12;
    if ( !v12 )
      goto LABEL_28;
    v13 = (_QWORD *)sub_22077B0((unsigned __int64)v12 << 6);
    v14 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v13;
    for ( j = &v13[8 * v14]; j != v13; v13 += 8 )
    {
      if ( v13 )
        *v13 = -8;
    }
  }
}
