// Function: sub_1B94FB0
// Address: 0x1b94fb0
//
_BOOL8 __fastcall sub_1B94FB0(__int64 a1, unsigned int *a2)
{
  __int64 v2; // r8
  _DWORD *v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rax
  __int64 v7; // r9
  unsigned int v8; // esi
  __int64 v9; // rdi
  __int64 v10; // rcx
  __int64 v11; // rdx

  v2 = *(_QWORD *)(a1 + 64);
  if ( !v2 )
  {
    v3 = *(_DWORD **)a1;
    v4 = *(_QWORD *)a1 + 4LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v4 )
    {
      while ( *v3 != *a2 )
      {
        if ( (_DWORD *)v4 == ++v3 )
          return v2;
      }
      return v3 != (_DWORD *)v4;
    }
    return v2;
  }
  v6 = *(_QWORD *)(a1 + 40);
  v7 = a1 + 32;
  if ( !v6 )
    return 0;
  v8 = *a2;
  v9 = a1 + 32;
  do
  {
    while ( 1 )
    {
      v10 = *(_QWORD *)(v6 + 16);
      v11 = *(_QWORD *)(v6 + 24);
      if ( *(_DWORD *)(v6 + 32) >= v8 )
        break;
      v6 = *(_QWORD *)(v6 + 24);
      if ( !v11 )
        goto LABEL_13;
    }
    v9 = v6;
    v6 = *(_QWORD *)(v6 + 16);
  }
  while ( v10 );
LABEL_13:
  v2 = 0;
  if ( v7 == v9 )
    return v2;
  return v8 >= *(_DWORD *)(v9 + 32);
}
