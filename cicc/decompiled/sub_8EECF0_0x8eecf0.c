// Function: sub_8EECF0
// Address: 0x8eecf0
//
__int64 __fastcall sub_8EECF0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  unsigned int v4; // r8d
  _DWORD *v5; // rdi
  _DWORD *v6; // rdx
  _DWORD *v7; // rax

  v2 = *(unsigned int *)(a1 + 2088);
  v3 = *(unsigned int *)(a2 + 2088);
  v4 = *(_DWORD *)(a1 + 2088) - v3;
  if ( !v4 && (_DWORD)v2 )
  {
    v5 = (_DWORD *)(a1 + 8);
    v6 = (_DWORD *)(a2 + 4 * v3 + 8);
    v7 = &v5[v2];
    while ( *--v7 == *--v6 )
    {
      if ( v5 == v7 )
        return v4;
    }
    return *v7 < *v6 ? -1 : 1;
  }
  return v4;
}
