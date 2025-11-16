// Function: sub_19E1C00
// Address: 0x19e1c00
//
__int64 __fastcall sub_19E1C00(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  unsigned int v4; // eax
  __int64 v5; // rdx
  size_t v6; // rdx

  v2 = 0;
  if ( *(_DWORD *)(a1 + 12) != *(_DWORD *)(a2 + 12) )
    return v2;
  if ( *(_QWORD *)(a1 + 40) != *(_QWORD *)(a2 + 40) )
    return v2;
  v4 = *(_DWORD *)(a1 + 36);
  if ( v4 != *(_DWORD *)(a2 + 36) || 8LL * v4 && memcmp(*(const void **)(a1 + 24), *(const void **)(a2 + 24), 8LL * v4) )
    return v2;
  v5 = *(unsigned int *)(a1 + 52);
  v2 = 0;
  if ( (_DWORD)v5 != *(_DWORD *)(a2 + 52) )
    return v2;
  v6 = 4 * v5;
  v2 = 1;
  if ( !v6 )
    return v2;
  LOBYTE(v2) = memcmp(*(const void **)(a1 + 56), *(const void **)(a2 + 56), v6) == 0;
  return v2;
}
