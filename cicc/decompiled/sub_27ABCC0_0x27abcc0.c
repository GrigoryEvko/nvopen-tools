// Function: sub_27ABCC0
// Address: 0x27abcc0
//
__int64 __fastcall sub_27ABCC0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v3; // rdx
  size_t v4; // rdx
  __int64 v6; // rdx
  size_t v7; // rdx

  v2 = 0;
  v3 = *(unsigned int *)(a1 + 8);
  if ( v3 != *(_DWORD *)(a2 + 8) )
    return v2;
  v4 = 8 * v3;
  if ( v4 )
  {
    if ( memcmp(*(const void **)a1, *(const void **)a2, v4) )
      return v2;
  }
  v6 = *(unsigned int *)(a1 + 56);
  v2 = 0;
  if ( v6 != *(_DWORD *)(a2 + 56) )
    return v2;
  v7 = 8 * v6;
  v2 = 1;
  if ( !v7 )
    return v2;
  LOBYTE(v2) = memcmp(*(const void **)(a1 + 48), *(const void **)(a2 + 48), v7) == 0;
  return v2;
}
