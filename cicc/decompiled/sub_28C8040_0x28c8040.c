// Function: sub_28C8040
// Address: 0x28c8040
//
__int64 __fastcall sub_28C8040(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  int v3; // r14d
  unsigned int v5; // eax

  v2 = 0;
  v3 = *(_DWORD *)(a2 + 8);
  if ( (unsigned int)(v3 - 11) > 1 )
    return v2;
  if ( *(_DWORD *)(a1 + 12) != *(_DWORD *)(a2 + 12) )
    return v2;
  if ( *(_QWORD *)(a1 + 40) != *(_QWORD *)(a2 + 40) )
    return v2;
  v5 = *(_DWORD *)(a1 + 36);
  if ( v5 != *(_DWORD *)(a2 + 36) || 8LL * v5 && memcmp(*(const void **)(a1 + 24), *(const void **)(a2 + 24), 8LL * v5) )
    return v2;
  if ( *(_QWORD *)(a1 + 48) != *(_QWORD *)(a2 + 48) )
    return 0;
  v2 = 1;
  if ( v3 != 12 )
    return v2;
  LOBYTE(v2) = *(_QWORD *)(a1 + 64) == *(_QWORD *)(a2 + 64);
  return v2;
}
