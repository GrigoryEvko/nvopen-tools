// Function: sub_19E1B80
// Address: 0x19e1b80
//
__int64 __fastcall sub_19E1B80(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  unsigned int v4; // eax

  v2 = 0;
  if ( *(_DWORD *)(a1 + 12) != *(_DWORD *)(a2 + 12) )
    return 0;
  if ( *(_QWORD *)(a1 + 40) != *(_QWORD *)(a2 + 40) )
    return 0;
  v4 = *(_DWORD *)(a1 + 36);
  if ( v4 != *(_DWORD *)(a2 + 36) || 8LL * v4 && memcmp(*(const void **)(a1 + 24), *(const void **)(a2 + 24), 8LL * v4) )
    return 0;
  LOBYTE(v2) = *(_QWORD *)(a1 + 48) == *(_QWORD *)(a2 + 48);
  return v2;
}
