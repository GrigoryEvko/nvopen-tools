// Function: sub_25DC1A0
// Address: 0x25dc1a0
//
__int64 __fastcall sub_25DC1A0(__int64 a1)
{
  __int64 v1; // rax
  unsigned int v2; // r8d

  if ( !a1 )
    return 0;
  v1 = *(_QWORD *)(a1 - 32);
  v2 = 0;
  if ( !v1 || *(_BYTE *)v1 || *(_QWORD *)(v1 + 24) != *(_QWORD *)(a1 + 80) || (*(_BYTE *)(v1 + 33) & 0x20) == 0 )
    return 0;
  LOBYTE(v2) = *(_DWORD *)(v1 + 36) == 238;
  return v2;
}
