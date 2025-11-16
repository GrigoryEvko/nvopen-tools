// Function: sub_24F3200
// Address: 0x24f3200
//
__int64 __fastcall sub_24F3200(__int64 a1)
{
  __int64 v1; // rax
  unsigned int v2; // r8d
  __int64 v4; // rdx

  v1 = *(_QWORD *)(a1 + 56);
  if ( !v1 )
    BUG();
  v2 = 0;
  if ( *(_BYTE *)(v1 - 24) != 85 )
    return 0;
  v4 = *(_QWORD *)(v1 - 56);
  if ( !v4 || *(_BYTE *)v4 || *(_QWORD *)(v4 + 24) != *(_QWORD *)(v1 + 56) || (*(_BYTE *)(v4 + 33) & 0x20) == 0 )
    return 0;
  LOBYTE(v2) = (unsigned int)(*(_DWORD *)(v4 + 36) - 60) <= 2;
  return v2;
}
