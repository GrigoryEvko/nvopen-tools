// Function: sub_14A9C30
// Address: 0x14a9c30
//
__int64 __fastcall sub_14A9C30(__int64 a1)
{
  __int64 v1; // r8

  v1 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    return *(_QWORD *)(a1 - 8);
  return v1;
}
