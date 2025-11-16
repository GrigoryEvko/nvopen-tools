// Function: sub_193FF80
// Address: 0x193ff80
//
__int64 __fastcall sub_193FF80(__int64 a1)
{
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    return *(_QWORD *)(a1 - 8) + 24LL * *(unsigned int *)(a1 + 56) + 8;
  else
    return a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) + 24LL * *(unsigned int *)(a1 + 56) + 8;
}
