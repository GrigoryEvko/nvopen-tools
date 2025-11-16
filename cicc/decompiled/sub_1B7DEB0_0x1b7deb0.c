// Function: sub_1B7DEB0
// Address: 0x1b7deb0
//
__int64 __fastcall sub_1B7DEB0(__int64 a1)
{
  __int64 v2; // rdi

  if ( *(_BYTE *)(a1 + 16) == 55 )
    return *(_QWORD *)(a1 - 48);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v2 = *(_QWORD *)(a1 - 8);
  else
    v2 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  return *(_QWORD *)(v2 + 24);
}
