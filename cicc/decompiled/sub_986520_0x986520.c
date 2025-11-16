// Function: sub_986520
// Address: 0x986520
//
__int64 __fastcall sub_986520(__int64 a1)
{
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    return *(_QWORD *)(a1 - 8);
  else
    return a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
}
