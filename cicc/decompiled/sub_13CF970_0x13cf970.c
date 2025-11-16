// Function: sub_13CF970
// Address: 0x13cf970
//
__int64 __fastcall sub_13CF970(__int64 a1)
{
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    return *(_QWORD *)(a1 - 8);
  else
    return a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
}
