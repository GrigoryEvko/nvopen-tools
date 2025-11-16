// Function: sub_6EB4C0
// Address: 0x6eb4c0
//
__int64 __fastcall sub_6EB4C0(__int64 a1)
{
  __int64 result; // rax

  result = qword_4D03C50;
  if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0 )
  {
    if ( *(_BYTE *)(qword_4D03C50 + 16LL) > 3u && (*(_BYTE *)(qword_4D03C50 + 17LL) & 1) != 0 )
    {
      result = sub_7340D0(a1, 0, 0);
      *(_BYTE *)(a1 + 49) |= 0x10u;
    }
    else
    {
      *(_BYTE *)(a1 + 49) |= 0x10u;
    }
  }
  return result;
}
