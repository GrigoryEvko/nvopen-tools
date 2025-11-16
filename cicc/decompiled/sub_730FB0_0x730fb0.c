// Function: sub_730FB0
// Address: 0x730fb0
//
_BOOL8 __fastcall sub_730FB0(char a1)
{
  _BOOL8 result; // rax

  if ( a1 == 20 )
    return 1;
  result = 0;
  if ( (unsigned __int8)(a1 - 29) <= 0x3Bu )
    return ((1LL << (a1 - 29)) & 0xC000007E0000001LL) != 0;
  return result;
}
