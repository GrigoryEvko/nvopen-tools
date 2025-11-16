// Function: sub_6E26D0
// Address: 0x6e26d0
//
__int64 __fastcall sub_6E26D0(char a1, __int64 a2)
{
  char v2; // dl
  __int64 result; // rax

  v2 = a1;
  result = word_4D04898;
  if ( !word_4D04898 )
  {
    if ( (a1 & 2) != 0 )
    {
      v2 = a1 | 1;
      result = qword_4D03C50;
      *(_BYTE *)(qword_4D03C50 + 19LL) |= 0x20u;
    }
    *(_BYTE *)(a2 + 64) |= v2;
  }
  return result;
}
