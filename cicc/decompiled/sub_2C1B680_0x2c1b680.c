// Function: sub_2C1B680
// Address: 0x2c1b680
//
__int64 __fastcall sub_2C1B680(__int64 a1, char a2)
{
  __int64 result; // rax

  result = *(unsigned __int8 *)(a1 + 160);
  if ( (_BYTE)result )
  {
    if ( a2 )
      return sub_2C46C30(a1 + 96);
  }
  return result;
}
