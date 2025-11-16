// Function: sub_67D310
// Address: 0x67d310
//
__int64 __fastcall sub_67D310(unsigned int a1, unsigned __int8 a2)
{
  __int64 result; // rax
  unsigned __int64 v3; // rdi
  __int64 v4; // rax

  if ( a1 == 3515 )
    return 0;
  if ( a1 > 0xDBB )
  {
    v3 = a1 - 3624;
    if ( (unsigned int)v3 > 0x29 )
      return (unk_4D04630 == 1) & (unsigned __int8)(a2 <= 5u);
    v4 = 0x2000000000BLL;
    if ( !_bittest64(&v4, v3) )
      return (unk_4D04630 == 1) & (unsigned __int8)(a2 <= 5u);
    return 0;
  }
  result = 1;
  if ( a1 != 1205 )
    return (unk_4D04630 == 1) & (unsigned __int8)(a2 <= 5u);
  return result;
}
