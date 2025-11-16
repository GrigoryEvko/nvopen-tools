// Function: sub_2FE2EC0
// Address: 0x2fe2ec0
//
__int64 __fastcall sub_2FE2EC0(__int64 a1, unsigned __int16 a2)
{
  int v2; // edx
  __int64 result; // rax

  v2 = word_4456340[a2 - 1];
  if ( (unsigned __int16)(a2 - 176) <= 0x34u )
    return (v2 & (v2 - 1)) == 0 ? 1 : 7;
  result = 5;
  if ( v2 != 1 )
    return (v2 & (v2 - 1)) == 0 ? 1 : 7;
  return result;
}
