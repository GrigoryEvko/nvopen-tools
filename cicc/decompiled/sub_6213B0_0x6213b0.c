// Function: sub_6213B0
// Address: 0x6213b0
//
__int64 __fastcall sub_6213B0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  for ( result = 14; result != -2; result -= 2 )
    *(_WORD *)(a1 + result) |= *(_WORD *)(a2 + result);
  return result;
}
