// Function: sub_620DC0
// Address: 0x620dc0
//
__int64 __fastcall sub_620DC0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  for ( result = 0; result != 16; result += 2 )
    *(_WORD *)(a1 + result) = *(_WORD *)(a2 + result);
  return result;
}
