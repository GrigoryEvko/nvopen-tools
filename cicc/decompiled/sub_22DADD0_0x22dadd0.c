// Function: sub_22DADD0
// Address: 0x22dadd0
//
__int64 __fastcall sub_22DADD0(__int64 *a1, __int64 a2)
{
  __int64 result; // rax

  result = *a1 & 7;
  *a1 = result | a2;
  return result;
}
