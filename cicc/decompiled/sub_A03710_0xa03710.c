// Function: sub_A03710
// Address: 0xa03710
//
__int64 __fastcall sub_A03710(__int64 *a1, __int64 *a2)
{
  __int64 result; // rax

  result = *a2;
  *a1 = *a2;
  *a2 = 0;
  return result;
}
