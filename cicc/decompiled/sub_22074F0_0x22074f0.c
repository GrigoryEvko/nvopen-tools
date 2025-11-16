// Function: sub_22074F0
// Address: 0x22074f0
//
__int64 __fastcall sub_22074F0(__int64 *a1, __int64 *a2)
{
  __int64 result; // rax

  result = *a1;
  *a1 = *a2;
  *a2 = result;
  return result;
}
