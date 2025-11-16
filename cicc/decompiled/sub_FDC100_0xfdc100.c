// Function: sub_FDC100
// Address: 0xfdc100
//
__int64 __fastcall sub_FDC100(__int64 *a1, __int64 *a2)
{
  __int64 result; // rax

  result = *a2;
  *a1 = *a2;
  *a2 = 0;
  return result;
}
