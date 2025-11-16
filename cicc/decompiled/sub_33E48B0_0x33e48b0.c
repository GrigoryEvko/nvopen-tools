// Function: sub_33E48B0
// Address: 0x33e48b0
//
unsigned __int64 __fastcall sub_33E48B0(__int64 *a1)
{
  __int64 v1; // rdx
  unsigned __int64 result; // rax

  v1 = *a1;
  a1[10] += 120;
  result = (v1 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[1] < result + 120 || !v1 )
    return sub_9D1E70((__int64)a1, 120, 120, 3);
  *a1 = result + 120;
  return result;
}
