// Function: sub_B141E0
// Address: 0xb141e0
//
__int64 __fastcall sub_B141E0(__int64 *a1)
{
  __int64 result; // rax

  result = *a1;
  *(_QWORD *)(*a1 + 64) = 0;
  *a1 = 0;
  return result;
}
