// Function: sub_134B290
// Address: 0x134b290
//
__int64 __fastcall sub_134B290(_QWORD *a1)
{
  __int64 result; // rax

  result = a1[103] - a1[105];
  a1[28] = 0;
  a1[29] = result;
  return result;
}
