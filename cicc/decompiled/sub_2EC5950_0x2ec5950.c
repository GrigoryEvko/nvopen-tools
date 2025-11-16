// Function: sub_2EC5950
// Address: 0x2ec5950
//
__int64 __fastcall sub_2EC5950(_QWORD *a1, _QWORD *a2)
{
  __int64 result; // rax

  *a1 = *a2;
  result = a2[1];
  *a2 = 0;
  a1[1] = result;
  return result;
}
