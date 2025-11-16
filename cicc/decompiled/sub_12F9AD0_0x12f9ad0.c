// Function: sub_12F9AD0
// Address: 0x12f9ad0
//
__int64 __fastcall sub_12F9AD0(_QWORD *a1, _QWORD *a2, __int64 *a3)
{
  __int64 result; // rax
  __int64 v5; // rdx

  result = sub_12FB050(*a1, a1[1], *a2, a2[1]);
  *a3 = result;
  a3[1] = v5;
  return result;
}
