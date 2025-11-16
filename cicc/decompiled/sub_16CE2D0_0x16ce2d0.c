// Function: sub_16CE2D0
// Address: 0x16ce2d0
//
__int64 __fastcall sub_16CE2D0(_QWORD *a1, _QWORD *a2)
{
  __int64 v2; // rax
  __int64 result; // rax

  *a1 = *a2;
  v2 = a2[1];
  *a2 = 0;
  a1[1] = v2;
  result = a2[2];
  a2[1] = 0;
  a1[2] = result;
  return result;
}
