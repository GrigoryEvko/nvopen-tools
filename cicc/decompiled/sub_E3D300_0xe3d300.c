// Function: sub_E3D300
// Address: 0xe3d300
//
__int64 __fastcall sub_E3D300(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax

  result = a3 + 8 * a4;
  *a2 = a3;
  a2[1] = result;
  *a1 = 0;
  return result;
}
