// Function: sub_C8EDF0
// Address: 0xc8edf0
//
__int64 __fastcall sub_C8EDF0(_QWORD *a1, _QWORD *a2)
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
