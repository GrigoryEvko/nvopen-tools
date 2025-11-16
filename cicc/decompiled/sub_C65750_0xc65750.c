// Function: sub_C65750
// Address: 0xc65750
//
__int64 __fastcall sub_C65750(_QWORD *a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 result; // rax

  v2 = *a2;
  *a2 = 0;
  *a1 = v2;
  result = a2[1];
  a2[1] = 0;
  a1[1] = result;
  return result;
}
