// Function: sub_24DC960
// Address: 0x24dc960
//
__int64 __fastcall sub_24DC960(_QWORD *a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 result; // rax

  v2 = *a2;
  *a2 = 0;
  *a1 = v2;
  v3 = a2[1];
  a2[1] = 0;
  a1[1] = v3;
  result = a2[2];
  a2[2] = 0;
  a1[2] = result;
  a1[3] = 0;
  a1[4] = 0;
  return result;
}
