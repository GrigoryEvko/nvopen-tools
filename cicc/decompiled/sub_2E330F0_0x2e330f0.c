// Function: sub_2E330F0
// Address: 0x2e330f0
//
__int64 __fastcall sub_2E330F0(_QWORD *a1, _QWORD *a2)
{
  __int64 v2; // rcx
  __int64 v3; // rdx
  __int64 result; // rax

  v2 = a1[23];
  v3 = a1[24];
  result = a1[25];
  a1[23] = *a2;
  a1[24] = a2[1];
  a1[25] = a2[2];
  *a2 = v2;
  a2[1] = v3;
  a2[2] = result;
  return result;
}
