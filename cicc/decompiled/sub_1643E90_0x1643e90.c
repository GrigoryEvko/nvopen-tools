// Function: sub_1643E90
// Address: 0x1643e90
//
__int64 __fastcall sub_1643E90(_QWORD *a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rax

  v3 = *a2;
  a1[3] = a2;
  a1[4] = a3;
  *a1 = v3;
  a1[2] = a1 + 3;
  a1[1] = 0x10000000ELL;
  return 0x10000000ELL;
}
