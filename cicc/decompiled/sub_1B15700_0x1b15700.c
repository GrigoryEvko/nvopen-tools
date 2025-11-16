// Function: sub_1B15700
// Address: 0x1b15700
//
_QWORD *__fastcall sub_1B15700(__int64 ***a1)
{
  __int64 **v1; // rbx
  _QWORD *v2; // r12

  v1 = *a1;
  v2 = sub_15E7C30(**a1, (__int64 *)*(*a1)[1], (__int64 *)*(*a1)[2]);
  sub_15F2440((__int64)v2, *(_DWORD *)v1[3]);
  return v2;
}
