// Function: sub_87ECE0
// Address: 0x87ece0
//
_QWORD *__fastcall sub_87ECE0(__int64 *a1, _QWORD *a2, int a3)
{
  _QWORD *v4; // r12

  v4 = sub_87EBB0(0x18u, *a1, a2);
  sub_879260((__int64)v4, (__int64)a1, a3);
  *((_DWORD *)v4 + 11) = ++dword_4F066AC;
  return v4;
}
