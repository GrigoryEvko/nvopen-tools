// Function: sub_16E8BB0
// Address: 0x16e8bb0
//
__off_t __fastcall sub_16E8BB0(_QWORD *a1, char *a2, size_t a3, __off_t a4)
{
  __off_t v6; // r13

  v6 = (*(__int64 (__fastcall **)(_QWORD *))(*a1 + 64LL))(a1) + a1[3] - a1[1];
  sub_16E8B50((__int64)a1, a4);
  sub_16E7EE0((__int64)a1, a2, a3);
  return sub_16E8B50((__int64)a1, v6);
}
