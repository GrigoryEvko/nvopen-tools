// Function: sub_CB7120
// Address: 0xcb7120
//
__off_t __fastcall sub_CB7120(_QWORD *a1, unsigned __int8 *a2, size_t a3, __off_t a4)
{
  __off_t v6; // r13

  v6 = (*(__int64 (__fastcall **)(_QWORD *))(*a1 + 80LL))(a1) + a1[4] - a1[2];
  sub_CB70C0((__int64)a1, a4);
  sub_CB6200((__int64)a1, a2, a3);
  return sub_CB70C0((__int64)a1, v6);
}
