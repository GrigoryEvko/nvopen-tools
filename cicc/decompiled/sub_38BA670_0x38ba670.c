// Function: sub_38BA670
// Address: 0x38ba670
//
void *__fastcall sub_38BA670(unsigned __int64 a1, __int64 a2, __int64 a3)
{
  _QWORD v5[4]; // [rsp+0h] [rbp-50h] BYREF
  int v6; // [rsp+20h] [rbp-30h]
  __int64 v7; // [rsp+28h] [rbp-28h]

  v7 = a2;
  v6 = 1;
  v5[0] = &unk_49EFC48;
  memset(&v5[1], 0, 24);
  sub_16E7A40((__int64)v5, 0, 0, 0);
  sub_38B9BB0(a1, (__int64)v5, a3);
  v5[0] = &unk_49EFD28;
  return sub_16E7960((__int64)v5);
}
