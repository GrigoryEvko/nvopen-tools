// Function: sub_A61DE0
// Address: 0xa61de0
//
__int64 (__fastcall *__fastcall sub_A61DE0(const char *a1, __int64 a2, __int64 a3))(_QWORD *, _QWORD *, __int64)
{
  _QWORD v5[18]; // [rsp+0h] [rbp-90h] BYREF

  sub_A558A0((__int64)v5, a3, (unsigned __int8)(*a1 - 5) <= 0x1Fu);
  sub_A619C0(a2, a1, (__int64)v5, a3, 0, 0);
  return sub_A55520(v5, (__int64)a1);
}
