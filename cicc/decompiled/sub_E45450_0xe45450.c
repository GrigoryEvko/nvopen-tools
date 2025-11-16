// Function: sub_E45450
// Address: 0xe45450
//
__int64 (__fastcall *__fastcall sub_E45450(__int64 *a1, __int64 a2))(_QWORD *, _QWORD *, __int64)
{
  signed __int64 v3; // rsi
  _QWORD v5[18]; // [rsp+0h] [rbp-90h] BYREF

  sub_A558A0((__int64)v5, *(_QWORD *)(*(_QWORD *)(*a1 + 72) + 40LL), 0);
  sub_A564B0((__int64)v5, *(_QWORD *)(*a1 + 72));
  v3 = (int)sub_A5A720((__int64)v5, *a1);
  sub_CB59F0(a2, v3);
  return sub_A55520(v5, v3);
}
