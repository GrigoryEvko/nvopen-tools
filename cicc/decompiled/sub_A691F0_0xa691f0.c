// Function: sub_A691F0
// Address: 0xa691f0
//
__int64 (__fastcall *__fastcall sub_A691F0(__int64 a1, __int64 a2, char a3))(_QWORD *, _QWORD *, __int64)
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rsi
  _QWORD v9[18]; // [rsp+0h] [rbp-90h] BYREF

  v5 = *(_QWORD *)(a1 + 16);
  if ( v5 && sub_B14170(*(_QWORD *)(a1 + 16)) && (v6 = *(_QWORD *)(sub_B14170(v5) + 72)) != 0 )
    v7 = *(_QWORD *)(v6 + 40);
  else
    v7 = 0;
  sub_A558A0((__int64)v9, v7, 1);
  sub_A690C0(a1, a2, (__int64)v9, a3);
  return sub_A55520(v9, a2);
}
