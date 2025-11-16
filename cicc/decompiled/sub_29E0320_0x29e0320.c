// Function: sub_29E0320
// Address: 0x29e0320
//
__int64 __fastcall sub_29E0320(__int64 a1, __int64 a2)
{
  int v2; // ebx
  _QWORD *v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 *v6; // rax
  __int64 v8[6]; // [rsp+0h] [rbp-30h] BYREF

  v2 = sub_CEFE00();
  v3 = (_QWORD *)sub_B2BE50(a1);
  v4 = sub_BCB2D0(v3);
  v5 = sub_ACD640(v4, v2, 0);
  v8[0] = a2;
  v8[1] = (__int64)sub_B98A20(v5, v2);
  v6 = (__int64 *)sub_B2BE50(a1);
  return sub_B9C770(v6, v8, (__int64 *)2, 0, 1);
}
