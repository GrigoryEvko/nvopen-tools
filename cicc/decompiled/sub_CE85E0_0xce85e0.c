// Function: sub_CE85E0
// Address: 0xce85e0
//
void __fastcall sub_CE85E0(__int64 a1, unsigned int a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 *v5; // rax
  __int64 v6; // rax
  __int64 v7[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = (_QWORD *)sub_BD5C60(a1);
  v3 = sub_BCB2D0(v2);
  v4 = sub_ACD640(v3, a2, 0);
  v7[0] = (__int64)sub_B98A20(v4, a2);
  v5 = (__int64 *)sub_BD5C60(a1);
  v6 = sub_B9C770(v5, v7, (__int64 *)1, 0, 1);
  sub_B9A090(a1, "nv.used_bytes_mask", 0x12u, v6);
}
