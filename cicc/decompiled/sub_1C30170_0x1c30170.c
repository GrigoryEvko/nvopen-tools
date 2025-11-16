// Function: sub_1C30170
// Address: 0x1c30170
//
void __fastcall sub_1C30170(__int64 a1, unsigned int a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 *v5; // rax
  __int64 v6; // rax
  __int64 v7[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = (_QWORD *)sub_16498A0(a1);
  v3 = sub_1643350(v2);
  v4 = sub_159C470(v3, a2, 0);
  v7[0] = (__int64)sub_1624210(v4);
  v5 = (__int64 *)sub_16498A0(a1);
  v6 = sub_1627350(v5, v7, (__int64 *)1, 0, 1);
  sub_1626100(a1, "nv.used_bytes_mask", 0x12u, v6);
}
