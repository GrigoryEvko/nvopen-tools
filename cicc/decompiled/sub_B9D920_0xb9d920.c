// Function: sub_B9D920
// Address: 0xb9d920
//
void __fastcall sub_B9D920(__int64 a1, unsigned int a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 *v5; // rax
  __int64 v6; // rax
  __int64 v7[3]; // [rsp+8h] [rbp-18h] BYREF

  sub_B98000(a1, 28);
  v2 = sub_BD5C60(a1, 28);
  v3 = sub_BCB2E0(v2);
  v4 = sub_ACD640(v3, a2, 0);
  v7[0] = (__int64)sub_B98A20(v4, a2);
  v5 = (__int64 *)sub_BD5C60(a1, a2);
  v6 = sub_B9C770(v5, v7, (__int64 *)1, 0, 1);
  sub_B994D0(a1, 28, v6);
}
