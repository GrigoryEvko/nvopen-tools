// Function: sub_1E85BF0
// Address: 0x1e85bf0
//
__int64 __fastcall sub_1E85BF0(unsigned int *a1)
{
  void *v1; // rax
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // r12
  _QWORD v6[3]; // [rsp+8h] [rbp-18h] BYREF

  v1 = sub_16E8CB0();
  v2 = sub_1263B40((__int64)v1, "- ValNo:       ");
  v3 = sub_16E7A90(v2, *a1);
  v4 = sub_1263B40(v3, " (def ");
  v6[0] = *((_QWORD *)a1 + 1);
  sub_1F10810(v6, v4);
  return sub_1263B40(v4, ")\n");
}
