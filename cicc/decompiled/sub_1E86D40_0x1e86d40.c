// Function: sub_1E86D40
// Address: 0x1e86d40
//
__int64 __fastcall sub_1E86D40(__int64 a1, const char *a2, __int64 a3, unsigned int a4, __int64 a5)
{
  void *v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r13
  void *v12; // rax
  void *v13; // rax

  sub_1E86C30(a1, a2, *(_QWORD *)(a3 + 16));
  v8 = sub_16E8CB0();
  v9 = sub_1263B40((__int64)v8, "- operand ");
  v10 = sub_16E7A90(v9, a4);
  sub_1263B40(v10, ":   ");
  v11 = *(_QWORD *)(a1 + 40);
  v12 = sub_16E8CB0();
  sub_1E33FC0(a3, (__int64)v12, a5, v11, 0);
  v13 = sub_16E8CB0();
  return sub_1263B40((__int64)v13, "\n");
}
