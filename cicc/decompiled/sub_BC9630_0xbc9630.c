// Function: sub_BC9630
// Address: 0xbc9630
//
__int64 __fastcall sub_BC9630(__int64 *a1, const char *a2, __int64 a3)
{
  __int64 v4; // r14
  size_t v5; // rax
  __int64 v6; // rax
  __int64 v8[6]; // [rsp+0h] [rbp-30h] BYREF

  v4 = sub_BCB2E0(a1);
  v5 = strlen(a2);
  v8[0] = sub_B9B140(a1, a2, v5);
  v6 = sub_AD64C0(v4, a3, 0);
  v8[1] = (__int64)sub_B98A20(v6, a3);
  return sub_B9C770(a1, v8, (__int64 *)2, 0, 1);
}
