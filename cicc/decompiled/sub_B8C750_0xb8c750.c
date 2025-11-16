// Function: sub_B8C750
// Address: 0xb8c750
//
__int64 __fastcall sub_B8C750(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  _QWORD v7[6]; // [rsp+0h] [rbp-30h] BYREF

  v7[0] = sub_B8C130(a1, (__int64)"section_prefix", 14);
  v4 = sub_B8C130(a1, a2, a3);
  v5 = *a1;
  v7[1] = v4;
  return sub_B9C770(v5, v7, 2, 0, 1);
}
