// Function: sub_161C2E0
// Address: 0x161c2e0
//
__int64 __fastcall sub_161C2E0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  _QWORD v7[6]; // [rsp+0h] [rbp-30h] BYREF

  v7[0] = sub_161BD10(a1, (__int64)"function_section_prefix", 23);
  v4 = sub_161BD10(a1, a2, a3);
  v5 = *a1;
  v7[1] = v4;
  return sub_1627350(v5, v7, 2, 0, 1);
}
