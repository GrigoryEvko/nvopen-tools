// Function: sub_161C8C0
// Address: 0x161c8c0
//
__int64 __fastcall sub_161C8C0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v10; // [rsp+10h] [rbp-30h] BYREF
  __int64 v11; // [rsp+18h] [rbp-28h]

  v10 = 0;
  v11 = 0;
  v3 = sub_161BD10(a1, (__int64)"loop_header_weight", 18);
  v4 = *a1;
  v10 = v3;
  v5 = sub_1643360(v4);
  v6 = sub_159C470(v5, a2, 0);
  v11 = sub_161BD20((__int64)a1, v6, v7, v8);
  return sub_1627350(*a1, &v10, 2, 0, 1);
}
