// Function: sub_B8D200
// Address: 0xb8d200
//
__int64 __fastcall sub_B8D200(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rdi
  _QWORD v12[4]; // [rsp+0h] [rbp-20h] BYREF

  v3 = sub_B8C130(a1, (__int64)"loop_header_weight", 18);
  v4 = *a1;
  v12[0] = v3;
  v5 = sub_BCB2E0(v4);
  v6 = sub_ACD640(v5, a2, 0);
  v9 = sub_B8C140((__int64)a1, v6, v7, v8);
  v10 = *a1;
  v12[1] = v9;
  return sub_B9C770(v10, v12, 2, 0, 1);
}
