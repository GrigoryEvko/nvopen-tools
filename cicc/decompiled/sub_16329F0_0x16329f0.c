// Function: sub_16329F0
// Address: 0x16329f0
//
__int64 __fastcall sub_16329F0(__int64 **a1, unsigned int a2, void *a3, size_t a4, __int64 a5)
{
  __int64 v9; // rax
  __int64 v10; // rax
  _QWORD *v11; // rax
  __int64 *v12; // rdi
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v16[10]; // [rsp+0h] [rbp-50h] BYREF

  v9 = sub_1643350(*a1);
  v10 = sub_15A0680(v9, a2, 0);
  v11 = sub_1624210(v10);
  v12 = *a1;
  v16[0] = (__int64)v11;
  v16[2] = a5;
  v16[1] = sub_161FF10(v12, a3, a4);
  v13 = sub_16329D0((__int64)a1);
  v14 = sub_1627350(*a1, v16, (__int64 *)3, 0, 1);
  return sub_1623CA0(v13, v14);
}
