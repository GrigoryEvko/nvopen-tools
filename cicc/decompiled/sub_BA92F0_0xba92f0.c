// Function: sub_BA92F0
// Address: 0xba92f0
//
void __fastcall sub_BA92F0(__int64 **a1, unsigned int a2, const void *a3, size_t a4, __int64 a5)
{
  __int64 v9; // rax
  __int64 v10; // rax
  _QWORD *v11; // rax
  __int64 *v12; // rdi
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15[10]; // [rsp+0h] [rbp-50h] BYREF

  v9 = sub_BCB2D0(*a1);
  v10 = sub_AD64C0(v9, a2, 0);
  v11 = sub_B98A20(v10, a2);
  v12 = *a1;
  v15[0] = (__int64)v11;
  v15[2] = a5;
  v15[1] = sub_B9B140(v12, a3, a4);
  v13 = sub_BA92C0((__int64)a1);
  v14 = sub_B9C770(*a1, v15, (__int64 *)3, 0, 1);
  sub_B979A0(v13, v14);
}
