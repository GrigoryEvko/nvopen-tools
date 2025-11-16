// Function: sub_15E8450
// Address: 0x15e8450
//
_QWORD *__fastcall sub_15E8450(__int64 *a1, int a2, __int64 **a3, __int64 a4, __int64 a5, int a6)
{
  __int64 *v10; // rax
  __int64 v11; // rax
  __int64 v13[7]; // [rsp+18h] [rbp-38h] BYREF

  v10 = (__int64 *)sub_157EB90(a1[1]);
  v13[0] = **a3;
  v11 = sub_15E26F0(v10, a2, v13, 1);
  return sub_15E6DE0(v11, (int)a3, a4, a1, a6, a5, 0, 0);
}
