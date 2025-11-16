// Function: sub_15E83D0
// Address: 0x15e83d0
//
_QWORD *__fastcall sub_15E83D0(__int64 *a1, int a2, __int64 *a3, __int64 a4, int a5)
{
  __int64 *v8; // rax
  __int64 v9; // rax
  __int64 v11[8]; // [rsp+10h] [rbp-40h] BYREF

  v8 = (__int64 *)sub_157EB90(a1[1]);
  v11[0] = *a3;
  v9 = sub_15E26F0(v8, a2, v11, 1);
  v11[0] = (__int64)a3;
  v11[1] = a4;
  return sub_15E6DE0(v9, (int)v11, 2, a1, a5, 0, 0, 0);
}
