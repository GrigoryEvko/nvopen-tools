// Function: sub_15E8360
// Address: 0x15e8360
//
_QWORD *__fastcall sub_15E8360(__int64 *a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v8; // [rsp+0h] [rbp-30h] BYREF
  __int64 v9; // [rsp+8h] [rbp-28h] BYREF

  v5 = a1[1];
  v8 = a3;
  v6 = sub_15E26F0(*(__int64 **)(*(_QWORD *)(v5 + 56) + 40LL), 77, &v8, 1);
  v9 = a2;
  return sub_15E6DE0(v6, (int)&v9, 1, a1, a4, 0, 0, 0);
}
