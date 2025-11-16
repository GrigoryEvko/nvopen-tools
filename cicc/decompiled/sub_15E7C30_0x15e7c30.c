// Function: sub_15E7C30
// Address: 0x15e7c30
//
_QWORD *__fastcall sub_15E7C30(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 v4; // rax
  __int64 *v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  _QWORD v10[2]; // [rsp+0h] [rbp-60h] BYREF
  __int64 v11[4]; // [rsp+10h] [rbp-50h] BYREF
  char v12; // [rsp+30h] [rbp-30h] BYREF
  __int16 v13; // [rsp+40h] [rbp-20h]

  v4 = a1[1];
  v10[1] = a3;
  v10[0] = a2;
  v5 = *(__int64 **)(*(_QWORD *)(v4 + 56) + 40LL);
  v6 = *a3;
  v7 = **(_QWORD **)(*a3 + 16);
  v11[2] = v6;
  v11[0] = v7;
  v11[1] = *a2;
  v8 = sub_15E26F0(v5, 88, v11, 3);
  v13 = 257;
  return sub_15E6DE0(v8, (int)v10, 2, a1, (int)&v12, 0, 0, 0);
}
