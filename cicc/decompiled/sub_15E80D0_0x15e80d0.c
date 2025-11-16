// Function: sub_15E80D0
// Address: 0x15e80d0
//
_QWORD *__fastcall sub_15E80D0(__int64 *a1, __int64 a2, __int64 *a3, unsigned int a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v13[2]; // [rsp+0h] [rbp-70h] BYREF
  _BYTE v14[16]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v15; // [rsp+20h] [rbp-50h]
  _QWORD v16[8]; // [rsp+30h] [rbp-40h] BYREF

  v8 = *a3;
  v9 = a1[3];
  v16[0] = a2;
  v16[1] = a3;
  v10 = *(_QWORD *)(v8 + 24);
  v13[1] = v8;
  v13[0] = v10;
  v11 = sub_1643350(v9);
  v16[2] = sub_159C470(v11, a4, 0);
  v15 = 257;
  v16[3] = a5;
  return sub_15E7FB0(a1, 131, (int)v16, 4, v13, 2, (__int64)v14);
}
