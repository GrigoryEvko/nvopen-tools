// Function: sub_3970BC0
// Address: 0x3970bc0
//
__int64 __fastcall sub_3970BC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // r12
  _QWORD v7[2]; // [rsp+0h] [rbp-A0h] BYREF
  _QWORD v8[2]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v9; // [rsp+20h] [rbp-80h]
  unsigned __int64 v10[2]; // [rsp+30h] [rbp-70h] BYREF
  _BYTE v11[96]; // [rsp+40h] [rbp-60h] BYREF

  v7[0] = a2;
  v7[1] = a3;
  v10[0] = (unsigned __int64)v11;
  v10[1] = 0x3C00000000LL;
  v3 = sub_396DDB0(a1);
  v9 = 261;
  v8[0] = v7;
  sub_38B9930((__int64)v10, (__int64)v8, v3);
  v4 = *(_QWORD *)(a1 + 248);
  v8[0] = v10;
  v9 = 262;
  v5 = sub_38BF510(v4, (__int64)v8);
  if ( (_BYTE *)v10[0] != v11 )
    _libc_free(v10[0]);
  return v5;
}
