// Function: sub_FE7C90
// Address: 0xfe7c90
//
_QWORD *__fastcall sub_FE7C90(__int64 a1, void *a2, void *a3)
{
  _QWORD *v3; // rdi
  _QWORD *result; // rax
  __int64 v5; // [rsp+8h] [rbp-B8h] BYREF
  _BYTE *v6; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v7; // [rsp+18h] [rbp-A8h]
  _QWORD v8[2]; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v9[2]; // [rsp+30h] [rbp-90h] BYREF
  _QWORD v10[2]; // [rsp+40h] [rbp-80h] BYREF
  void *v11[4]; // [rsp+50h] [rbp-70h] BYREF
  __int16 v12; // [rsp+70h] [rbp-50h]
  void *v13; // [rsp+80h] [rbp-40h] BYREF
  __int16 v14; // [rsp+A0h] [rbp-20h]

  v11[1] = a3;
  v11[0] = a2;
  v5 = a1;
  v14 = 257;
  v12 = 261;
  v9[0] = (__int64)v10;
  sub_FDB1F0(v9, byte_3F871B3, (__int64)byte_3F871B3);
  sub_FE7580((__int64)&v6, (size_t)&v5, v11, 0, &v13, (__int64)v9);
  if ( (_QWORD *)v9[0] != v10 )
    j_j___libc_free_0(v9[0], v10[0] + 1LL);
  v3 = v6;
  if ( v7 )
  {
    sub_C67930(v6, v7, 0, 0);
    v3 = v6;
  }
  result = v8;
  if ( v3 != v8 )
    return (_QWORD *)j_j___libc_free_0(v3, v8[0] + 1LL);
  return result;
}
