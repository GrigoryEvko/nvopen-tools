// Function: sub_DD21F0
// Address: 0xdd21f0
//
_QWORD *__fastcall sub_DD21F0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  _QWORD *v5; // rax
  _QWORD *v6; // rbx
  _QWORD *v7; // rax
  _QWORD *v8; // r12
  _QWORD v10[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v11[6]; // [rsp+10h] [rbp-30h] BYREF

  v4 = sub_D95540(a2);
  v5 = sub_DA2C50((__int64)a1, v4, 1, 0);
  v6 = sub_DCEE80(a1, a2, (__int64)v5, 0);
  v7 = sub_DCC810(a1, a2, (__int64)v6, 0, 0);
  v11[1] = sub_DCB270((__int64)a1, (__int64)v7, a3);
  v10[0] = v11;
  v11[0] = v6;
  v10[1] = 0x200000002LL;
  v8 = sub_DC7EB0(a1, (__int64)v10, 0, 0);
  if ( (_QWORD *)v10[0] != v11 )
    _libc_free(v10[0], v10);
  return v8;
}
