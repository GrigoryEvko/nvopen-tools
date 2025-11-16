// Function: sub_DCAF50
// Address: 0xdcaf50
//
__int64 *__fastcall sub_DCAF50(__int64 *a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 *v8; // r12
  _QWORD v9[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v10[6]; // [rsp+10h] [rbp-30h] BYREF

  if ( *(_WORD *)(a2 + 24) )
  {
    v6 = sub_D95540(a2);
    v7 = sub_D97090((__int64)a1, v6);
    v10[1] = sub_DA2C50((__int64)a1, v7, -1, 1u);
    v9[0] = v10;
    v10[0] = a2;
    v9[1] = 0x200000002LL;
    v8 = sub_DC8BD0(a1, (__int64)v9, a3, 0);
    if ( (_QWORD *)v9[0] != v10 )
      _libc_free(v9[0], v9);
    return v8;
  }
  else
  {
    v3 = sub_AD6890(*(_QWORD *)(a2 + 32), 0);
    return sub_DA2570((__int64)a1, v3);
  }
}
