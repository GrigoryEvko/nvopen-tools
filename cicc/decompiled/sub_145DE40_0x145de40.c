// Function: sub_145DE40
// Address: 0x145de40
//
_QWORD *__fastcall sub_145DE40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  _QWORD *v5; // r12
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // [rsp+10h] [rbp-E0h]
  __int64 v10; // [rsp+18h] [rbp-D8h]
  __int64 v11; // [rsp+28h] [rbp-C8h] BYREF
  unsigned __int64 v12[2]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE v13[176]; // [rsp+40h] [rbp-B0h] BYREF

  v12[1] = 0x2000000000LL;
  v12[0] = (unsigned __int64)v13;
  sub_16BD3E0(v12, 1);
  sub_16BD4C0(v12, a2);
  sub_16BD4C0(v12, a3);
  v11 = 0;
  v4 = sub_16BDDE0(a1 + 840, v12, &v11);
  if ( v4 )
  {
    v5 = (_QWORD *)(v4 - 8);
  }
  else
  {
    v7 = sub_16BD760(v12, a1 + 864);
    v10 = v8;
    v9 = v7;
    v5 = (_QWORD *)sub_145CBF0((__int64 *)(a1 + 864), 56, 16);
    sub_1458540(v5, v9, v10, a2, a3);
    sub_16BDA20(a1 + 840, v5 + 1, v11);
  }
  if ( (_BYTE *)v12[0] != v13 )
    _libc_free(v12[0]);
  return v5;
}
