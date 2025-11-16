// Function: sub_DCACD0
// Address: 0xdcacd0
//
_QWORD *__fastcall sub_DCACD0(__int64 *a1, __int64 a2, __int64 a3, char a4)
{
  _QWORD *v4; // r12
  _QWORD v7[2]; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v8[6]; // [rsp+20h] [rbp-30h] BYREF

  v4 = sub_DA2C50((__int64)a1, a2, a3, 0);
  if ( !a4 )
    return v4;
  v8[1] = sub_DA3710((__int64)a1, a2);
  v8[0] = v4;
  v7[0] = v8;
  v7[1] = 0x200000002LL;
  v4 = sub_DC8BD0(a1, (__int64)v7, 0, 0);
  if ( (_QWORD *)v7[0] == v8 )
    return v4;
  _libc_free(v7[0], v7);
  return v4;
}
