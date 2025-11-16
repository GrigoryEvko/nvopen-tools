// Function: sub_DCF070
// Address: 0xdcf070
//
_QWORD *__fastcall sub_DCF070(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // r12
  _QWORD v6[2]; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v7[4]; // [rsp+10h] [rbp-20h] BYREF

  v7[1] = a3;
  v6[0] = v7;
  v7[0] = a2;
  v6[1] = 0x200000002LL;
  v4 = sub_DCEEE0(a1, (__int64)v6, a4, a4, a2);
  if ( (_QWORD *)v6[0] != v7 )
    _libc_free(v6[0], v6);
  return v4;
}
