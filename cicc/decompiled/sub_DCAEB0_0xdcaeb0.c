// Function: sub_DCAEB0
// Address: 0xdcaeb0
//
_QWORD *__fastcall sub_DCAEB0(__int64 *a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // r12
  char v5; // [rsp+Ch] [rbp-44h]
  _QWORD v6[2]; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v7[6]; // [rsp+20h] [rbp-30h] BYREF

  v5 = BYTE4(a3);
  v3 = sub_DA2C50((__int64)a1, a2, (unsigned int)a3, 0);
  if ( !v5 )
    return v3;
  v7[1] = sub_DA3710((__int64)a1, a2);
  v7[0] = v3;
  v6[0] = v7;
  v6[1] = 0x200000002LL;
  v3 = sub_DC8BD0(a1, (__int64)v6, 0, 0);
  if ( (_QWORD *)v6[0] == v7 )
    return v3;
  _libc_free(v6[0], v6);
  return v3;
}
