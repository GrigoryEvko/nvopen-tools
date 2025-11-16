// Function: sub_281DF50
// Address: 0x281df50
//
_QWORD *__fastcall sub_281DF50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  _QWORD *v8; // r13
  unsigned __int64 v10[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v11[8]; // [rsp+10h] [rbp-40h] BYREF

  v8 = sub_DC5760((__int64)a5, a2, a3, 0);
  if ( !sub_D96900(a4) )
  {
    v11[1] = sub_DC5760((__int64)a5, a4, a3, 0);
    v11[0] = v8;
    v10[0] = (unsigned __int64)v11;
    v10[1] = 0x200000002LL;
    v8 = sub_DC8BD0(a5, (__int64)v10, 2u, 0);
    if ( (_QWORD *)v10[0] != v11 )
      _libc_free(v10[0]);
  }
  return sub_DCC810(a5, a1, (__int64)v8, 0, 0);
}
