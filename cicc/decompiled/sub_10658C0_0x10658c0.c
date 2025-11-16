// Function: sub_10658C0
// Address: 0x10658c0
//
_QWORD *__fastcall sub_10658C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *result; // rax
  _QWORD *v7; // [rsp+8h] [rbp-68h]
  __int64 v8; // [rsp+10h] [rbp-60h] BYREF
  char *v9; // [rsp+18h] [rbp-58h]
  __int64 v10; // [rsp+20h] [rbp-50h]
  int v11; // [rsp+28h] [rbp-48h]
  char v12; // [rsp+2Ch] [rbp-44h]
  char v13; // [rsp+30h] [rbp-40h] BYREF

  v8 = 0;
  v9 = &v13;
  v10 = 8;
  v11 = 0;
  v12 = 1;
  result = sub_10648E0(a1, a2, (__int64)&v8, a4, a5, a6);
  if ( !v12 )
  {
    v7 = result;
    _libc_free(v9, a2);
    return v7;
  }
  return result;
}
