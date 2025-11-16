// Function: sub_BD7E80
// Address: 0xbd7e80
//
__int64 __fastcall sub_BD7E80(unsigned __int8 *a1, unsigned __int8 *a2, __int64 *a3)
{
  __int64 *v4; // r14
  __int64 *i; // r15
  __int64 *v6; // r15
  __int64 v7; // r14
  __int64 *v9; // [rsp+18h] [rbp-B8h]
  __int64 *v10; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v11; // [rsp+28h] [rbp-A8h]
  _BYTE v12[48]; // [rsp+30h] [rbp-A0h] BYREF
  __int64 *v13; // [rsp+60h] [rbp-70h] BYREF
  __int64 v14; // [rsp+68h] [rbp-68h]
  _BYTE v15[96]; // [rsp+70h] [rbp-60h] BYREF

  v10 = (__int64 *)v12;
  v11 = 0x600000000LL;
  v14 = 0x600000000LL;
  v13 = (__int64 *)v15;
  sub_AE7A50((__int64)&v10, (__int64)a1, (__int64)&v13);
  v4 = &v10[(unsigned int)v11];
  for ( i = v10; v4 != i; ++i )
  {
    if ( a3 != *(__int64 **)(*i + 40) )
      sub_B59720(*i, (__int64)a1, a2);
  }
  v6 = v13;
  v9 = &v13[(unsigned int)v14];
  if ( v13 != v9 )
  {
    do
    {
      v7 = *v6;
      if ( a3 != (__int64 *)sub_B14180(*(_QWORD *)(*v6 + 16)) )
        sub_B13360(v7, a1, a2, 0);
      ++v6;
    }
    while ( v9 != v6 );
    v9 = v13;
  }
  if ( v9 != (__int64 *)v15 )
    _libc_free(v9, v15);
  if ( v10 != (__int64 *)v12 )
    _libc_free(v10, v15);
  v13 = a3;
  return sub_BD79D0(
           (__int64 *)a1,
           (__int64 *)a2,
           (unsigned __int8 (__fastcall *)(__int64, __int64 *))sub_BD3050,
           (__int64)&v13);
}
