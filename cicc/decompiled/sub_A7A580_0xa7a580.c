// Function: sub_A7A580
// Address: 0xa7a580
//
__int64 __fastcall sub_A7A580(__int64 *a1, __int64 *a2, int a3)
{
  __int64 result; // rax
  __int64 v5; // rax
  __int64 v6; // [rsp+8h] [rbp-98h]
  __int64 *v7; // [rsp+10h] [rbp-90h] BYREF
  _BYTE *v8; // [rsp+18h] [rbp-88h]
  __int64 v9; // [rsp+20h] [rbp-80h]
  _BYTE v10[120]; // [rsp+28h] [rbp-78h] BYREF

  if ( (unsigned __int8)sub_A73170(a1, a3) )
    return *a1;
  v7 = a2;
  v8 = v10;
  v9 = 0x800000000LL;
  sub_A77B20(&v7, a3);
  v5 = sub_A7A280(a2, (__int64)&v7);
  result = sub_A7A4C0(a1, a2, v5);
  if ( v8 != v10 )
  {
    v6 = result;
    _libc_free(v8, a2);
    return v6;
  }
  return result;
}
