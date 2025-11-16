// Function: sub_C65C80
// Address: 0xc65c80
//
_QWORD *__fastcall sub_C65C80(__int64 *a1, __int64 *a2, void (__fastcall **a3)(__int64 *, __int64 *, _QWORD *))
{
  __int64 *v5; // rsi
  _QWORD *v6; // r12
  __int64 *v8; // [rsp+8h] [rbp-C8h] BYREF
  _QWORD v9[2]; // [rsp+10h] [rbp-C0h] BYREF
  _BYTE v10[176]; // [rsp+20h] [rbp-B0h] BYREF

  v9[1] = 0x2000000000LL;
  v9[0] = v10;
  (*a3)(a1, a2, v9);
  v5 = v9;
  v6 = sub_C65B40((__int64)a1, (__int64)v9, (__int64 *)&v8, (__int64)a3);
  if ( !v6 )
  {
    v5 = a2;
    v6 = a2;
    sub_C657C0(a1, a2, v8, (__int64)a3);
  }
  if ( (_BYTE *)v9[0] != v10 )
    _libc_free(v9[0], v5);
  return v6;
}
