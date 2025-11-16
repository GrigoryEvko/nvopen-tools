// Function: sub_A79A50
// Address: 0xa79a50
//
__int64 __fastcall sub_A79A50(__int64 *a1, unsigned __int64 *a2, __int64 a3)
{
  unsigned __int64 *v5; // rbx
  unsigned __int64 *v6; // r14
  unsigned __int64 *v7; // rdi
  _QWORD *v8; // rsi
  __int64 v9; // rbx
  __int64 v11; // rax
  __int64 v12; // [rsp+8h] [rbp-E8h]
  __int64 v13; // [rsp+18h] [rbp-D8h]
  __int64 v14; // [rsp+28h] [rbp-C8h] BYREF
  _QWORD v15[2]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE v16[176]; // [rsp+40h] [rbp-B0h] BYREF

  if ( !a3 )
    return 0;
  v13 = *a1;
  v15[0] = v16;
  v15[1] = 0x2000000000LL;
  v5 = &a2[a3];
  v12 = 8 * a3;
  if ( v5 != a2 )
  {
    v6 = a2;
    do
    {
      v7 = v6++;
      sub_A718C0(v7, (__int64)v15);
    }
    while ( v5 != v6 );
  }
  v8 = v15;
  v9 = sub_C65B40(v13 + 432, v15, &v14, off_49D9A90);
  if ( !v9 )
  {
    v11 = sub_22077B0(v12 + 64);
    v9 = v11;
    if ( v11 )
      sub_A79560(v11, a2, a3);
    v8 = (_QWORD *)v9;
    sub_C657C0(v13 + 432, v9, v14, off_49D9A90);
  }
  if ( (_BYTE *)v15[0] != v16 )
    _libc_free(v15[0], v8);
  return v9;
}
