// Function: sub_155F990
// Address: 0x155f990
//
__int64 __fastcall sub_155F990(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // r13
  __int64 v6; // r12
  __int64 v8; // rax
  __int64 v9; // [rsp+18h] [rbp-C8h] BYREF
  unsigned __int64 v10[2]; // [rsp+20h] [rbp-C0h] BYREF
  _BYTE v11[176]; // [rsp+30h] [rbp-B0h] BYREF

  v4 = *a1;
  v10[1] = 0x2000000000LL;
  v5 = v4 + 224;
  v10[0] = (unsigned __int64)v11;
  sub_155F930((__int64)v10, a2, a3);
  v6 = sub_16BDDE0(v5, v10, &v9);
  if ( !v6 )
  {
    v8 = sub_22077B0(8 * a3 + 32);
    v6 = v8;
    if ( v8 )
      sub_155F870(v8, (__int64)a1, a2, a3);
    sub_16BDA20(v5, v6, v9);
  }
  if ( (_BYTE *)v10[0] != v11 )
    _libc_free(v10[0]);
  return v6;
}
