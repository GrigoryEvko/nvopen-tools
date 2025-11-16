// Function: sub_C49420
// Address: 0xc49420
//
__int64 __fastcall sub_C49420(__int64 a1, __int64 a2, char a3)
{
  _BYTE *v4; // rsi
  __int64 result; // rax
  __int64 v6; // [rsp-8h] [rbp-60h]
  _BYTE *v7; // [rsp+8h] [rbp-50h] BYREF
  __int64 v8; // [rsp+10h] [rbp-48h]
  __int64 v9; // [rsp+18h] [rbp-40h]
  _BYTE v10[56]; // [rsp+20h] [rbp-38h] BYREF

  v7 = v10;
  v8 = 0;
  v9 = 40;
  sub_C48A30(a1, &v7, 0xAu, a3, 0, 1, 0);
  v4 = v7;
  sub_CB6200(a2, v7, v8);
  result = v6;
  if ( v7 != v10 )
    return _libc_free(v7, v4);
  return result;
}
