// Function: sub_25BFB50
// Address: 0x25bfb50
//
__int64 __fastcall sub_25BFB50(__int64 a1, __int64 a2)
{
  unsigned int v2; // ebx
  __int64 v4; // [rsp+0h] [rbp-80h] BYREF
  __int64 v5; // [rsp+8h] [rbp-78h]
  __int64 v6; // [rsp+10h] [rbp-70h]
  __int64 v7; // [rsp+18h] [rbp-68h]
  _BYTE *v8; // [rsp+20h] [rbp-60h]
  __int64 v9; // [rsp+28h] [rbp-58h]
  _BYTE v10[80]; // [rsp+30h] [rbp-50h] BYREF

  v4 = 0;
  v5 = 0;
  v6 = 0;
  v7 = 0;
  v8 = v10;
  v9 = 0x800000000LL;
  v2 = sub_25BF070(a1, 1, a2, (__int64)&v4);
  if ( v8 != v10 )
    _libc_free((unsigned __int64)v8);
  sub_C7D6A0(v5, 8LL * (unsigned int)v7, 8);
  return v2;
}
