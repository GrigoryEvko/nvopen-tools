// Function: sub_30B8650
// Address: 0x30b8650
//
void __fastcall sub_30B8650(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r8
  _BYTE *v10; // [rsp+10h] [rbp-60h] BYREF
  __int64 v11; // [rsp+18h] [rbp-58h]
  _BYTE v12[80]; // [rsp+20h] [rbp-50h] BYREF

  v10 = v12;
  v11 = 0x400000000LL;
  sub_30B6B70(a1, a2, (unsigned int *)&v10, a4, (__int64)&v10, a6);
  if ( (_DWORD)v11 )
  {
    sub_30B7CA0(a1, (__int64)&v10, a4, a5);
    if ( *(_DWORD *)(a4 + 8) )
      sub_30B84A0((__int64)a1, a2, a3, a4, v8);
  }
  if ( v10 != v12 )
    _libc_free((unsigned __int64)v10);
}
