// Function: sub_1490760
// Address: 0x1490760
//
void __fastcall sub_1490760(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __m128i a6, __m128i a7)
{
  _BYTE *v10; // [rsp+10h] [rbp-60h] BYREF
  __int64 v11; // [rsp+18h] [rbp-58h]
  _BYTE v12[80]; // [rsp+20h] [rbp-50h] BYREF

  v10 = v12;
  v11 = 0x400000000LL;
  sub_14857C0((__int64)a1, a2, (__int64 *)&v10, a6, a7);
  if ( (_DWORD)v11 )
  {
    sub_14900D0(a1, (__int64)&v10, a4, a5, a6, a7);
    if ( *(_DWORD *)(a4 + 8) )
      sub_14905F0(a1, (__int64)a2, a3, a4, a6, a7);
  }
  if ( v10 != v12 )
    _libc_free((unsigned __int64)v10);
}
