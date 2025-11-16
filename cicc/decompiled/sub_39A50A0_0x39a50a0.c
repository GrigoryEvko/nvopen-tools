// Function: sub_39A50A0
// Address: 0x39a50a0
//
void __fastcall sub_39A50A0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // rsi
  unsigned __int64 v5; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v6; // [rsp+8h] [rbp-28h]

  v4 = (__int64 *)(a3 + 32);
  if ( *(void **)(a3 + 32) == sub_16982C0() )
    sub_169D930((__int64)&v5, (__int64)v4);
  else
    sub_169D7E0((__int64)&v5, v4);
  sub_39A4F50(a1, a2, (__int64 *)&v5, 1);
  if ( v6 > 0x40 )
  {
    if ( v5 )
      j_j___libc_free_0_0(v5);
  }
}
