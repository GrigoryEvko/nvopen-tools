// Function: sub_324A320
// Address: 0x324a320
//
void __fastcall sub_324A320(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // rsi
  unsigned __int64 v5; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v6; // [rsp+8h] [rbp-28h]

  v4 = (__int64 *)(a3 + 24);
  if ( *(void **)(a3 + 24) == sub_C33340() )
    sub_C3E660((__int64)&v5, (__int64)v4);
  else
    sub_C3A850((__int64)&v5, v4);
  sub_324A2D0(a1, a2, (__int64)&v5, 1);
  if ( v6 > 0x40 )
  {
    if ( v5 )
      j_j___libc_free_0_0(v5);
  }
}
