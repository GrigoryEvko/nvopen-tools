// Function: sub_DF6950
// Address: 0xdf6950
//
__int64 __fastcall sub_DF6950(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // [rsp+10h] [rbp-20h] BYREF
  unsigned int v6; // [rsp+18h] [rbp-18h]
  __int64 v7; // [rsp+20h] [rbp-10h]
  __int64 v8; // [rsp+28h] [rbp-8h]

  v6 = *(_DWORD *)(a4 + 8);
  if ( v6 <= 0x40 )
  {
    LOBYTE(v8) = 0;
    return v7;
  }
  sub_C43780((__int64)&v5, (const void **)a4);
  LOBYTE(v8) = 0;
  if ( v6 <= 0x40 || !v5 )
    return v7;
  j_j___libc_free_0_0(v5);
  return v7;
}
