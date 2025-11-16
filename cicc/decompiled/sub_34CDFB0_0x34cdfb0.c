// Function: sub_34CDFB0
// Address: 0x34cdfb0
//
unsigned __int64 __fastcall sub_34CDFB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v5; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v6; // [rsp+18h] [rbp-28h]
  unsigned __int64 v7; // [rsp+20h] [rbp-20h] BYREF
  __int64 v8; // [rsp+28h] [rbp-18h]

  v6 = *(_DWORD *)(a4 + 8);
  if ( v6 <= 0x40 || (sub_C43780((__int64)&v5, (const void **)a4), LODWORD(v8) = v6, v6 <= 0x40) )
  {
    LOBYTE(v8) = 0;
    return v7;
  }
  sub_C43780((__int64)&v7, (const void **)&v5);
  if ( (unsigned int)v8 > 0x40 && v7 )
    j_j___libc_free_0_0(v7);
  LOBYTE(v8) = 0;
  if ( v6 <= 0x40 || !v5 )
    return v7;
  j_j___libc_free_0_0(v5);
  return v7;
}
