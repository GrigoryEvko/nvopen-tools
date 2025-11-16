// Function: sub_9717D0
// Address: 0x9717d0
//
__int64 __fastcall sub_9717D0(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 result; // rax
  __int64 v4; // [rsp+8h] [rbp-18h]
  __int64 v5; // [rsp+10h] [rbp-10h] BYREF
  unsigned int v6; // [rsp+18h] [rbp-8h]

  v6 = 64;
  v5 = 0;
  result = sub_9714E0(a1, a2, (__int64)&v5, a3);
  if ( v6 > 0x40 )
  {
    if ( v5 )
    {
      v4 = result;
      j_j___libc_free_0_0(v5);
      return v4;
    }
  }
  return result;
}
