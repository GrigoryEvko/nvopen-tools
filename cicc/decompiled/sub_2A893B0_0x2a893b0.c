// Function: sub_2A893B0
// Address: 0x2a893b0
//
__int64 __fastcall sub_2A893B0(__int64 a1, unsigned int a2, __int64 a3, _BYTE *a4)
{
  __int64 result; // rax
  __int64 v5; // [rsp+8h] [rbp-18h]
  unsigned __int64 v6; // [rsp+10h] [rbp-10h] BYREF
  unsigned int v7; // [rsp+18h] [rbp-8h]

  v7 = 32;
  v6 = a2;
  result = sub_9714E0(a1, a3, (__int64)&v6, a4);
  if ( v7 > 0x40 )
  {
    if ( v6 )
    {
      v5 = result;
      j_j___libc_free_0_0(v6);
      return v5;
    }
  }
  return result;
}
