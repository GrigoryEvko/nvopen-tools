// Function: sub_11AEFE0
// Address: 0x11aefe0
//
__int64 __fastcall sub_11AEFE0(__int64 a1)
{
  unsigned int v1; // ebx
  __int64 *v2; // rax
  __int64 result; // rax
  unsigned int v4; // ebx
  __int64 v5; // [rsp+8h] [rbp-38h]
  __int64 v6; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v7; // [rsp+18h] [rbp-28h]

  v1 = *(_DWORD *)(a1 + 32);
  if ( v1 == 64 )
    return 0;
  if ( v1 <= 0x40 || (v4 = v1 - sub_C444A0(a1 + 24), result = 0, v4 <= 0x40) )
  {
    sub_C44AB0((__int64)&v6, a1 + 24, 0x40u);
    v2 = (__int64 *)sub_BD5C60(a1);
    result = sub_ACCFD0(v2, (__int64)&v6);
    if ( v7 > 0x40 )
    {
      if ( v6 )
      {
        v5 = result;
        j_j___libc_free_0_0(v6);
        return v5;
      }
    }
  }
  return result;
}
