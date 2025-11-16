// Function: sub_159C470
// Address: 0x159c470
//
__int64 __fastcall sub_159C470(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  unsigned int v3; // ecx
  __int64 result; // rax
  __int64 v5; // [rsp+8h] [rbp-28h]
  unsigned __int64 v6; // [rsp+10h] [rbp-20h] BYREF
  unsigned int v7; // [rsp+18h] [rbp-18h]

  v3 = *(_DWORD *)(a1 + 8) >> 8;
  v7 = v3;
  if ( v3 > 0x40 )
    sub_16A4EF0(&v6, a2, a3);
  else
    v6 = a2 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v3);
  result = sub_159C0E0(*(__int64 **)a1, (__int64)&v6);
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
