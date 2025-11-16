// Function: sub_C49DE0
// Address: 0xc49de0
//
__int64 __fastcall sub_C49DE0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // eax
  unsigned __int64 v5; // rdx
  bool v6; // [rsp+Fh] [rbp-21h] BYREF
  __int64 v7; // [rsp+10h] [rbp-20h] BYREF
  unsigned int v8; // [rsp+18h] [rbp-18h]

  sub_C49BE0((__int64)&v7, a2, a3, &v6);
  if ( !v6 )
  {
    *(_DWORD *)(a1 + 8) = v8;
    *(_QWORD *)a1 = v7;
    return a1;
  }
  v4 = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a1 + 8) = v4;
  if ( v4 > 0x40 )
  {
    sub_C43690(a1, -1, 1);
  }
  else
  {
    v5 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v4;
    if ( !v4 )
      v5 = 0;
    *(_QWORD *)a1 = v5;
  }
  if ( v8 <= 0x40 || !v7 )
    return a1;
  j_j___libc_free_0_0(v7);
  return a1;
}
