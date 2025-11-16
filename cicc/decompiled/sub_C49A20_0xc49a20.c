// Function: sub_C49A20
// Address: 0xc49a20
//
__int64 __fastcall sub_C49A20(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned int v4; // eax
  bool v5; // [rsp+Fh] [rbp-21h] BYREF
  __int64 v6; // [rsp+10h] [rbp-20h] BYREF
  unsigned int v7; // [rsp+18h] [rbp-18h]

  sub_C499A0((__int64)&v6, a2, a3, &v5);
  if ( !v5 )
  {
    *(_DWORD *)(a1 + 8) = v7;
    *(_QWORD *)a1 = v6;
    return a1;
  }
  v4 = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a1 + 8) = v4;
  if ( v4 > 0x40 )
    sub_C43690(a1, 0, 0);
  else
    *(_QWORD *)a1 = 0;
  if ( v7 <= 0x40 || !v6 )
    return a1;
  j_j___libc_free_0_0(v6);
  return a1;
}
