// Function: sub_158A9F0
// Address: 0x158a9f0
//
__int64 __fastcall sub_158A9F0(__int64 a1, __int64 a2)
{
  unsigned int v2; // ecx
  __int64 v4; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v5; // [rsp+8h] [rbp-28h]

  if ( sub_158A0B0(a2) || sub_158A670(a2) )
  {
    v2 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 8) = v2;
    if ( v2 <= 0x40 )
      *(_QWORD *)a1 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
    else
      sub_16A4EF0(a1, -1, 1);
    return a1;
  }
  else
  {
    v5 = *(_DWORD *)(a2 + 24);
    if ( v5 > 0x40 )
      sub_16A4FD0(&v4, a2 + 16);
    else
      v4 = *(_QWORD *)(a2 + 16);
    sub_16A7800(&v4, 1);
    *(_DWORD *)(a1 + 8) = v5;
    *(_QWORD *)a1 = v4;
    return a1;
  }
}
