// Function: sub_158ABC0
// Address: 0x158abc0
//
__int64 __fastcall sub_158ABC0(__int64 a1, __int64 a2)
{
  unsigned int v2; // edx
  unsigned int v3; // r13d
  __int64 v4; // rbx
  __int64 v6; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v7; // [rsp+8h] [rbp-28h]

  if ( sub_158A0B0(a2) || (int)sub_16AEA10(a2, a2 + 16) > 0 )
  {
    v2 = *(_DWORD *)(a2 + 8);
    v3 = v2 - 1;
    *(_DWORD *)(a1 + 8) = v2;
    v4 = ~(1LL << ((unsigned __int8)v2 - 1));
    if ( v2 <= 0x40 )
    {
      *(_QWORD *)a1 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
    }
    else
    {
      sub_16A4EF0(a1, -1, 1);
      if ( *(_DWORD *)(a1 + 8) > 0x40u )
      {
        *(_QWORD *)(*(_QWORD *)a1 + 8LL * (v3 >> 6)) &= v4;
        return a1;
      }
    }
    *(_QWORD *)a1 &= v4;
    return a1;
  }
  else
  {
    v7 = *(_DWORD *)(a2 + 24);
    if ( v7 > 0x40 )
      sub_16A4FD0(&v6, a2 + 16);
    else
      v6 = *(_QWORD *)(a2 + 16);
    sub_16A7800(&v6, 1);
    *(_DWORD *)(a1 + 8) = v7;
    *(_QWORD *)a1 = v6;
    return a1;
  }
}
