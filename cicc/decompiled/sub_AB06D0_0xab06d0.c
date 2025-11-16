// Function: sub_AB06D0
// Address: 0xab06d0
//
__int64 __fastcall sub_AB06D0(__int64 a1)
{
  unsigned int v1; // eax
  unsigned int v2; // r12d
  __int64 v4; // rdx
  __int64 v5; // rax

  LOBYTE(v1) = sub_AAF7D0(a1);
  if ( (_BYTE)v1 )
    return 1;
  v2 = v1;
  if ( !sub_AAF760(a1) && !sub_AB01B0(a1) )
  {
    v2 = *(_DWORD *)(a1 + 24);
    v4 = *(_QWORD *)(a1 + 16);
    v5 = 1LL << ((unsigned __int8)v2 - 1);
    if ( v2 > 0x40 )
    {
      if ( (*(_QWORD *)(v4 + 8LL * ((v2 - 1) >> 6)) & v5) == 0 )
      {
        LOBYTE(v2) = v2 == (unsigned int)sub_C444A0(a1 + 16);
        return v2;
      }
    }
    else if ( (v5 & v4) == 0 )
    {
      LOBYTE(v2) = v4 == 0;
      return v2;
    }
    return 1;
  }
  return v2;
}
