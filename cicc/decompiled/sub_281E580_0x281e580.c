// Function: sub_281E580
// Address: 0x281e580
//
__int64 __fastcall sub_281E580(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // r13
  __int64 v6; // rdi
  unsigned int v7; // r14d

  if ( a1 )
  {
    v3 = 0;
    if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) != 3 )
      return v3;
    v4 = *(_QWORD *)(a1 - 96);
    if ( *(_BYTE *)v4 != 82 )
      return v3;
    v6 = *(_QWORD *)(v4 - 32);
    if ( *(_BYTE *)v6 != 17 )
      return v3;
    v7 = *(_DWORD *)(v6 + 32);
    if ( v7 <= 0x40 )
    {
      if ( *(_QWORD *)(v6 + 24) )
        return v3;
    }
    else if ( v7 != (unsigned int)sub_C444A0(v6 + 24) )
    {
      return v3;
    }
    if ( (*(_WORD *)(v4 + 2) & 0x3F) == 0x21 )
    {
      if ( a2 == *(_QWORD *)(a1 - 32) )
        return *(_QWORD *)(v4 - 64);
    }
    else if ( a2 == *(_QWORD *)(a1 - 64) && (*(_WORD *)(v4 + 2) & 0x3F) == 0x20 )
    {
      return *(_QWORD *)(v4 - 64);
    }
  }
  return 0;
}
