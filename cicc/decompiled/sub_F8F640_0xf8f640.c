// Function: sub_F8F640
// Address: 0xf8f640
//
__int64 __fastcall sub_F8F640(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned int v4; // eax

  while ( a1 != a2 )
  {
    if ( !a1 )
      BUG();
    if ( *(_BYTE *)(a1 - 24) != 85 )
      return 0;
    v3 = *(_QWORD *)(a1 - 56);
    if ( !v3 || *(_BYTE *)v3 || *(_QWORD *)(v3 + 24) != *(_QWORD *)(a1 + 56) || (*(_BYTE *)(v3 + 33) & 0x20) == 0 )
      return 0;
    v4 = *(_DWORD *)(v3 + 36);
    if ( v4 > 0x47 )
    {
      if ( v4 != 210 )
        return 0;
    }
    else if ( v4 <= 0x44 )
    {
      return 0;
    }
    a1 = *(_QWORD *)(a1 + 8);
  }
  return 1;
}
