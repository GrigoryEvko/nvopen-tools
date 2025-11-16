// Function: sub_310F8B0
// Address: 0x310f8b0
//
_BOOL8 __fastcall sub_310F8B0(__int64 a1)
{
  _BOOL4 v1; // r8d
  __int64 v3; // rax
  char *v4; // rcx
  unsigned __int8 v5; // dl
  __int64 v6; // rdx

  if ( (*(_BYTE *)(a1 + 80) & 1) != 0 )
    return 0;
  if ( (*(_BYTE *)(a1 + 32) & 0xFu) - 7 > 1 )
    return 0;
  if ( sub_B2FC80(a1) )
    return 0;
  if ( (unsigned __int8)sub_B2F6B0(a1) )
    return 0;
  v1 = (*(_BYTE *)(a1 + 80) & 2) != 0;
  if ( (*(_BYTE *)(a1 + 80) & 2) != 0 )
  {
    return 0;
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 16);
    if ( v3 )
    {
      while ( 1 )
      {
        v4 = *(char **)(v3 + 24);
        v5 = *v4;
        if ( (unsigned __int8)*v4 <= 0x1Cu )
          break;
        if ( v5 == 62 )
        {
          v6 = *((_QWORD *)v4 - 8);
          if ( v6 && a1 == v6 || (v4[2] & 1) != 0 )
            return v1;
          if ( *(_QWORD *)(v6 + 8) != *(_QWORD *)(a1 + 24) )
            return 0;
        }
        else
        {
          if ( v5 != 61 || (v4[2] & 1) != 0 )
            return v1;
          if ( *((_QWORD *)v4 + 1) != *(_QWORD *)(a1 + 24) )
            return 0;
        }
        v3 = *(_QWORD *)(v3 + 8);
        if ( !v3 )
          return 1;
      }
    }
    else
    {
      return 1;
    }
  }
  return v1;
}
