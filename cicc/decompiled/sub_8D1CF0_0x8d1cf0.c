// Function: sub_8D1CF0
// Address: 0x8d1cf0
//
__int64 __fastcall sub_8D1CF0(__int64 a1, _DWORD *a2)
{
  char v2; // al
  unsigned int v3; // r8d
  char v5; // al
  bool v6; // zf

  v2 = *(_BYTE *)(a1 + 140);
  if ( (unsigned __int8)(v2 - 9) <= 2u )
  {
    v3 = 0;
    if ( (*(_BYTE *)(a1 + 177) & 0x20) != 0 )
      return v3;
    goto LABEL_3;
  }
  v3 = 0;
  if ( v2 != 2 )
    return v3;
  if ( (*(_BYTE *)(a1 + 161) & 8) != 0 )
  {
LABEL_3:
    v3 = 0;
    if ( (*(_BYTE *)(a1 + 88) & 0x70) == 0 )
    {
      if ( !*(_QWORD *)(a1 + 8) )
      {
        if ( dword_4F60588 )
        {
          v5 = *(_BYTE *)(a1 + 89);
          if ( (v5 & 4) != 0 )
          {
            if ( (v5 & 1) != 0 )
              dword_4F6058C = 1;
            return v3;
          }
        }
      }
      *a2 = 1;
      if ( (*(_BYTE *)(a1 + 89) & 1) != 0 )
      {
        v6 = *(_QWORD *)(a1 + 8) == 0;
        v3 = 1;
        dword_4F6058C = 1;
        if ( !v6 )
          return v3;
      }
      else if ( *(_QWORD *)(a1 + 8) )
      {
        return 1;
      }
      dword_4F60590 = 1;
      return 1;
    }
    return v3;
  }
  return 0;
}
