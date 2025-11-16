// Function: sub_730990
// Address: 0x730990
//
_BOOL8 __fastcall sub_730990(__int64 a1)
{
  bool v1; // bl
  _BOOL4 v2; // r13d
  char v3; // al
  char v5; // al
  __int64 i; // r13

  if ( !dword_4D04964 || dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774) )
  {
    v2 = 1;
    v1 = 1;
    if ( *(_BYTE *)(a1 + 173) == 12 )
      return 1;
  }
  else
  {
    v1 = unk_4D047FC != 0;
    v2 = unk_4D047FC != 0;
    if ( *(_BYTE *)(a1 + 173) == 12 )
      return 1;
  }
  if ( !sub_712570(a1) )
  {
    v3 = *(_BYTE *)(a1 + 173);
    if ( v3 == 6 )
    {
      v5 = *(_BYTE *)(a1 + 176);
      if ( !v5 )
      {
        if ( !*(_QWORD *)(a1 + 184) )
          return v2;
        return 1;
      }
      if ( v5 == 1 )
      {
        if ( !*(_QWORD *)(a1 + 184) )
          return v2;
        v2 = sub_730840(a1);
        if ( !v2 )
        {
          if ( !*(_QWORD *)(a1 + 192) && HIDWORD(qword_4F077B4) && !(_DWORD)qword_4F077B4 )
            return qword_4F077A8 <= 0x76BFu;
          return v2;
        }
        return 1;
      }
    }
    else
    {
      if ( v3 == 7 )
      {
        if ( !*(_QWORD *)(a1 + 200) )
          return v2;
        return 1;
      }
      if ( v3 == 1 && v1 )
      {
        for ( i = *(_QWORD *)(a1 + 128); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        if ( (unsigned int)sub_8D2EF0(i) || (unsigned int)sub_8D3D10(i) )
          return sub_6210B0(a1, 0) == 0;
      }
    }
    return 0;
  }
  return v2;
}
