// Function: sub_80A630
// Address: 0x80a630
//
__int64 __fastcall sub_80A630(__int64 a1, char a2)
{
  char v2; // al
  char v4; // dl
  char v5; // al
  __int64 v6; // rcx
  _BOOL4 v7; // eax
  char v8; // al

  while ( 1 )
  {
    if ( a2 == 6 )
    {
      v4 = *(_BYTE *)(a1 + 140);
      if ( (unsigned __int8)(v4 - 9) > 2u )
      {
        v2 = *(_BYTE *)(a1 + 89);
        if ( v4 != 2 || (*(_BYTE *)(a1 + 161) & 8) == 0 )
          goto LABEL_6;
      }
      else if ( !*(_QWORD *)a1 || !*(_QWORD *)(*(_QWORD *)a1 + 96LL) || *(_QWORD *)(*(_QWORD *)(a1 + 168) + 256LL) )
      {
LABEL_18:
        v2 = *(_BYTE *)(a1 + 89);
        goto LABEL_6;
      }
      v5 = *(_BYTE *)(a1 + 89);
      if ( (v5 & 0x40) != 0 || ((v5 & 8) == 0 ? (v6 = *(_QWORD *)(a1 + 8)) : (v6 = *(_QWORD *)(a1 + 24)), !v6) )
      {
        if ( (v5 & 5) == 0 )
        {
          if ( v4 != 9 )
            return 1;
          v7 = sub_80A5F0(a1);
          if ( ((*(_BYTE *)(*(_QWORD *)(a1 + 168) + 109LL) & 0x20) == 0
             || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a1 + 96LL) + 181LL) & 8) == 0)
            && !v7 )
          {
            return 1;
          }
        }
      }
      goto LABEL_18;
    }
    v2 = *(_BYTE *)(a1 + 89);
    if ( a2 == 11 )
      break;
    if ( a2 == 7 )
    {
      if ( (v2 & 5) == 0 && *(_BYTE *)(a1 + 136) == 2 )
        return 1;
LABEL_6:
      if ( (v2 & 4) == 0 )
        return 0;
LABEL_7:
      a1 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
LABEL_8:
      if ( !unk_4D04440 )
        return 0;
      goto LABEL_9;
    }
    if ( (v2 & 4) == 0 )
      return 0;
    a1 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
    if ( a2 != 8 )
      goto LABEL_8;
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 168) + 113LL) || !unk_4D04440 )
      return 0;
LABEL_9:
    a2 = 6;
  }
  if ( (*(_BYTE *)(a1 + 198) & 0x20) != 0 )
  {
    v8 = v2 & 4;
  }
  else
  {
    if ( (v2 & 4) != 0 )
      goto LABEL_7;
    if ( *(_BYTE *)(a1 + 172) != 2 )
      return 0;
    if ( !sub_736A10(a1) )
      return 1;
    v8 = *(_BYTE *)(a1 + 89) & 4;
  }
  if ( v8 )
    goto LABEL_7;
  return 0;
}
