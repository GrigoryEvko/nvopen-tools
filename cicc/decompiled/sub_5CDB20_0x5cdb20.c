// Function: sub_5CDB20
// Address: 0x5cdb20
//
__int64 __fastcall sub_5CDB20(__int64 a1, __int64 a2, char a3)
{
  __int64 result; // rax
  char v4; // dl

  result = a2;
  if ( a3 != 6 )
  {
    if ( *(_BYTE *)(a1 + 8) && a3 != 3 )
    {
      if ( a3 == 12 && unk_4F077C4 == 2 )
      {
        if ( unk_4D04964 )
        {
          sub_5CCAE0(unk_4F07471, a1);
          result = a2;
          if ( unk_4F07471 > 5u )
          {
            *(_BYTE *)(a1 + 8) = 0;
            return result;
          }
        }
      }
      goto LABEL_5;
    }
    return result;
  }
  v4 = *(_BYTE *)(a2 + 140);
  if ( (unsigned __int8)(v4 - 9) > 2u )
  {
    if ( v4 == 2 )
    {
      if ( (*(_BYTE *)(a2 + 161) & 8) != 0 )
        goto LABEL_12;
    }
    else if ( v4 == 12 && *(_QWORD *)(a2 + 8) )
    {
      goto LABEL_12;
    }
    sub_5CCAE0(8u, a1);
    *(_BYTE *)(a1 + 8) = 0;
    return a2;
  }
LABEL_12:
  if ( *(_BYTE *)(a1 + 8) )
LABEL_5:
    *(_BYTE *)(result + 91) |= 4u;
  return result;
}
