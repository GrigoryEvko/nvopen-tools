// Function: sub_12905B0
// Address: 0x12905b0
//
__int64 __fastcall sub_12905B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 result; // rax

  if ( a1 )
  {
    v2 = 0;
    if ( *(_BYTE *)(a1 + 40) == 11 )
    {
      v2 = *(_QWORD *)(a1 + 72);
      if ( !a2 )
        goto LABEL_5;
    }
    else if ( !a2 )
    {
LABEL_5:
      v3 = 0;
      goto LABEL_6;
    }
    if ( *(_BYTE *)(a2 + 40) != 11 )
      goto LABEL_5;
  }
  else
  {
    if ( !a2 )
      return 0;
    v2 = 0;
    v3 = 0;
    if ( *(_BYTE *)(a2 + 40) != 11 )
      goto LABEL_10;
  }
  v3 = *(_QWORD *)(a2 + 72);
  if ( a1 )
  {
LABEL_6:
    result = 1;
    if ( (*(_BYTE *)(a1 + 41) & 0x10) != 0 )
      return result;
    if ( !v2 )
      goto LABEL_9;
    goto LABEL_8;
  }
  if ( v2 )
  {
LABEL_8:
    result = 1;
    if ( (*(_BYTE *)(v2 + 41) & 0x10) != 0 )
      return result;
LABEL_9:
    if ( !a2 )
      goto LABEL_11;
  }
LABEL_10:
  result = 1;
  if ( (*(_BYTE *)(a2 + 41) & 0x20) != 0 )
    return result;
LABEL_11:
  if ( !v3 || (result = 1, (*(_BYTE *)(v3 + 41) & 0x20) == 0) )
  {
    if ( !a1 || (result = 2, (*(_BYTE *)(a1 + 41) & 0x20) == 0) )
    {
      if ( !v2 || (result = 2, (*(_BYTE *)(v2 + 41) & 0x20) == 0) )
      {
        if ( !a2 || (result = 2, (*(_BYTE *)(a2 + 41) & 0x10) == 0) )
        {
          if ( v3 )
          {
            result = *(_BYTE *)(v3 + 41) & 0x10;
            if ( (*(_BYTE *)(v3 + 41) & 0x10) != 0 )
              return 2;
            return result;
          }
          return 0;
        }
      }
    }
  }
  return result;
}
