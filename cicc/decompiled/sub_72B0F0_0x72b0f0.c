// Function: sub_72B0F0
// Address: 0x72b0f0
//
__int64 __fastcall sub_72B0F0(__int64 a1, _QWORD *a2)
{
  char v2; // al
  __int64 result; // rax
  char v4; // al
  __int64 v5; // rax

  v2 = *(_BYTE *)(a1 + 24);
  if ( v2 == 1 )
  {
    v4 = *(_BYTE *)(a1 + 56);
    if ( (unsigned __int8)(v4 - 100) <= 1u )
    {
      a1 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL);
      v2 = *(_BYTE *)(a1 + 24);
      if ( v2 == 2 )
        goto LABEL_11;
      if ( v2 != 1 )
      {
LABEL_3:
        if ( v2 == 20 )
          goto LABEL_4;
        goto LABEL_8;
      }
      v4 = *(_BYTE *)(a1 + 56);
    }
    if ( !v4 )
    {
      a1 = *(_QWORD *)(a1 + 72);
      if ( *(_BYTE *)(a1 + 24) == 20 )
      {
LABEL_4:
        result = *(_QWORD *)(a1 + 56);
        if ( !a2 )
          return result;
LABEL_10:
        *a2 = a1;
        return result;
      }
    }
    goto LABEL_8;
  }
  if ( v2 != 2 )
    goto LABEL_3;
LABEL_11:
  if ( (*(_BYTE *)(a1 + 25) & 1) != 0
    || (v5 = *(_QWORD *)(a1 + 56), *(_BYTE *)(v5 + 173) != 6)
    || *(_BYTE *)(v5 + 176)
    || *(_QWORD *)(v5 + 192)
    || (*(_BYTE *)(v5 + 168) & 8) != 0 )
  {
LABEL_8:
    a1 = 0;
    result = 0;
    goto LABEL_9;
  }
  result = *(_QWORD *)(v5 + 184);
  a1 = 0;
LABEL_9:
  if ( a2 )
    goto LABEL_10;
  return result;
}
