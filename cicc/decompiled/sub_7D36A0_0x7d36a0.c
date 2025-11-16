// Function: sub_7D36A0
// Address: 0x7d36a0
//
_QWORD *__fastcall sub_7D36A0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r12
  _QWORD *v4; // rax

  for ( ; *(_BYTE *)(a2 + 140) == 12; a2 = *(_QWORD *)(a2 + 160) )
    ;
  v2 = *(_QWORD **)(a1 + 24);
  if ( (*(_DWORD *)(a2 + 160) & 0x901000) == 0x101000 && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)a2 + 96LL) + 48LL) & 1) == 0 )
  {
    sub_8947D0(a2);
    if ( (*(_BYTE *)(a1 + 17) & 0x20) != 0 )
      return 0;
LABEL_5:
    if ( v2 )
      return v2;
    v4 = *(_QWORD **)(a2 + 168);
    if ( (*(_BYTE *)(a2 + 161) & 0x10) != 0 )
    {
      if ( !v4 )
      {
LABEL_12:
        *(_QWORD *)(a1 + 24) = v2;
        return v2;
      }
      v4 = (_QWORD *)v4[12];
    }
    if ( v4 )
    {
      while ( 1 )
      {
        v2 = (_QWORD *)*v4;
        if ( *(_QWORD *)*v4 == *(_QWORD *)a1 )
          break;
        v4 = (_QWORD *)v4[15];
        if ( !v4 )
          goto LABEL_18;
      }
    }
    else
    {
LABEL_18:
      v2 = 0;
    }
    goto LABEL_12;
  }
  if ( (*(_BYTE *)(a1 + 17) & 0x20) == 0 )
    goto LABEL_5;
  return 0;
}
