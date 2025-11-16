// Function: sub_727F30
// Address: 0x727f30
//
__int64 __fastcall sub_727F30(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rsi
  __int64 result; // rax
  _QWORD *v6; // r12
  __int64 v7; // rax
  __int64 v8; // rax

  if ( *(_BYTE *)(a1 + 136) <= 2u )
  {
    if ( (*(_BYTE *)(a2 + 28) == 6) != ((*(_BYTE *)(a1 + 89) & 4) != 0) || (v8 = *(_QWORD *)(a2 + 112)) == 0 )
    {
LABEL_3:
      if ( (*(_BYTE *)(a1 + 89) & 4) == 0 )
        goto LABEL_10;
      goto LABEL_4;
    }
    while ( a1 != v8 )
    {
      v8 = *(_QWORD *)(v8 + 112);
      if ( !v8 )
        goto LABEL_3;
    }
    return a2;
  }
  v7 = *(_QWORD *)(a2 + 120);
  if ( v7 )
  {
    while ( a1 != v7 )
    {
      v7 = *(_QWORD *)(v7 + 112);
      if ( !v7 )
        goto LABEL_15;
    }
    return a2;
  }
LABEL_15:
  if ( (*(_BYTE *)(a1 + 89) & 4) == 0 )
  {
LABEL_10:
    v6 = *(_QWORD **)(a2 + 160);
    if ( !v6 )
      return 0;
    do
    {
      result = sub_727F30(a1);
      if ( result )
        break;
      v6 = (_QWORD *)*v6;
    }
    while ( v6 );
    return result;
  }
LABEL_4:
  v3 = *(_QWORD *)(a2 + 104);
  if ( !v3 )
    goto LABEL_10;
  while ( 1 )
  {
    if ( (unsigned __int8)(*(_BYTE *)(v3 + 140) - 9) <= 2u )
    {
      v4 = *(_QWORD *)(*(_QWORD *)(v3 + 168) + 152LL);
      if ( v4 )
      {
        if ( (*(_BYTE *)(v4 + 29) & 0x20) == 0 )
        {
          result = sub_727F30(a1);
          if ( result )
            return result;
        }
      }
    }
    v3 = *(_QWORD *)(v3 + 112);
    if ( !v3 )
      goto LABEL_10;
  }
}
