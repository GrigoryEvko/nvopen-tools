// Function: sub_737590
// Address: 0x737590
//
_QWORD *__fastcall sub_737590(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v5; // r12
  __int64 i; // rdi
  _QWORD *v8; // rax
  __int64 j; // rsi

  v5 = a1;
  if ( *((_BYTE *)a1 + 24) == 1 )
  {
    if ( (unsigned __int8)(*((_BYTE *)a1 + 56) - 8) > 1u )
      goto LABEL_8;
    for ( i = *a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v8 = (_QWORD *)v5[9];
    for ( j = *v8; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    if ( j != i )
    {
      if ( !(unsigned int)sub_8D97D0(i, j, 0, a4, a5) )
        goto LABEL_7;
      v8 = (_QWORD *)v5[9];
    }
    v5 = v8;
LABEL_7:
    while ( *((_BYTE *)v5 + 24) == 1 )
    {
LABEL_8:
      if ( (*((_BYTE *)v5 + 27) & 2) == 0 )
        return v5;
      if ( *((_BYTE *)v5 + 56) != 14 )
        return v5;
      v5 = (_QWORD *)v5[9];
    }
  }
  return v5;
}
