// Function: sub_6DF740
// Address: 0x6df740
//
_QWORD *__fastcall sub_6DF740(__int64 a1, _DWORD *a2)
{
  char v2; // al
  _QWORD *result; // rax
  __int64 v4; // r9
  __int64 *v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rax
  char v8; // al

  *a2 = 0;
  if ( (*(_BYTE *)(a1 + 25) & 1) != 0 )
    return 0;
  v2 = *(_BYTE *)(a1 + 24);
  if ( v2 == 2 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 56) + 173LL) == 12 && (unsigned int)sub_6DF6A0(*(_QWORD *)(a1 + 56), a2) )
    {
      v5 = (__int64 *)sub_73A720(v4);
      v6 = *v5;
      v5[10] = *(_QWORD *)(a1 + 80);
      v5[8] = *(_QWORD *)(a1 + 64);
      v7 = sub_73DC30(116, v6, v5);
      sub_730620(a1, v7);
      return (_QWORD *)a1;
    }
    return 0;
  }
  if ( v2 != 1 )
    return 0;
  v8 = *(_BYTE *)(a1 + 56);
  if ( (unsigned __int8)(v8 - 100) <= 1u )
  {
    result = (_QWORD *)sub_6DF740(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL));
    if ( !result )
      return result;
    goto LABEL_13;
  }
  if ( v8 != 25 )
    return 0;
  result = (_QWORD *)sub_6DF740(*(_QWORD *)(a1 + 72));
  if ( result )
  {
LABEL_13:
    *(_BYTE *)(a1 + 25) |= 1u;
    *(_QWORD *)a1 = *result;
  }
  return result;
}
