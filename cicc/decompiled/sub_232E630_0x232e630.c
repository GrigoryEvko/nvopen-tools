// Function: sub_232E630
// Address: 0x232e630
//
_BYTE *__fastcall sub_232E630(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _BYTE *v6; // rax
  _BYTE *result; // rax

  sub_904010(a2, "function");
  if ( *(_BYTE *)(a1 + 16) || *(_BYTE *)(a1 + 17) )
  {
    sub_904010(a2, "<");
    if ( *(_BYTE *)(a1 + 16) )
    {
      sub_904010(a2, "eager-inv");
      if ( *(_BYTE *)(a1 + 16) )
      {
        if ( !*(_BYTE *)(a1 + 17) )
          goto LABEL_10;
        sub_904010(a2, ";");
      }
    }
    if ( *(_BYTE *)(a1 + 17) )
      sub_904010(a2, "no-rerun");
LABEL_10:
    sub_904010(a2, ">");
    v6 = *(_BYTE **)(a2 + 32);
    if ( (unsigned __int64)v6 < *(_QWORD *)(a2 + 24) )
      goto LABEL_4;
    goto LABEL_11;
  }
  v6 = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)v6 < *(_QWORD *)(a2 + 24) )
  {
LABEL_4:
    *(_QWORD *)(a2 + 32) = v6 + 1;
    *v6 = 40;
    goto LABEL_5;
  }
LABEL_11:
  sub_CB5D20(a2, 40);
LABEL_5:
  (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a1 + 8) + 24LL))(
    *(_QWORD *)(a1 + 8),
    a2,
    a3,
    a4);
  result = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 24) )
    return (_BYTE *)sub_CB5D20(a2, 41);
  *(_QWORD *)(a2 + 32) = result + 1;
  *result = 41;
  return result;
}
