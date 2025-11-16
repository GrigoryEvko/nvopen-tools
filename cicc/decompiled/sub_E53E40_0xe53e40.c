// Function: sub_E53E40
// Address: 0xe53e40
//
_BYTE *__fastcall sub_E53E40(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4)
{
  _BYTE *result; // rax

  if ( a3 == 11 )
  {
    sub_904010(*(_QWORD *)(a1 + 304), "\t.extern\t");
  }
  else if ( a3 > 0xB )
  {
    if ( a3 != 24 )
      goto LABEL_22;
    sub_904010(*(_QWORD *)(a1 + 304), *(const char **)(*(_QWORD *)(a1 + 312) + 296LL));
  }
  else
  {
    if ( a3 != 9 )
    {
      if ( a3 == 10 )
      {
        sub_904010(*(_QWORD *)(a1 + 304), "\t.lglobl\t");
        goto LABEL_6;
      }
LABEL_22:
      sub_C64ED0("unhandled linkage type", 1u);
    }
    sub_904010(*(_QWORD *)(a1 + 304), *(const char **)(*(_QWORD *)(a1 + 312) + 272LL));
  }
LABEL_6:
  sub_EA12C0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  if ( a4 == 13 )
  {
    sub_904010(*(_QWORD *)(a1 + 304), ",exported");
LABEL_11:
    result = sub_E4D880(a1);
    if ( !*(_BYTE *)(a2 + 72) )
      return result;
    return sub_E4E8B0(a1, a2, *(char **)(a2 + 56), *(_QWORD *)(a2 + 64));
  }
  if ( a4 <= 0xD )
  {
    if ( !a4 )
      goto LABEL_11;
    if ( a4 == 12 )
    {
      sub_904010(*(_QWORD *)(a1 + 304), ",hidden");
      goto LABEL_11;
    }
LABEL_21:
    sub_C64ED0("unexpected value for Visibility type", 1u);
  }
  if ( a4 != 22 )
    goto LABEL_21;
  sub_904010(*(_QWORD *)(a1 + 304), ",protected");
  result = sub_E4D880(a1);
  if ( *(_BYTE *)(a2 + 72) )
    return sub_E4E8B0(a1, a2, *(char **)(a2 + 56), *(_QWORD *)(a2 + 64));
  return result;
}
