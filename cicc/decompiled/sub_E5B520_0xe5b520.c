// Function: sub_E5B520
// Address: 0xe5b520
//
_QWORD *__fastcall sub_E5B520(_QWORD *a1)
{
  _QWORD *result; // rax
  __int64 v2; // r13

  if ( !*(_BYTE *)(a1[1] + 1793LL) )
  {
    if ( !*(_BYTE *)(a1[39] + 21LL) )
      goto LABEL_3;
    return (_QWORD *)sub_E77660(
                       a1,
                       *(unsigned __int16 *)(a1[41] + 72LL)
                     | ((unsigned __int64)*(unsigned __int8 *)(a1[41] + 74LL) << 16));
  }
  sub_E7A190();
  if ( *(_BYTE *)(a1[39] + 21LL) )
    return (_QWORD *)sub_E77660(
                       a1,
                       *(unsigned __int16 *)(a1[41] + 72LL)
                     | ((unsigned __int64)*(unsigned __int8 *)(a1[41] + 74LL) << 16));
LABEL_3:
  result = (_QWORD *)a1[1];
  if ( result[221] )
  {
    v2 = *(_QWORD *)(result[219] + 40LL);
    if ( v2 )
    {
      sub_E5B460((__int64)a1, *(void (__fastcall ****)(_QWORD, _QWORD, __int64, __int64, _QWORD))(result[21] + 96LL), 0);
      sub_E98820(a1, v2, 0);
      sub_EA12C0(v2, a1[38], a1[39]);
      sub_904010(a1[38], *(const char **)(a1[39] + 72LL));
      return sub_E4D880((__int64)a1);
    }
  }
  return result;
}
