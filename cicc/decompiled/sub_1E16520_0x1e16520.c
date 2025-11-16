// Function: sub_1E16520
// Address: 0x1e16520
//
_QWORD *__fastcall sub_1E16520(__int64 a1)
{
  _QWORD *result; // rax
  _BYTE *v2; // rdx

  result = (_QWORD *)sub_1E16510(a1);
  if ( **(_WORD **)(a1 + 16) == 12 )
  {
    v2 = *(_BYTE **)(a1 + 32);
    if ( !*v2 && v2[40] == 1 )
      return (_QWORD *)sub_15C48E0(result, 1, 0, 0, 0);
  }
  return result;
}
