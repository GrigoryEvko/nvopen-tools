// Function: sub_10D1CB0
// Address: 0x10d1cb0
//
bool __fastcall sub_10D1CB0(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  bool result; // al
  _BYTE *v5; // r13
  _BYTE *v6; // r13
  char v7; // al
  __int64 v8; // rsi
  char v9; // al
  __int64 v10; // rsi

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = (_BYTE *)*((_QWORD *)a3 - 8);
  if ( *v5 == 59
    && ((v9 = sub_995B10(a1, *((_QWORD *)v5 - 8)), v10 = *((_QWORD *)v5 - 4), v9) && v10 == *a1[1]
     || (unsigned __int8)sub_995B10(a1, v10) && *((_QWORD *)v5 - 8) == *a1[1]) )
  {
    v6 = (_BYTE *)*((_QWORD *)a3 - 4);
    result = 1;
    if ( (_BYTE *)*a1[2] == v6 )
      return result;
  }
  else
  {
    v6 = (_BYTE *)*((_QWORD *)a3 - 4);
  }
  result = *v6 == 59
        && ((v7 = sub_995B10(a1, *((_QWORD *)v6 - 8)), v8 = *((_QWORD *)v6 - 4), v7) && v8 == *a1[1]
         || (unsigned __int8)sub_995B10(a1, v8) && *((_QWORD *)v6 - 8) == *a1[1])
        && *a1[2] == *((_QWORD *)a3 - 8);
  return result;
}
