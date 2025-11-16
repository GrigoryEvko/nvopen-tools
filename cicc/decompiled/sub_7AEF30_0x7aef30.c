// Function: sub_7AEF30
// Address: 0x7aef30
//
_QWORD *__fastcall sub_7AEF30(__int64 a1)
{
  _QWORD *v1; // rdx
  _QWORD *result; // rax
  _QWORD *v3; // rsi

  **(_QWORD **)a1 = qword_4F06430;
  qword_4F06430 = *(_QWORD **)a1;
  v1 = qword_4F06430;
  result = (_QWORD *)qword_4F06430[13];
  if ( result )
  {
    v3 = qword_4F064A0;
    while ( 1 )
    {
      v1[13] = *result;
      *result = v3;
      v3 = result;
      qword_4F064A0 = result;
      v1 = *(_QWORD **)a1;
      if ( !*(_QWORD *)(*(_QWORD *)a1 + 104LL) )
        break;
      result = *(_QWORD **)(*(_QWORD *)a1 + 104LL);
    }
  }
  *(_QWORD *)a1 = 0;
  return result;
}
