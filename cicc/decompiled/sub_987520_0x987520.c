// Function: sub_987520
// Address: 0x987520
//
_QWORD *__fastcall sub_987520(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int8 v5; // bl
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rsi

  v5 = a3;
  v6 = sub_C33340(a1, a2, a3, a4, a5);
  v7 = v6;
  if ( a2 == v6 )
  {
    sub_C3C500(a1, v6, 0);
    v8 = v5;
    if ( *a1 != v7 )
      goto LABEL_3;
  }
  else
  {
    sub_C373C0(a1, a2, 0);
    v8 = v5;
    if ( *a1 != v7 )
    {
LABEL_3:
      sub_C36EF0(a1, v8);
      return a1;
    }
  }
  sub_C3CF20(a1, v8);
  return a1;
}
