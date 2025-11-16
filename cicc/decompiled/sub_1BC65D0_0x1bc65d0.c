// Function: sub_1BC65D0
// Address: 0x1bc65d0
//
_QWORD *__fastcall sub_1BC65D0(_QWORD *a1, __int64 a2, _QWORD *a3, unsigned __int8 (__fastcall *a4)(_QWORD, _QWORD))
{
  __int64 v4; // rsi
  _QWORD *v5; // r12
  __int64 v6; // rbx
  __int64 v8; // r15
  _QWORD *v9; // r13

  v4 = a2 - (_QWORD)a1;
  v5 = a1;
  v6 = v4 >> 3;
  if ( v4 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v8 = v6 >> 1;
        v9 = &v5[v6 >> 1];
        if ( a4(*a3, *v9) )
          break;
        v5 = v9 + 1;
        v6 = v6 - v8 - 1;
        if ( v6 <= 0 )
          return v5;
      }
      v6 >>= 1;
    }
    while ( v8 > 0 );
  }
  return v5;
}
