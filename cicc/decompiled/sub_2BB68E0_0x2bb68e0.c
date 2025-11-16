// Function: sub_2BB68E0
// Address: 0x2bb68e0
//
_QWORD *__fastcall sub_2BB68E0(
        _QWORD *a1,
        __int64 a2,
        _QWORD *a3,
        unsigned __int8 (__fastcall *a4)(__int64, _QWORD, _QWORD),
        __int64 a5)
{
  __int64 v5; // rsi
  _QWORD *v6; // r12
  __int64 v7; // rbx
  __int64 v9; // r13
  _QWORD *v10; // r14

  v5 = a2 - (_QWORD)a1;
  v6 = a1;
  v7 = v5 >> 3;
  if ( v5 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v9 = v7 >> 1;
        v10 = &v6[v7 >> 1];
        if ( !a4(a5, *v10, *a3) )
          break;
        v6 = v10 + 1;
        v7 = v7 - v9 - 1;
        if ( v7 <= 0 )
          return v6;
      }
      v7 >>= 1;
    }
    while ( v9 > 0 );
  }
  return v6;
}
