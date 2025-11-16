// Function: sub_1DCCCA0
// Address: 0x1dccca0
//
_QWORD *__fastcall sub_1DCCCA0(char *a1, int a2, __int64 a3, __int64 a4)
{
  char *v6; // rax
  _QWORD *v7; // rdx
  _QWORD *result; // rax

  v6 = sub_1DCC790(a1, a2);
  v7 = (_QWORD *)*((_QWORD *)v6 + 5);
  for ( result = (_QWORD *)*((_QWORD *)v6 + 4); result != v7; ++result )
  {
    while ( a3 != *result )
    {
      if ( ++result == v7 )
        return result;
    }
    *result = a4;
  }
  return result;
}
