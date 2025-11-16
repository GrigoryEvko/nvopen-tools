// Function: sub_161CC10
// Address: 0x161cc10
//
_QWORD *__fastcall sub_161CC10(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        unsigned __int64 a9)
{
  __int64 v10; // rdx
  _QWORD *v11; // rcx
  _QWORD *result; // rax

  v10 = (a2 - 1) / 2;
  if ( a2 > a3 )
  {
    while ( 1 )
    {
      v11 = (_QWORD *)(a1 + 24 * v10);
      result = (_QWORD *)(a1 + 24 * a2);
      if ( a9 <= v11[2] )
        break;
      *result = *v11;
      result[1] = v11[1];
      result[2] = v11[2];
      a2 = v10;
      if ( a3 >= v10 )
      {
        result = (_QWORD *)(a1 + 24 * v10);
        break;
      }
      v10 = (v10 - 1) / 2;
    }
  }
  else
  {
    result = (_QWORD *)(a1 + 24 * a2);
  }
  result[2] = a9;
  *result = a7;
  result[1] = a8;
  return result;
}
