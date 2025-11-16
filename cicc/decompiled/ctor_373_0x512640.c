// Function: ctor_373
// Address: 0x512640
//
_QWORD *__fastcall ctor_373(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *result; // rax

  stru_4FD6AA0.__list.__next = 0;
  qword_4FD6AD8 = 72704;
  *(_OWORD *)&stru_4FD6AA0.__lock = 0;
  *((_OWORD *)&stru_4FD6AA0.__align + 1) = 0;
  result = (_QWORD *)malloc(72704, a2, a3, a4, a5, a6);
  qword_4FD6AD0 = (__int64)result;
  if ( result )
  {
    qword_4FD6AC8 = (__int64)result;
    *result = 72704;
    result[1] = 0;
  }
  else
  {
    qword_4FD6AD8 = 0;
    qword_4FD6AC8 = 0;
  }
  return result;
}
