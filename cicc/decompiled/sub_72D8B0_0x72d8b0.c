// Function: sub_72D8B0
// Address: 0x72d8b0
//
_QWORD *__fastcall sub_72D8B0(__int64 a1)
{
  _QWORD *result; // rax

  result = (_QWORD *)qword_4F04C50;
  if ( qword_4F04C50 )
  {
    result = *(_QWORD **)(qword_4F04C50 + 208LL);
    if ( result )
    {
      while ( result[1] != a1 )
      {
        result = (_QWORD *)*result;
        if ( !result )
          return result;
      }
      if ( result[3] )
        return (_QWORD *)result[3];
    }
  }
  return result;
}
