// Function: sub_5CB9F0
// Address: 0x5cb9f0
//
_QWORD **__fastcall sub_5CB9F0(_QWORD **a1)
{
  _QWORD **result; // rax
  _QWORD *v2; // rax
  _QWORD *v3; // rdx

  result = 0;
  if ( a1 )
  {
    v2 = *a1;
    if ( *a1 )
    {
      do
      {
        v3 = v2;
        v2 = (_QWORD *)*v2;
      }
      while ( v2 );
      return (_QWORD **)v3;
    }
    else
    {
      return a1;
    }
  }
  return result;
}
