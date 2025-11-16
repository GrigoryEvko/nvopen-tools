// Function: sub_7293C0
// Address: 0x7293c0
//
_QWORD *__fastcall sub_7293C0(char a1, _QWORD *a2, _QWORD **a3)
{
  _BYTE *v4; // rax
  _QWORD *v5; // rcx
  _QWORD *result; // rax
  _QWORD **v7; // rdx

  v4 = sub_727450();
  v4[16] = a1;
  v5 = v4;
  *((_QWORD *)v4 + 1) = *a2;
  result = *a3;
  if ( *a3 )
  {
    do
    {
      v7 = (_QWORD **)result;
      result = (_QWORD *)*result;
    }
    while ( result );
    a3 = v7;
  }
  *a3 = v5;
  return result;
}
