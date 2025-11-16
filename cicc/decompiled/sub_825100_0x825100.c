// Function: sub_825100
// Address: 0x825100
//
_QWORD *sub_825100()
{
  _QWORD *result; // rax
  _QWORD *v1; // rbx
  _BYTE *v2; // rdi

  result = &qword_4F07280;
  v1 = (_QWORD *)qword_4F07320[0];
  if ( qword_4F07320[0] )
  {
    do
    {
      result = (_QWORD *)v1[4];
      v2 = (_BYTE *)result[4];
      if ( v2 )
        result = (_QWORD *)sub_824EB0(v2);
      v1 = (_QWORD *)*v1;
    }
    while ( v1 );
  }
  return result;
}
