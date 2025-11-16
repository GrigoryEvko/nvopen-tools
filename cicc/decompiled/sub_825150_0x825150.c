// Function: sub_825150
// Address: 0x825150
//
_QWORD *sub_825150()
{
  _QWORD *result; // rax
  _QWORD *v1; // rbx
  _QWORD *v2; // rdi

  result = &qword_4F07280;
  v1 = (_QWORD *)qword_4F07320[0];
  if ( qword_4F07320[0] )
  {
    do
    {
      result = (_QWORD *)v1[4];
      v2 = (_QWORD *)result[4];
      if ( v2 )
        result = (_QWORD *)sub_824ED0(v2);
      v1 = (_QWORD *)*v1;
    }
    while ( v1 );
  }
  return result;
}
