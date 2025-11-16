// Function: sub_80D400
// Address: 0x80d400
//
_QWORD *sub_80D400()
{
  _QWORD *result; // rax
  _QWORD *v1; // rbx
  _QWORD *v2; // rcx
  _QWORD *v3; // rdx

  qword_4F18BC0 = 0;
  qword_4F18BB8 = 0;
  result = (_QWORD *)sub_823970(16);
  qword_4F18BD0 = (__int64)result;
  if ( result )
  {
    v1 = result;
    result = (_QWORD *)sub_823970(4096);
    v2 = result;
    v3 = result + 512;
    do
    {
      if ( result )
        *result = 0;
      result += 2;
    }
    while ( result != v3 );
    *v1 = v2;
    v1[1] = 255;
  }
  return result;
}
