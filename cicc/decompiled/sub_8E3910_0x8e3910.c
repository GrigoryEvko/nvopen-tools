// Function: sub_8E3910
// Address: 0x8e3910
//
_QWORD *sub_8E3910()
{
  _QWORD *result; // rax
  _QWORD *v1; // rbx
  _QWORD *v2; // rcx
  _QWORD *v3; // rdx

  result = (_QWORD *)sub_823970(16);
  qword_4F60598 = (__int64)result;
  if ( result )
  {
    v1 = result;
    result = (_QWORD *)sub_823970(0x8000);
    v2 = result;
    v3 = result + 4096;
    do
    {
      if ( result )
      {
        *result = 0;
        result[1] = 0;
        result[2] = 0;
      }
      result += 4;
    }
    while ( result != v3 );
    *v1 = v2;
    v1[1] = 1023;
  }
  return result;
}
