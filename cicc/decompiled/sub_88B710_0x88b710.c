// Function: sub_88B710
// Address: 0x88b710
//
_QWORD *sub_88B710()
{
  _QWORD *result; // rax
  _QWORD *v1; // rbx
  _QWORD *v2; // rcx
  _QWORD *v3; // rdx

  qword_4D03FB8 = (void *)sub_823970(8LL * unk_4A598C0);
  memset(qword_4D03FB8, 0, 8LL * unk_4A598C0);
  result = (_QWORD *)sub_823970(16);
  qword_4F600F8 = (__int64)result;
  if ( result )
  {
    v1 = result;
    result = (_QWORD *)sub_823970(0x4000);
    v2 = result;
    v3 = result + 2048;
    do
    {
      if ( result )
        *result = 0;
      result += 2;
    }
    while ( v3 != result );
    *v1 = v2;
    v1[1] = 1023;
  }
  return result;
}
