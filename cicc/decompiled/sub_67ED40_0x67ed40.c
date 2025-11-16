// Function: sub_67ED40
// Address: 0x67ed40
//
_QWORD *sub_67ED40()
{
  __int64 *v0; // rax
  __int64 *v1; // rbx
  __int64 v2; // rax
  _QWORD *result; // rax
  _QWORD *v4; // rbx

  v0 = (__int64 *)sub_823970(24);
  qword_4CFFD80 = (__int64)v0;
  if ( v0 )
  {
    *v0 = 0;
    v1 = v0;
    v0[1] = 0;
    v0[2] = 0;
    v2 = sub_823970(6144);
    v1[1] = 256;
    *v1 = v2;
  }
  result = (_QWORD *)sub_823970(24);
  qword_4CFFD78 = (__int64)result;
  v4 = result;
  if ( result )
  {
    *result = 0;
    result[1] = 0;
    result[2] = 0;
    result = (_QWORD *)sub_823970(128);
    v4[1] = 16;
    *v4 = result;
  }
  return result;
}
