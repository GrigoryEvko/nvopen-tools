// Function: sub_1A54170
// Address: 0x1a54170
//
_QWORD *__fastcall sub_1A54170(unsigned int *a1, _QWORD *a2)
{
  unsigned int v2; // edx
  _QWORD *result; // rax
  __int64 v4; // rdx

  v2 = a1[2];
  if ( a1[3] <= v2 )
  {
    sub_1A53FD0(a1, 0);
    v2 = a1[2];
  }
  result = (_QWORD *)(*(_QWORD *)a1 + 16LL * v2);
  if ( result )
  {
    *result = *a2;
    v4 = a2[1];
    a2[1] = 0;
    result[1] = v4;
    v2 = a1[2];
  }
  a1[2] = v2 + 1;
  return result;
}
