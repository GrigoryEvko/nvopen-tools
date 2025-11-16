// Function: sub_D23F30
// Address: 0xd23f30
//
_QWORD *__fastcall sub_D23F30(_QWORD *a1, __int64 a2)
{
  _QWORD *result; // rax
  _QWORD *v3; // rdi

  *a1 = a2;
  a1[1] = a1 + 3;
  a1[2] = 0x400000000LL;
  result = a1 + 9;
  v3 = a1 + 17;
  *(v3 - 10) = 0;
  *(v3 - 9) = 1;
  do
  {
    if ( result )
      *result = -4096;
    result += 2;
  }
  while ( v3 != result );
  return result;
}
