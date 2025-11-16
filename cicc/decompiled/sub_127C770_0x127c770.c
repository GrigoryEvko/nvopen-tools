// Function: sub_127C770
// Address: 0x127c770
//
_DWORD *__fastcall sub_127C770(_QWORD *a1)
{
  _DWORD *result; // rax

  result = (_DWORD *)*(unsigned int *)a1;
  if ( (_DWORD)result )
  {
    *(_QWORD *)dword_4F07508 = *a1;
    return dword_4F07508;
  }
  return result;
}
