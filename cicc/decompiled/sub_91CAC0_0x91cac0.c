// Function: sub_91CAC0
// Address: 0x91cac0
//
_DWORD *__fastcall sub_91CAC0(_QWORD *a1)
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
