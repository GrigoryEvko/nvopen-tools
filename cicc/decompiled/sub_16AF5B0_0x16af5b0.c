// Function: sub_16AF5B0
// Address: 0x16af5b0
//
_QWORD *__fastcall sub_16AF5B0(_QWORD *a1, unsigned __int64 a2)
{
  _QWORD *result; // rax
  unsigned __int64 v3; // rdx

  result = a1;
  v3 = *a1 - a2;
  if ( *a1 <= a2 )
    v3 = 0;
  *a1 = v3;
  return result;
}
