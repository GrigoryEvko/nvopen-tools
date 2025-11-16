// Function: sub_143ACA0
// Address: 0x143aca0
//
_QWORD *__fastcall sub_143ACA0(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax

  *(_QWORD *)a1 = 0;
  result = (_QWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 8) = 1;
  do
  {
    if ( result )
      *result = -8;
    result += 2;
  }
  while ( (_QWORD *)(a1 + 528) != result );
  *(_QWORD *)(a1 + 544) = a2;
  *(_DWORD *)(a1 + 536) = 0;
  *(_QWORD *)(a1 + 528) = a2 + 40;
  return result;
}
