// Function: sub_1C2E400
// Address: 0x1c2e400
//
_BYTE *__fastcall sub_1C2E400(_QWORD *a1)
{
  _BYTE *result; // rax

  result = (_BYTE *)*a1;
  if ( (unsigned __int8)(*(_BYTE *)*a1 - 4) >= 0x1Fu )
    return 0;
  return result;
}
