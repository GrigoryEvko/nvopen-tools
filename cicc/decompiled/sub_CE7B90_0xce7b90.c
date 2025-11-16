// Function: sub_CE7B90
// Address: 0xce7b90
//
_BYTE *__fastcall sub_CE7B90(_QWORD *a1)
{
  _BYTE *result; // rax

  result = (_BYTE *)*a1;
  if ( (unsigned __int8)(*(_BYTE *)*a1 - 5) >= 0x20u )
    return 0;
  return result;
}
