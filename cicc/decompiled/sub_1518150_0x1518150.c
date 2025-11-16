// Function: sub_1518150
// Address: 0x1518150
//
_BYTE *__fastcall sub_1518150(__int64 *a1, unsigned int a2)
{
  _BYTE *result; // rax

  result = (_BYTE *)sub_1517EB0(*a1, a2);
  if ( result )
  {
    if ( (unsigned __int8)(*result - 4) >= 0x1Fu )
      return 0;
  }
  return result;
}
