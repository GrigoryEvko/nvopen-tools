// Function: sub_1671900
// Address: 0x1671900
//
_BYTE *__fastcall sub_1671900(__int64 *a1, char *a2)
{
  _BYTE *result; // rax

  sub_1671660(*a1, a2);
  result = (_BYTE *)a1[1];
  *result = 1;
  return result;
}
