// Function: sub_22316B0
// Address: 0x22316b0
//
_BYTE *__fastcall sub_22316B0(__int64 a1, char *a2, __int64 a3, char a4, __int64 a5, _BYTE *a6, __int64 a7, int *a8)
{
  _BYTE *result; // rax

  result = (_BYTE *)(sub_2231480(a6, a4, a2, a3, a7, a7 + *a8) - a6);
  *a8 = (int)result;
  return result;
}
