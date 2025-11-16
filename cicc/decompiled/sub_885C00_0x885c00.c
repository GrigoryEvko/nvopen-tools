// Function: sub_885C00
// Address: 0x885c00
//
_QWORD *__fastcall sub_885C00(__int16 a1, char *a2)
{
  size_t v2; // rax
  _QWORD *result; // rax

  v2 = strlen(a2);
  result = sub_885B80(a2, v2, 0, -1);
  *((_WORD *)result + 44) = a1;
  return result;
}
