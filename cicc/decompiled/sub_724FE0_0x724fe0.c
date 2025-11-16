// Function: sub_724FE0
// Address: 0x724fe0
//
_QWORD *sub_724FE0()
{
  _QWORD *result; // rax

  result = sub_7247C0(32);
  *((_WORD *)result + 12) &= 0xFCu;
  *result = 0;
  result[1] = 0;
  result[2] = 0;
  return result;
}
