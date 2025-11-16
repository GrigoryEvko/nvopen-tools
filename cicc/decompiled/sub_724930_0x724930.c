// Function: sub_724930
// Address: 0x724930
//
_QWORD *__fastcall sub_724930(__int64 a1, int a2)
{
  _QWORD *result; // rax

  result = sub_7247C0(56);
  result[1] = a1;
  *((_DWORD *)result + 4) = a2;
  result[3] = 0;
  result[4] = 0;
  result[5] = 0;
  result[6] = 0;
  *result = 0;
  return result;
}
