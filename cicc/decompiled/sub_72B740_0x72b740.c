// Function: sub_72B740
// Address: 0x72b740
//
_QWORD *__fastcall sub_72B740(__int64 a1, int a2)
{
  _QWORD *result; // rax

  result = sub_7259C0(12);
  result[20] = a1;
  *((_BYTE *)result + 184) = (a2 == 0) + 2;
  return result;
}
