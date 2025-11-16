// Function: sub_72B620
// Address: 0x72b620
//
_QWORD *__fastcall sub_72B620(__int64 a1, char a2)
{
  _QWORD *result; // rax

  result = sub_7259C0(16);
  result[16] = 0;
  *((_DWORD *)result + 34) = 0;
  result[20] = a1;
  *((_BYTE *)result + 168) = a2;
  return result;
}
