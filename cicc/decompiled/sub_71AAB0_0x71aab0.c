// Function: sub_71AAB0
// Address: 0x71aab0
//
_QWORD *__fastcall sub_71AAB0(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax

  result = (_QWORD *)sub_726700(5);
  result[7] = a2;
  *result = *(_QWORD *)(a1 + 128);
  *(_QWORD *)(a1 + 144) = result;
  return result;
}
