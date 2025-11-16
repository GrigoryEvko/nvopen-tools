// Function: sub_8DD360
// Address: 0x8dd360
//
_QWORD *__fastcall sub_8DD360(__int64 a1)
{
  _QWORD *result; // rax

  result = &qword_4F07280;
  if ( unk_4F072F4 )
  {
    result = &qword_4F04C50;
    if ( qword_4F04C50 )
      return (_QWORD *)sub_8D9600(
                         a1,
                         (__int64 (__fastcall *)(__int64, unsigned int *))sub_8D1B30,
                         2 * (unsigned int)(dword_4F077C4 != 2) + 129);
  }
  return result;
}
