// Function: sub_1322320
// Address: 0x1322320
//
_QWORD *__fastcall sub_1322320(__int64 a1)
{
  _BYTE *v2; // rdi

  v2 = (_BYTE *)(__readfsqword(0) - 2664);
  if ( __readfsbyte(0xFFFFF8C8) )
    v2 = (_BYTE *)sub_1313D30((__int64)v2, 0);
  return sub_1322110(v2, a1, 1, 0);
}
