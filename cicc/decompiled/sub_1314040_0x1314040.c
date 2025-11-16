// Function: sub_1314040
// Address: 0x1314040
//
void sub_1314040()
{
  _BYTE *v0; // rdi

  v0 = (_BYTE *)(__readfsqword(0) - 2664);
  if ( __readfsbyte(0xFFFFF8C8) )
    v0 = (_BYTE *)sub_1313D30((__int64)v0, 0);
  sub_1313A40(v0);
}
