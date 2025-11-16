// Function: sub_6E2EF0
// Address: 0x6e2ef0
//
_QWORD *sub_6E2EF0()
{
  _QWORD *v0; // r12

  v0 = (_QWORD *)qword_4D03A88;
  if ( qword_4D03A88 )
    qword_4D03A88 = *(_QWORD *)qword_4D03A88;
  else
    v0 = (_QWORD *)sub_823970(360);
  *v0 = 0;
  sub_6E2E50(0, (__int64)(v0 + 1));
  return v0;
}
