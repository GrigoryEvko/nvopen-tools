// Function: sub_7B80F0
// Address: 0x7b80f0
//
_QWORD *sub_7B80F0()
{
  _QWORD *v0; // rdx

  v0 = (_QWORD *)qword_4F08530;
  if ( qword_4F08530 )
    qword_4F08530 = *(_QWORD *)qword_4F08530;
  else
    v0 = (_QWORD *)sub_823970(368);
  *v0 = 0;
  v0[1] = 0;
  *(_QWORD *)((char *)v0 + 358) = 0;
  memset(
    (void *)((unsigned __int64)(v0 + 2) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)v0 - (((_DWORD)v0 + 16) & 0xFFFFFFF8) + 366) >> 3));
  *v0 = qword_4F061C8;
  qword_4F061C8 = v0;
  return &qword_4F061C8;
}
