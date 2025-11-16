// Function: sub_85E950
// Address: 0x85e950
//
_QWORD *sub_85E950()
{
  _QWORD *v0; // r8
  _QWORD *result; // rax

  v0 = (_QWORD *)qword_4F5FD50;
  if ( qword_4F5FD50 )
  {
    qword_4F5FD50 = *(_QWORD *)qword_4F5FD50;
    *v0 = 0;
    return v0;
  }
  else
  {
    result = (_QWORD *)sub_823970(256);
    result[1] = 0;
    result[31] = 0;
    memset(
      (void *)((unsigned __int64)(result + 2) & 0xFFFFFFFFFFFFFFF8LL),
      0,
      8LL * (((unsigned int)result - (((_DWORD)result + 16) & 0xFFFFFFF8) + 256) >> 3));
    *result = 0;
  }
  return result;
}
