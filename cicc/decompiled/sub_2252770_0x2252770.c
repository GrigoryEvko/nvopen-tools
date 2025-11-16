// Function: sub_2252770
// Address: 0x2252770
//
_QWORD *__fastcall sub_2252770(__int64 a1)
{
  _QWORD *v1; // rax

  v1 = (_QWORD *)malloc(a1 + 128);
  if ( !v1 )
  {
    v1 = sub_22526A0(a1 + 128);
    if ( !v1 )
      sub_2207530();
  }
  *v1 = 0;
  v1[15] = 0;
  memset(
    (void *)((unsigned __int64)(v1 + 1) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)v1 - (((_DWORD)v1 + 8) & 0xFFFFFFF8) + 128) >> 3));
  return v1 + 16;
}
