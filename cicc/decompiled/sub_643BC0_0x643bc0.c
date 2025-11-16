// Function: sub_643BC0
// Address: 0x643bc0
//
__int64 sub_643BC0()
{
  __int64 v0; // r8

  sub_823970(472);
  v0 = qword_4CFDE78;
  if ( qword_4CFDE78 )
    qword_4CFDE78 = *(_QWORD *)(qword_4CFDE78 + 464);
  else
    v0 = sub_823970(472);
  *(_QWORD *)v0 = 0;
  *(_QWORD *)(v0 + 464) = 0;
  memset(
    (void *)((v0 + 8) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)v0 - (((_DWORD)v0 + 8) & 0xFFFFFFF8) + 472) >> 3));
  *(_QWORD *)(v0 + 152) = v0;
  *(_QWORD *)(v0 + 24) = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    *(_BYTE *)(v0 + 178) |= 1u;
  return v0;
}
