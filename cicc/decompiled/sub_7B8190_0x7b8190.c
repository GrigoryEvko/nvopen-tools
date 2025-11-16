// Function: sub_7B8190
// Address: 0x7b8190
//
_QWORD *sub_7B8190()
{
  __int64 v0; // rbx
  char v1; // dl
  char v2; // cl
  char v3; // al

  v0 = qword_4F08528;
  if ( qword_4F08528 )
    qword_4F08528 = *(_QWORD *)qword_4F08528;
  else
    v0 = sub_823970(64);
  *(_QWORD *)v0 = 0;
  *(_QWORD *)(v0 + 8) = 0;
  *(_QWORD *)(v0 + 16) = *(_QWORD *)&dword_4F077C8;
  sub_7ADF70(v0 + 24, 0);
  v1 = *(_BYTE *)(v0 + 56) & 0xF0;
  *(_QWORD *)v0 = qword_4F061C0;
  qword_4F061C0 = (_QWORD *)v0;
  v2 = unk_4F04D84 & 1;
  *(_QWORD *)(v0 + 16) = *(_QWORD *)dword_4F07508;
  v3 = dword_4F04D80;
  dword_4F04D80 = 0;
  *(_BYTE *)(v0 + 56) = v1 | v2 | (2 * (v3 & 1));
  sub_7B80F0();
  qword_4F061D0 = *(_QWORD *)&dword_4F077C8;
  return &qword_4F061D0;
}
