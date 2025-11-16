// Function: sub_823E10
// Address: 0x823e10
//
_QWORD *sub_823E10()
{
  int v0; // eax

  dword_4F195D8 = 1;
  v0 = unk_4D04508;
  if ( unk_4D04508 )
  {
    sub_8539C0(&off_4B7D3C0);
    v0 = unk_4D04508;
  }
  dword_4F195C8 = 0;
  qword_4F195C0 = 0;
  qword_4F195B8 = 0;
  dword_4F195D8 = v0 == 0;
  qword_4F07380 = 0;
  sub_8D0840(dword_4F073B8, 4, 0);
  qword_4F195E0 = 0;
  qword_4F195D0 = 0;
  qword_4F072B0 = 0;
  qword_4F072B8 = 0;
  return &qword_4F07280;
}
