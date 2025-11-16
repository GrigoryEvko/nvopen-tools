// Function: sub_7277D0
// Address: 0x7277d0
//
__int64 sub_7277D0()
{
  xmmword_4F079A0 = 0u;
  xmmword_4F079B0 = 0u;
  unk_4F06CF8 = (dword_4F077C4 != 2) + 2;
  xmmword_4F079C0 = 0u;
  xmmword_4F079D0 = 0u;
  xmmword_4F079E0 = *(unsigned __int64 *)&dword_4F077C8;
  *(_QWORD *)&xmmword_4F079F0 = 0;
  DWORD2(xmmword_4F079F0) = DWORD2(xmmword_4F079F0) & 0xF8000000 | 4;
  xmmword_4F07A00 = 0u;
  if ( unk_4D04508 )
    sub_8539C0(&off_4B6EB40);
  sub_8D0840(&dword_4F0798C, 4, 0);
  sub_8D0840(&qword_4F07978, 8, 0);
  sub_8D0840(&qword_4F07970, 8, 0);
  sub_8D0840(&qword_4F06BB8, 8, 0);
  sub_8D0840(&qword_4F06BB0, 8, 0);
  sub_8D0840(&qword_4F07968, 8, 0);
  return sub_8D0840(&dword_4F07988, 4, 0);
}
