// Function: sub_67B660
// Address: 0x67b660
//
__int64 sub_67B660()
{
  bool v0; // dl

  if ( !qword_4D039E8 )
  {
    qword_4D039E8 = sub_8237A0(1024);
    qword_4D039E0 = sub_8237A0(128);
  }
  ((void (*)(void))sub_823800)();
  sub_823800(qword_4D039E0);
  sub_7461E0(&qword_4CFFDC0);
  qword_4CFFDC0 = (__int64)sub_729390;
  qword_4CFFDD0 = qword_4D039E8;
  byte_4CFFE49 = dword_4F077C4 == 1;
  v0 = 0;
  if ( dword_4F077C4 != 2 )
    v0 = unk_4F07778 > 199900;
  byte_4CFFE4C = v0;
  byte_4CFFE51 = dword_4F07460;
  return dword_4F07460;
}
