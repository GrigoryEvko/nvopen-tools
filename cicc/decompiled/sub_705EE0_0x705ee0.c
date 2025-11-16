// Function: sub_705EE0
// Address: 0x705ee0
//
__int64 sub_705EE0()
{
  char *v0; // rsi
  bool v1; // al
  time_t timer; // [rsp+8h] [rbp-18h] BYREF

  time(&timer);
  v0 = ctime(&timer);
  if ( !v0 )
    v0 = "Sun Jan 01 00:00:00 1900\n";
  strcpy(dest, v0);
  dword_4F07588 = 1;
  unk_4D03B90 = -1;
  sub_67EDE0();
  nullsub_8();
  sub_77FA70();
  sub_823FD0();
  sub_723DD0();
  unk_4F076B8 = sub_722560(qword_4F076F0, ".o");
  sub_750620();
  sub_738210();
  sub_76C4D0();
  sub_7CA900();
  sub_887D50();
  sub_867FB0();
  sub_666F10();
  sub_62F580();
  nullsub_3();
  sub_603B70();
  sub_7AB670();
  sub_6797C0();
  sub_89F510();
  sub_8E3910();
  sub_8D05E0();
  sub_6F0720();
  nullsub_10();
  sub_822A50();
  sub_8754D0();
  sub_8539E0();
  sub_854CA0();
  sub_858EA0();
  sub_88D3E0();
  sub_622C70();
  sub_70C940();
  sub_7F4F10();
  sub_7DAF20();
  sub_80D400();
  sub_5D2A00();
  sub_825260();
  if ( dword_4F077C4 == 2 && unk_4F06968 )
    unk_4F068FC = 0;
  unk_4F07310 = 0;
  unk_4F072C8 = dword_4F077C4 != 2;
  unk_4F072A8 = unk_4F06B98;
  unk_4F072D0 = dword_4F077C4 == 1;
  unk_4F07308 = 0;
  unk_4F072CC = unk_4F07778;
  unk_4F072D1 = unk_4D04000;
  unk_4F072D4 = unk_4D0438C;
  unk_4F072D8 = dword_4F077C0;
  unk_4F072D9 = dword_4F077BC;
  unk_4F072DA = qword_4F077B4;
  unk_4F072E0 = qword_4F077A8;
  unk_4F072E8 = qword_4F077A0;
  unk_4F072F3 = 0;
  unk_4F072F4 = 0;
  unk_4F072F6 = dword_4F07590;
  v1 = 0;
  if ( unk_4F0758C )
    v1 = dword_4D047B0 != 0;
  unk_4F072F7 = v1;
  unk_4F072F8 = dword_4F077C4 != 2;
  if ( unk_4D04940 )
  {
    if ( qword_4D04920 )
      unk_4D04928 = sub_685E40(qword_4D04920, 0, 0, 16, 1513);
    else
      unk_4D04928 = stdout;
  }
  unk_4F07280 = 0;
  return sub_720FB0();
}
