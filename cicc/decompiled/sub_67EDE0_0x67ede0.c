// Function: sub_67EDE0
// Address: 0x67ede0
//
void sub_67EDE0()
{
  _BYTE *v0; // rax

  dword_4D03A14 = 0;
  unk_4F074A0 = 0;
  unk_4F074A8 = 0;
  unk_4F074B0 = 0;
  unk_4F074B8 = 0;
  unk_4F074C0 = 0;
  unk_4F074C8 = 0;
  unk_4F074D0 = 0;
  unk_4F074D8 = 0;
  unk_4F074E0 = 0;
  unk_4F074E8 = 0;
  unk_4F074F0 = 0;
  unk_4F074F8 = 0;
  unk_4F07500 = 0;
  qword_4D039F8 = 0;
  unk_4F07488 = 0;
  qword_4D039F0 = 0;
  unk_4D03A10 = 0;
  dword_4D03A04 = 0;
  unk_4D03A0C = 0;
  unk_4F07490 = 0;
  memset(qword_4CFDEC0, 0, sizeof(qword_4CFDEC0));
  if ( !dword_4CFFE68 )
  {
    v0 = byte_4CFFE80;
    do
    {
      v0[2] &= 0xF8u;
      v0 += 4;
      *(v0 - 3) = 0;
    }
    while ( v0 != &byte_4CFFE80[15180] );
  }
  dword_4CFFE68 = 0;
  sub_67D0D0();
  dword_4CFDEAC = 0;
  dword_4CFDEA8 = 0;
  dword_4CFDEA4 = 0;
  dword_4CFDEA0 = 0;
}
