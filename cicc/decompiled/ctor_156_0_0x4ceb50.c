// Function: ctor_156_0
// Address: 0x4ceb50
//
int ctor_156_0()
{
  __int64 v0; // rax
  __int64 v1; // r13
  int v2; // eax
  __int64 *v3; // rax
  __int64 v4; // r12
  int v5; // eax
  __int64 *v6; // rax
  __int64 v7; // rax
  __int64 *v9; // rsi
  __int64 *v10; // rcx
  __int64 *v11; // rsi
  __int64 *v12; // rcx
  __int64 v13; // [rsp+20h] [rbp-80h]
  __int64 v14; // [rsp+20h] [rbp-80h]
  int v15; // [rsp+30h] [rbp-70h] BYREF
  int v16; // [rsp+34h] [rbp-6Ch] BYREF
  __int64 *v17; // [rsp+38h] [rbp-68h] BYREF
  __int64 *v18; // [rsp+40h] [rbp-60h] BYREF
  __int64 v19; // [rsp+48h] [rbp-58h] BYREF
  const char *v20; // [rsp+50h] [rbp-50h] BYREF
  __int64 v21; // [rsp+58h] [rbp-48h]
  char v22; // [rsp+60h] [rbp-40h]
  char v23; // [rsp+61h] [rbp-3Fh]

  dword_4FA0208[0] = 0;
  *(_QWORD *)&dword_4FA0208[2] = 0;
  *(_QWORD *)&dword_4FA0208[4] = dword_4FA0208;
  *(_QWORD *)&dword_4FA0208[6] = dword_4FA0208;
  *(_QWORD *)&dword_4FA0208[8] = 0;
  __cxa_atexit(sub_C4FEF0, &unk_4FA0200, &qword_4A427C0);
  qword_4FA01C0[2] = byte_3F871B3;
  qword_4FA01C0[0] = "General options";
  qword_4FA01C0[1] = 15;
  qword_4FA01C0[3] = 0;
  sub_16B1A80(qword_4FA01C0);
  qword_4FA0150 = (__int64)off_4985000;
  byte_4FA0158 = 0;
  __cxa_atexit(nullsub_605, &qword_4FA0150, &qword_4A427C0);
  qword_4FA0140 = (__int64)off_4985028;
  byte_4FA0148 = 0;
  __cxa_atexit(nullsub_606, &qword_4FA0140, &qword_4A427C0);
  qword_4FA0130 = (__int64)off_4985000;
  byte_4FA0138 = 0;
  __cxa_atexit(nullsub_605, &qword_4FA0130, &qword_4A427C0);
  qword_4FA0120 = (__int64)off_4985000;
  byte_4FA0128 = 1;
  __cxa_atexit(nullsub_605, &qword_4FA0120, &qword_4A427C0);
  qword_4FA0110 = (__int64)off_4985028;
  byte_4FA0118 = 0;
  __cxa_atexit(nullsub_606, &qword_4FA0110, &qword_4A427C0);
  qword_4FA0100 = (__int64)off_4985028;
  byte_4FA0108 = 1;
  __cxa_atexit(nullsub_606, &qword_4FA0100, &qword_4A427C0);
  qword_4FA00F0 = (__int64)&qword_4FA0130;
  qword_4FA00F8 = (__int64)&qword_4FA0110;
  qword_4FA00D0 = (__int64)byte_3F871B3;
  qword_4FA00E0 = (__int64)&qword_4FA0120;
  qword_4FA00E8 = (__int64)&qword_4FA0100;
  qword_4FA00C0 = (__int64)"Generic Options";
  qword_4FA00C8 = 15;
  qword_4FA00D8 = 0;
  sub_16B1A80(&qword_4FA00C0);
  v18 = &qword_4FA00C0;
  v19 = sub_16B4B80(&unk_4FA0170);
  v20 = "Display list of available options (-help-list-hidden for more)";
  v17 = &qword_4FA0130;
  v16 = 3;
  v15 = 1;
  v21 = 62;
  sub_4CEA00((__int64)&unk_4F9FFE0, "help-list", (__int64 *)&v20, &v17, &v15, &v16, &v18, &v19);
  __cxa_atexit(sub_16B00F0, &unk_4F9FFE0, &qword_4A427C0);
  v0 = sub_16B4B80(&unk_4FA0170);
  v20 = "Display list of all available options";
  v19 = v0;
  v17 = &qword_4FA0120;
  v18 = &qword_4FA00C0;
  v16 = 3;
  v15 = 1;
  v21 = 37;
  sub_4CEA00((__int64)&unk_4F9FF00, "help-list-hidden", (__int64 *)&v20, &v17, &v15, &v16, &v18, &v19);
  __cxa_atexit(sub_16B00F0, &unk_4F9FF00, &qword_4A427C0);
  v13 = sub_16B4B80(&unk_4FA0170);
  sub_12F0CE0(&qword_4F9FE20, 0, 0);
  qword_4F9FEC8 = (__int64)off_49EEF30;
  qword_4F9FE20 = (__int64)off_49EEFD0;
  qword_4F9FED8 = (__int64)&qword_4FA0150;
  qword_4F9FED0 = (__int64)&unk_49EEDB0;
  qword_4F9FEE0 = (__int64)&qword_4FA0140;
  qword_4F9FEC0 = 0;
  sub_16B8280(&qword_4F9FE20, "help", 4);
  qword_4F9FE50 = 49;
  qword_4F9FE48 = (__int64)"Display available options (-help-hidden for more)";
  sub_4CE990(&qword_4F9FEC0, (__int64)&qword_4F9FE20, (__int64)&qword_4FA00F0);
  byte_4F9FE2C |= 0x18u;
  qword_4F9FE68 = (__int64)&qword_4FA00C0;
  sub_4CE9E0(v13, (__int64)&qword_4F9FE20);
  sub_16B88A0(&qword_4F9FE20);
  __cxa_atexit(sub_16B00A0, &qword_4F9FE20, &qword_4A427C0);
  v14 = sub_16B4B80(&unk_4FA0170);
  sub_12F0CE0(&qword_4F9FD40, 0, 0);
  qword_4F9FDE8 = (__int64)off_49EEF30;
  qword_4F9FD40 = (__int64)off_49EEFD0;
  qword_4F9FDF0 = (__int64)&unk_49EEDB0;
  qword_4F9FDF8 = (__int64)&qword_4FA0150;
  qword_4F9FE00 = (__int64)&qword_4FA0140;
  qword_4F9FDE0 = 0;
  sub_16B8280(&qword_4F9FD40, "help-hidden", 11);
  qword_4F9FD70 = 29;
  qword_4F9FD68 = (__int64)"Display all available options";
  sub_4CE990(&qword_4F9FDE0, (__int64)&qword_4F9FD40, (__int64)&qword_4FA00E0);
  qword_4F9FD88 = (__int64)&qword_4FA00C0;
  byte_4F9FD4C = byte_4F9FD4C & 0x87 | 0x38;
  sub_4CE9E0(v14, (__int64)&qword_4F9FD40);
  sub_16B88A0(&qword_4F9FD40);
  __cxa_atexit(sub_16B00A0, &qword_4F9FD40, &qword_4A427C0);
  v1 = sub_16B4B80(&unk_4FA0170);
  qword_4F9FC60 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9FC6C &= 0xF000u;
  qword_4F9FC70 = 0;
  qword_4F9FC78 = 0;
  qword_4F9FC80 = 0;
  qword_4F9FC88 = 0;
  dword_4F9FC68 = v2;
  qword_4F9FCB8 = (__int64)&unk_4F9FCD8;
  qword_4F9FCC0 = (__int64)&unk_4F9FCD8;
  qword_4F9FC90 = 0;
  qword_4F9FCA8 = (__int64)qword_4FA01C0;
  qword_4F9FD08 = (__int64)&unk_49E74E8;
  qword_4F9FC98 = 0;
  qword_4F9FCA0 = 0;
  qword_4F9FC60 = (__int64)&unk_49EEC70;
  qword_4F9FCB0 = 0;
  qword_4F9FCC8 = 4;
  dword_4F9FCD0 = 0;
  byte_4F9FCF8 = 0;
  byte_4F9FD00 = 0;
  word_4F9FD10 = 256;
  qword_4F9FD18 = (__int64)&unk_49EEDB0;
  sub_16B8280(&qword_4F9FC60, "print-options", 13);
  qword_4F9FC90 = 52;
  qword_4F9FC88 = (__int64)"Print non-default options after command line parsing";
  byte_4F9FD00 = 0;
  word_4F9FD10 = 256;
  qword_4F9FCA8 = (__int64)&qword_4FA00C0;
  LOBYTE(word_4F9FC6C) = word_4F9FC6C & 0x9F | 0x20;
  v3 = (__int64 *)qword_4F9FCB8;
  if ( qword_4F9FCC0 != qword_4F9FCB8 )
    goto LABEL_2;
  v9 = (__int64 *)(qword_4F9FCB8 + 8LL * HIDWORD(qword_4F9FCC8));
  if ( (__int64 *)qword_4F9FCB8 == v9 )
  {
LABEL_27:
    if ( HIDWORD(qword_4F9FCC8) >= (unsigned int)qword_4F9FCC8 )
    {
LABEL_2:
      sub_16CCBA0(&qword_4F9FCB0, v1);
      goto LABEL_3;
    }
    ++HIDWORD(qword_4F9FCC8);
    *v9 = v1;
    ++qword_4F9FCB0;
  }
  else
  {
    v10 = 0;
    while ( v1 != *v3 )
    {
      if ( *v3 == -2 )
        v10 = v3;
      if ( v9 == ++v3 )
      {
        if ( !v10 )
          goto LABEL_27;
        *v10 = v1;
        --dword_4F9FCD0;
        ++qword_4F9FCB0;
        break;
      }
    }
  }
LABEL_3:
  sub_16B88A0(&qword_4F9FC60);
  __cxa_atexit(sub_12EDEC0, &qword_4F9FC60, &qword_4A427C0);
  v4 = sub_16B4B80(&unk_4FA0170);
  qword_4F9FB80 = (__int64)&unk_49EED30;
  v5 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4F9FB8C &= 0xF000u;
  qword_4F9FB90 = 0;
  qword_4F9FB98 = 0;
  qword_4F9FBA0 = 0;
  qword_4F9FBA8 = 0;
  dword_4F9FB88 = v5;
  qword_4F9FBD8 = (__int64)&unk_4F9FBF8;
  qword_4F9FBE0 = (__int64)&unk_4F9FBF8;
  qword_4F9FBB0 = 0;
  qword_4F9FBC8 = (__int64)qword_4FA01C0;
  qword_4F9FC28 = (__int64)&unk_49E74E8;
  word_4F9FC30 = 256;
  qword_4F9FBB8 = 0;
  qword_4F9FBC0 = 0;
  qword_4F9FB80 = (__int64)&unk_49EEC70;
  qword_4F9FBD0 = 0;
  byte_4F9FC18 = 0;
  qword_4F9FC38 = (__int64)&unk_49EEDB0;
  qword_4F9FBE8 = 4;
  dword_4F9FBF0 = 0;
  byte_4F9FC20 = 0;
  sub_16B8280(&qword_4F9FB80, "print-all-options", 17);
  qword_4F9FBB0 = 50;
  qword_4F9FBA8 = (__int64)"Print all option values after command line parsing";
  byte_4F9FC20 = 0;
  word_4F9FC30 = 256;
  qword_4F9FBC8 = (__int64)&qword_4FA00C0;
  LOBYTE(word_4F9FB8C) = word_4F9FB8C & 0x9F | 0x20;
  v6 = (__int64 *)qword_4F9FBD8;
  if ( qword_4F9FBE0 != qword_4F9FBD8 )
  {
LABEL_4:
    sub_16CCBA0(&qword_4F9FBD0, v4);
    goto LABEL_5;
  }
  v11 = (__int64 *)(qword_4F9FBD8 + 8LL * HIDWORD(qword_4F9FBE8));
  if ( (__int64 *)qword_4F9FBD8 == v11 )
  {
LABEL_25:
    if ( HIDWORD(qword_4F9FBE8) >= (unsigned int)qword_4F9FBE8 )
      goto LABEL_4;
    ++HIDWORD(qword_4F9FBE8);
    *v11 = v4;
    ++qword_4F9FBD0;
  }
  else
  {
    v12 = 0;
    while ( v4 != *v6 )
    {
      if ( *v6 == -2 )
        v12 = v6;
      if ( v11 == ++v6 )
      {
        if ( !v12 )
          goto LABEL_25;
        *v12 = v4;
        --dword_4F9FBF0;
        ++qword_4F9FBD0;
        break;
      }
    }
  }
LABEL_5:
  sub_16B88A0(&qword_4F9FB80);
  __cxa_atexit(sub_12EDEC0, &qword_4F9FB80, &qword_4A427C0);
  qword_4F9FB70 = 0;
  __cxa_atexit(sub_16B05A0, &unk_4F9FB60, &qword_4A427C0);
  sub_12F0CE0(&qword_4F9FA80, 0, 0);
  qword_4F9FB28 = (__int64)off_49EEF30;
  qword_4F9FA80 = (__int64)off_49EF050;
  qword_4F9FB20 = 0;
  qword_4F9FB30 = (__int64)&unk_49EEDB0;
  sub_16B8280(&qword_4F9FA80, "version", 7);
  qword_4F9FAB0 = 35;
  qword_4F9FAA8 = (__int64)"Display the version of this program";
  if ( qword_4F9FB20 )
  {
    v7 = sub_16E8CB0();
    v23 = 1;
    v20 = "cl::location(x) specified more than once!";
    v22 = 3;
    sub_16B1F90(&qword_4F9FA80, &v20, 0, 0, v7);
  }
  else
  {
    qword_4F9FB20 = (__int64)&unk_4F9FB40;
  }
  byte_4F9FA8C |= 0x18u;
  qword_4F9FAC8 = (__int64)&qword_4FA00C0;
  sub_16B88A0(&qword_4F9FA80);
  return __cxa_atexit(sub_16B0050, &qword_4F9FA80, &qword_4A427C0);
}
