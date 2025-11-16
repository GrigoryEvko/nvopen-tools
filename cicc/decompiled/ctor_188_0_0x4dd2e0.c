// Function: ctor_188_0
// Address: 0x4dd2e0
//
int ctor_188_0()
{
  int v0; // eax
  int v1; // eax
  int v2; // eax
  int v4; // [rsp+1Ch] [rbp-F4h] BYREF
  _QWORD v5[2]; // [rsp+20h] [rbp-F0h] BYREF
  _QWORD v6[2]; // [rsp+30h] [rbp-E0h] BYREF
  _QWORD v7[2]; // [rsp+40h] [rbp-D0h] BYREF
  int v8; // [rsp+50h] [rbp-C0h]
  const char *v9; // [rsp+58h] [rbp-B8h]
  __int64 v10; // [rsp+60h] [rbp-B0h]
  char *v11; // [rsp+68h] [rbp-A8h]
  __int64 v12; // [rsp+70h] [rbp-A0h]
  int v13; // [rsp+78h] [rbp-98h]
  const char *v14; // [rsp+80h] [rbp-90h]
  __int64 v15; // [rsp+88h] [rbp-88h]
  char *v16; // [rsp+90h] [rbp-80h]
  __int64 v17; // [rsp+98h] [rbp-78h]
  int v18; // [rsp+A0h] [rbp-70h]
  const char *v19; // [rsp+A8h] [rbp-68h]
  __int64 v20; // [rsp+B0h] [rbp-60h]

  qword_4FAC500 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAC50C &= 0xF000u;
  qword_4FAC510 = 0;
  qword_4FAC518 = 0;
  qword_4FAC520 = 0;
  qword_4FAC528 = 0;
  dword_4FAC508 = v0;
  qword_4FAC558 = (__int64)&unk_4FAC578;
  qword_4FAC560 = (__int64)&unk_4FAC578;
  qword_4FAC530 = 0;
  qword_4FAC548 = (__int64)qword_4FA01C0;
  qword_4FAC5A8 = (__int64)&unk_49E74E8;
  word_4FAC5B0 = 256;
  qword_4FAC538 = 0;
  qword_4FAC540 = 0;
  qword_4FAC500 = (__int64)&unk_49EEC70;
  qword_4FAC550 = 0;
  byte_4FAC598 = 0;
  qword_4FAC5B8 = (__int64)&unk_49EEDB0;
  qword_4FAC568 = 4;
  dword_4FAC570 = 0;
  byte_4FAC5A0 = 0;
  sub_16B8280(&qword_4FAC500, "lowertypetests-avoid-reuse", 26);
  qword_4FAC528 = (__int64)"Try to avoid reuse of byte array addresses using aliases";
  word_4FAC5B0 = 257;
  byte_4FAC5A0 = 1;
  qword_4FAC530 = 56;
  LOBYTE(word_4FAC50C) = word_4FAC50C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FAC500);
  __cxa_atexit(sub_12EDEC0, &qword_4FAC500, &qword_4A427C0);
  v4 = 1;
  v7[0] = "none";
  v9 = "Do nothing";
  v11 = "import";
  v14 = "Import typeid resolutions from summary and globals";
  v16 = "export";
  v19 = "Export typeid resolutions to summary and globals";
  v6[1] = 0x400000003LL;
  v6[0] = v7;
  v7[1] = 4;
  v8 = 0;
  v10 = 10;
  v12 = 6;
  v13 = 1;
  v15 = 50;
  v17 = 6;
  v18 = 2;
  v20 = 48;
  v5[0] = "What to do with the summary when running this pass";
  v5[1] = 50;
  sub_187D6D0(&unk_4FAC2A0, "lowertypetests-summary-action", v5, v6, &v4);
  if ( (_QWORD *)v6[0] != v7 )
    _libc_free(v6[0], "lowertypetests-summary-action");
  __cxa_atexit(sub_1872400, &unk_4FAC2A0, &qword_4A427C0);
  qword_4FAC180 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAC18C &= 0xF000u;
  qword_4FAC190 = 0;
  qword_4FAC198 = 0;
  qword_4FAC1A0 = 0;
  qword_4FAC1A8 = 0;
  qword_4FAC1B0 = 0;
  dword_4FAC188 = v1;
  qword_4FAC1D8 = (__int64)&unk_4FAC1F8;
  qword_4FAC1E0 = (__int64)&unk_4FAC1F8;
  qword_4FAC220 = (__int64)&byte_4FAC230;
  qword_4FAC248 = (__int64)&byte_4FAC258;
  qword_4FAC1C8 = (__int64)qword_4FA01C0;
  qword_4FAC1B8 = 0;
  qword_4FAC240 = (__int64)&unk_49EED10;
  qword_4FAC180 = (__int64)&unk_49EEBF0;
  qword_4FAC1C0 = 0;
  qword_4FAC278 = (__int64)&byte_4FAC288;
  qword_4FAC270 = (__int64)&unk_49EEE90;
  qword_4FAC1D0 = 0;
  qword_4FAC1E8 = 4;
  dword_4FAC1F0 = 0;
  byte_4FAC218 = 0;
  qword_4FAC228 = 0;
  byte_4FAC230 = 0;
  qword_4FAC250 = 0;
  byte_4FAC258 = 0;
  byte_4FAC268 = 0;
  qword_4FAC280 = 0;
  byte_4FAC288 = 0;
  sub_16B8280(&byte_4FAC288 - 264, "lowertypetests-read-summary", 27);
  qword_4FAC1B0 = 53;
  qword_4FAC1A8 = (__int64)"Read summary from given YAML file before running pass";
  LOBYTE(word_4FAC18C) = word_4FAC18C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FAC180);
  __cxa_atexit(sub_12F0C20, &qword_4FAC180, &qword_4A427C0);
  qword_4FAC060 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FAC06C &= 0xF000u;
  qword_4FAC070 = 0;
  qword_4FAC078 = 0;
  qword_4FAC080 = 0;
  qword_4FAC120 = (__int64)&unk_49EED10;
  qword_4FAC088 = 0;
  dword_4FAC068 = v2;
  qword_4FAC0B8 = (__int64)&unk_4FAC0D8;
  qword_4FAC0C0 = (__int64)&unk_4FAC0D8;
  qword_4FAC100 = (__int64)&byte_4FAC110;
  qword_4FAC128 = (__int64)&byte_4FAC138;
  qword_4FAC0A8 = (__int64)qword_4FA01C0;
  qword_4FAC060 = (__int64)&unk_49EEBF0;
  qword_4FAC150 = (__int64)&unk_49EEE90;
  qword_4FAC158 = (__int64)&byte_4FAC168;
  qword_4FAC090 = 0;
  qword_4FAC098 = 0;
  qword_4FAC0A0 = 0;
  qword_4FAC0B0 = 0;
  qword_4FAC0C8 = 4;
  dword_4FAC0D0 = 0;
  byte_4FAC0F8 = 0;
  qword_4FAC108 = 0;
  byte_4FAC110 = 0;
  qword_4FAC130 = 0;
  byte_4FAC138 = 0;
  byte_4FAC148 = 0;
  qword_4FAC160 = 0;
  byte_4FAC168 = 0;
  sub_16B8280(&byte_4FAC168 - 264, "lowertypetests-write-summary", 28);
  qword_4FAC090 = 51;
  qword_4FAC088 = (__int64)"Write summary to given YAML file after running pass";
  LOBYTE(word_4FAC06C) = word_4FAC06C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FAC060);
  return __cxa_atexit(sub_12F0C20, &qword_4FAC060, &qword_4A427C0);
}
