// Function: ctor_233
// Address: 0x4eb6e0
//
int ctor_233()
{
  __int64 v0; // rax
  const char *v2; // [rsp+0h] [rbp-50h] BYREF
  char v3; // [rsp+10h] [rbp-40h]
  char v4; // [rsp+11h] [rbp-3Fh]

  sub_12F0CE0(&qword_4FB61A0, 0, 0);
  word_4FB6250 = 256;
  byte_4FB6240 = 0;
  qword_4FB61A0 = (__int64)&unk_49EEC70;
  qword_4FB6248 = (__int64)&unk_49E74E8;
  qword_4FB6258 = (__int64)&unk_49EEDB0;
  sub_16B8280(&qword_4FB61A0, "spp-print-liveset", 17);
  word_4FB6250 = 256;
  byte_4FB6240 = 0;
  byte_4FB61AC = byte_4FB61AC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FB61A0);
  __cxa_atexit(sub_12EDEC0, &qword_4FB61A0, &qword_4A427C0);
  sub_12F0CE0(&qword_4FB60C0, 0, 0);
  word_4FB6170 = 256;
  byte_4FB6160 = 0;
  qword_4FB6168 = (__int64)&unk_49E74E8;
  qword_4FB60C0 = (__int64)&unk_49EEC70;
  qword_4FB6178 = (__int64)&unk_49EEDB0;
  sub_16B8280(&qword_4FB60C0, "spp-print-liveset-size", 22);
  word_4FB6170 = 256;
  byte_4FB6160 = 0;
  byte_4FB60CC = byte_4FB60CC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FB60C0);
  __cxa_atexit(sub_12EDEC0, &qword_4FB60C0, &qword_4A427C0);
  sub_12F0CE0(&qword_4FB5FE0, 0, 0);
  word_4FB6090 = 256;
  byte_4FB6080 = 0;
  qword_4FB6088 = (__int64)&unk_49E74E8;
  qword_4FB5FE0 = (__int64)&unk_49EEC70;
  qword_4FB6098 = (__int64)&unk_49EEDB0;
  sub_16B8280(&qword_4FB5FE0, "spp-print-base-pointers", 23);
  word_4FB6090 = 256;
  byte_4FB6080 = 0;
  byte_4FB5FEC = byte_4FB5FEC & 0x9F | 0x20;
  sub_16B88A0(&qword_4FB5FE0);
  __cxa_atexit(sub_12EDEC0, &qword_4FB5FE0, &qword_4A427C0);
  sub_12F0CE0(&qword_4FB5F00, 0, 0);
  dword_4FB5FA0 = 0;
  byte_4FB5FB4 = 1;
  qword_4FB5FA8 = (__int64)&unk_49E74A8;
  dword_4FB5FB0 = 0;
  qword_4FB5F00 = (__int64)&unk_49EEAF0;
  qword_4FB5FB8 = (__int64)&unk_49EEE10;
  sub_16B8280(&qword_4FB5F00, "spp-rematerialization-threshold", 31);
  dword_4FB5FA0 = 6;
  byte_4FB5FB4 = 1;
  dword_4FB5FB0 = 6;
  byte_4FB5F0C = byte_4FB5F0C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FB5F00);
  __cxa_atexit(sub_12EDE60, &qword_4FB5F00, &qword_4A427C0);
  sub_12F0CE0(&qword_4FB5E20, 0, 0);
  byte_4FB5ED1 = 0;
  qword_4FB5EC8 = (__int64)&unk_49E74E8;
  qword_4FB5EC0 = 0;
  qword_4FB5E20 = (__int64)&unk_49EAB58;
  qword_4FB5ED8 = (__int64)&unk_49EEDB0;
  sub_16B8280(&qword_4FB5E20, "rs4gc-clobber-non-live", 22);
  if ( qword_4FB5EC0 )
  {
    v0 = sub_16E8CB0();
    v4 = 1;
    v2 = "cl::location(x) specified more than once!";
    v3 = 3;
    sub_16B1F90(&qword_4FB5E20, &v2, 0, 0, v0);
  }
  else
  {
    byte_4FB5ED1 = 1;
    qword_4FB5EC0 = (__int64)&byte_4FB5EE8;
    byte_4FB5ED0 = byte_4FB5EE8;
  }
  byte_4FB5E2C = byte_4FB5E2C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FB5E20);
  __cxa_atexit(sub_13F9A70, &qword_4FB5E20, &qword_4A427C0);
  sub_12F0CE0(&qword_4FB5D40, 0, 0);
  qword_4FB5DE8 = (__int64)&unk_49E74E8;
  word_4FB5DF0 = 256;
  byte_4FB5DE0 = 0;
  qword_4FB5D40 = (__int64)&unk_49EEC70;
  qword_4FB5DF8 = (__int64)&unk_49EEDB0;
  sub_16B8280(&qword_4FB5D40, "rs4gc-allow-statepoint-with-no-deopt-info", 41);
  word_4FB5DF0 = 257;
  byte_4FB5DE0 = 1;
  byte_4FB5D4C = byte_4FB5D4C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FB5D40);
  return __cxa_atexit(sub_12EDEC0, &qword_4FB5D40, &qword_4A427C0);
}
