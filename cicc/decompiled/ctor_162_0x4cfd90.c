// Function: ctor_162
// Address: 0x4cfd90
//
int ctor_162()
{
  int v0; // eax
  int v1; // eax

  qword_4FA1320 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA132C &= 0xF000u;
  qword_4FA1330 = 0;
  qword_4FA1338 = 0;
  qword_4FA1340 = 0;
  qword_4FA1348 = 0;
  qword_4FA1350 = 0;
  dword_4FA1328 = v0;
  qword_4FA1358 = 0;
  qword_4FA1368 = (__int64)qword_4FA01C0;
  qword_4FA1378 = (__int64)&unk_4FA1398;
  qword_4FA1380 = (__int64)&unk_4FA1398;
  qword_4FA1360 = 0;
  qword_4FA1370 = 0;
  word_4FA13D0 = 256;
  qword_4FA13C8 = (__int64)&unk_49E74E8;
  qword_4FA1388 = 4;
  qword_4FA1320 = (__int64)&unk_49EEC70;
  byte_4FA13B8 = 0;
  qword_4FA13D8 = (__int64)&unk_49EEDB0;
  dword_4FA1390 = 0;
  byte_4FA13C0 = 0;
  sub_16B8280(&qword_4FA1320, "stats", 5);
  qword_4FA1350 = 62;
  qword_4FA1348 = (__int64)"Enable statistics output from program (available with Asserts)";
  LOBYTE(word_4FA132C) = word_4FA132C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FA1320);
  __cxa_atexit(sub_12EDEC0, &qword_4FA1320, &qword_4A427C0);
  qword_4FA1240 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA124C &= 0xF000u;
  word_4FA12F0 = 256;
  qword_4FA1250 = 0;
  qword_4FA1258 = 0;
  qword_4FA1260 = 0;
  qword_4FA1268 = 0;
  dword_4FA1248 = v1;
  qword_4FA12E8 = (__int64)&unk_49E74E8;
  qword_4FA1288 = (__int64)qword_4FA01C0;
  qword_4FA1298 = (__int64)&unk_4FA12B8;
  qword_4FA12A0 = (__int64)&unk_4FA12B8;
  qword_4FA1240 = (__int64)&unk_49EEC70;
  qword_4FA12F8 = (__int64)&unk_49EEDB0;
  qword_4FA1270 = 0;
  qword_4FA1278 = 0;
  qword_4FA1280 = 0;
  qword_4FA1290 = 0;
  qword_4FA12A8 = 4;
  dword_4FA12B0 = 0;
  byte_4FA12D8 = 0;
  byte_4FA12E0 = 0;
  sub_16B8280(&qword_4FA1240, "stats-json", 10);
  qword_4FA1270 = 31;
  qword_4FA1268 = (__int64)"Display statistics as json data";
  LOBYTE(word_4FA124C) = word_4FA124C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FA1240);
  return __cxa_atexit(sub_12EDEC0, &qword_4FA1240, &qword_4A427C0);
}
