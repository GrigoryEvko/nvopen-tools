// Function: ctor_108
// Address: 0x4aa430
//
int ctor_108()
{
  int v0; // edx

  sub_2208040(&unk_4F968E8);
  __cxa_atexit(sub_2208810, &unk_4F968E8, &qword_4A427C0);
  qword_4F96820[0] = &unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  WORD2(qword_4F96820[1]) &= 0xF000u;
  qword_4F96820[9] = &unk_4FA01C0;
  qword_4F96820[11] = &qword_4F96820[15];
  qword_4F96820[12] = &qword_4F96820[15];
  LODWORD(qword_4F96820[1]) = v0;
  qword_4F96820[2] = 0;
  qword_4F96820[21] = &unk_49E74E8;
  LOWORD(qword_4F96820[22]) = 256;
  qword_4F96820[3] = 0;
  qword_4F96820[4] = 0;
  qword_4F96820[0] = &unk_49EEC70;
  qword_4F96820[5] = 0;
  qword_4F96820[6] = 0;
  qword_4F96820[23] = &unk_49EEDB0;
  qword_4F96820[7] = 0;
  qword_4F96820[8] = 0;
  qword_4F96820[10] = 0;
  qword_4F96820[13] = 4;
  LODWORD(qword_4F96820[14]) = 0;
  LOBYTE(qword_4F96820[19]) = 0;
  LOBYTE(qword_4F96820[20]) = 0;
  sub_16B8280(qword_4F96820, "lnk-disable-allopts", 19);
  qword_4F96820[6] = 35;
  BYTE4(qword_4F96820[1]) = BYTE4(qword_4F96820[1]) & 0x9F | 0x20;
  qword_4F96820[5] = "Disable all lnk Optimization passes";
  sub_16B88A0(qword_4F96820);
  return __cxa_atexit(sub_12EDEC0, qword_4F96820, &qword_4A427C0);
}
