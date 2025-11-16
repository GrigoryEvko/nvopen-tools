// Function: ctor_255
// Address: 0x4f0c10
//
int ctor_255()
{
  int v0; // edx

  sub_2208040(&unk_4FBA448);
  __cxa_atexit(sub_2208810, &unk_4FBA448, &qword_4A427C0);
  qword_4FBA380[0] = &unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  WORD2(qword_4FBA380[1]) &= 0xF000u;
  qword_4FBA380[9] = qword_4FA01C0;
  qword_4FBA380[11] = &qword_4FBA380[15];
  qword_4FBA380[12] = &qword_4FBA380[15];
  LODWORD(qword_4FBA380[1]) = v0;
  qword_4FBA380[2] = 0;
  qword_4FBA380[21] = &unk_49E74E8;
  LOWORD(qword_4FBA380[22]) = 256;
  qword_4FBA380[3] = 0;
  qword_4FBA380[4] = 0;
  qword_4FBA380[0] = &unk_49EEC70;
  qword_4FBA380[5] = 0;
  qword_4FBA380[6] = 0;
  qword_4FBA380[23] = &unk_49EEDB0;
  qword_4FBA380[7] = 0;
  qword_4FBA380[8] = 0;
  qword_4FBA380[10] = 0;
  qword_4FBA380[13] = 4;
  LODWORD(qword_4FBA380[14]) = 0;
  LOBYTE(qword_4FBA380[19]) = 0;
  LOBYTE(qword_4FBA380[20]) = 0;
  sub_16B8280(qword_4FBA380, "disable-attrib-transplant", 25);
  LOWORD(qword_4FBA380[22]) = 256;
  LOBYTE(qword_4FBA380[20]) = 0;
  qword_4FBA380[5] = "Do not transplant metadata onto functions";
  qword_4FBA380[6] = 41;
  sub_16B88A0(qword_4FBA380);
  return __cxa_atexit(sub_12EDEC0, qword_4FBA380, &qword_4A427C0);
}
