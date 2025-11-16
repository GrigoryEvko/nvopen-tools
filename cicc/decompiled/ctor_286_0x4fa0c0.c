// Function: ctor_286
// Address: 0x4fa0c0
//
int ctor_286()
{
  int v0; // eax
  int v1; // eax
  int v2; // eax
  int v3; // eax
  int v5; // [rsp+1Ch] [rbp-54h] BYREF
  __int64 (__fastcall **v6)(_QWORD, _QWORD); // [rsp+20h] [rbp-50h] BYREF
  __int64 (__fastcall *v7)(_QWORD, _QWORD); // [rsp+28h] [rbp-48h] BYREF
  _QWORD v8[8]; // [rsp+30h] [rbp-40h] BYREF

  qword_4FC1DE0 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC1DEC &= 0xF000u;
  qword_4FC1DF0 = 0;
  qword_4FC1DF8 = 0;
  qword_4FC1E00 = 0;
  qword_4FC1E08 = 0;
  qword_4FC1E10 = 0;
  dword_4FC1DE8 = v0;
  qword_4FC1E18 = 0;
  qword_4FC1E28 = (__int64)qword_4FA01C0;
  qword_4FC1E38 = (__int64)&unk_4FC1E58;
  qword_4FC1E40 = (__int64)&unk_4FC1E58;
  qword_4FC1E20 = 0;
  qword_4FC1E30 = 0;
  qword_4FC1E88 = (__int64)&unk_49E74C8;
  qword_4FC1E48 = 4;
  dword_4FC1E50 = 0;
  qword_4FC1DE0 = (__int64)&unk_49EEB70;
  byte_4FC1E78 = 0;
  dword_4FC1E80 = 0;
  qword_4FC1E98 = (__int64)&unk_49EEDF0;
  byte_4FC1E94 = 1;
  dword_4FC1E90 = 0;
  sub_16B8280(&qword_4FC1DE0, "fast-isel-abort", 15);
  qword_4FC1E10 = 238;
  LOBYTE(word_4FC1DEC) = word_4FC1DEC & 0x9F | 0x20;
  qword_4FC1E08 = (__int64)"Enable abort calls when \"fast\" instruction selection fails to lower an instruction: 0 disab"
                           "le the abort, 1 will abort but for args, calls and terminators, 2 will also abort for argumen"
                           "t lowering, and 3 will never fallback to SelectionDAG.";
  sub_16B88A0(&qword_4FC1DE0);
  __cxa_atexit(sub_12EDEA0, &qword_4FC1DE0, &qword_4A427C0);
  qword_4FC1D00 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC1D0C &= 0xF000u;
  qword_4FC1D10 = 0;
  qword_4FC1D18 = 0;
  qword_4FC1D20 = 0;
  qword_4FC1D28 = 0;
  qword_4FC1D30 = 0;
  dword_4FC1D08 = v1;
  qword_4FC1D38 = 0;
  qword_4FC1D48 = (__int64)qword_4FA01C0;
  qword_4FC1D58 = (__int64)&unk_4FC1D78;
  qword_4FC1D60 = (__int64)&unk_4FC1D78;
  qword_4FC1D40 = 0;
  qword_4FC1D50 = 0;
  word_4FC1DB0 = 256;
  qword_4FC1DA8 = (__int64)&unk_49E74E8;
  qword_4FC1D00 = (__int64)&unk_49EEC70;
  byte_4FC1D98 = 0;
  qword_4FC1DB8 = (__int64)&unk_49EEDB0;
  qword_4FC1D68 = 4;
  dword_4FC1D70 = 0;
  byte_4FC1DA0 = 0;
  sub_16B8280(&qword_4FC1D00, "fast-isel-report-on-fallback", 28);
  qword_4FC1D30 = 79;
  LOBYTE(word_4FC1D0C) = word_4FC1D0C & 0x9F | 0x20;
  qword_4FC1D28 = (__int64)"Emit a diagnostic when \"fast\" instruction selection falls back to SelectionDAG.";
  sub_16B88A0(&qword_4FC1D00);
  __cxa_atexit(sub_12EDEC0, &qword_4FC1D00, &qword_4A427C0);
  qword_4FC1C20 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC1C2C &= 0xF000u;
  word_4FC1CD0 = 256;
  qword_4FC1C30 = 0;
  qword_4FC1C38 = 0;
  qword_4FC1CC8 = (__int64)&unk_49E74E8;
  qword_4FC1C40 = 0;
  dword_4FC1C28 = v2;
  qword_4FC1C20 = (__int64)&unk_49EEC70;
  qword_4FC1C68 = (__int64)qword_4FA01C0;
  qword_4FC1C78 = (__int64)&unk_4FC1C98;
  qword_4FC1C80 = (__int64)&unk_4FC1C98;
  qword_4FC1CD8 = (__int64)&unk_49EEDB0;
  qword_4FC1C48 = 0;
  qword_4FC1C50 = 0;
  qword_4FC1C58 = 0;
  qword_4FC1C60 = 0;
  qword_4FC1C70 = 0;
  qword_4FC1C88 = 4;
  dword_4FC1C90 = 0;
  byte_4FC1CB8 = 0;
  byte_4FC1CC0 = 0;
  sub_16B8280(&qword_4FC1C20, "use-mbpi", 8);
  qword_4FC1C48 = (__int64)"use Machine Branch Probability Info";
  word_4FC1CD0 = 257;
  byte_4FC1CC0 = 1;
  qword_4FC1C50 = 35;
  LOBYTE(word_4FC1C2C) = word_4FC1C2C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FC1C20);
  __cxa_atexit(sub_12EDEC0, &qword_4FC1C20, &qword_4A427C0);
  qword_4FC1B40 = (__int64)&unk_49EED30;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FC1B4C &= 0xF000u;
  word_4FC1BF0 = 256;
  qword_4FC1B50 = 0;
  qword_4FC1B58 = 0;
  qword_4FC1BE8 = (__int64)&unk_49E74E8;
  qword_4FC1B60 = 0;
  dword_4FC1B48 = v3;
  qword_4FC1B40 = (__int64)&unk_49EEC70;
  qword_4FC1B88 = (__int64)qword_4FA01C0;
  qword_4FC1B98 = (__int64)&unk_4FC1BB8;
  qword_4FC1BA0 = (__int64)&unk_4FC1BB8;
  qword_4FC1BF8 = (__int64)&unk_49EEDB0;
  qword_4FC1B68 = 0;
  qword_4FC1B70 = 0;
  qword_4FC1B78 = 0;
  qword_4FC1B80 = 0;
  qword_4FC1B90 = 0;
  qword_4FC1BA8 = 4;
  dword_4FC1BB0 = 0;
  byte_4FC1BD8 = 0;
  byte_4FC1BE0 = 0;
  sub_16B8280(&qword_4FC1B40, "dag-disable-combine", 19);
  qword_4FC1B70 = 35;
  word_4FC1BF0 = 256;
  byte_4FC1BE0 = 0;
  LOBYTE(word_4FC1B4C) = word_4FC1B4C & 0x9F | 0x20;
  qword_4FC1B68 = (__int64)"Disable DAG Combining optimizations";
  sub_16B88A0(&qword_4FC1B40);
  __cxa_atexit(sub_12EDEC0, &qword_4FC1B40, &qword_4A427C0);
  v8[0] = "Instruction schedulers available (before register allocation):";
  v7 = sub_1D469E0;
  v6 = &v7;
  v8[1] = 62;
  v5 = 1;
  sub_1D52180(&unk_4FC1860, "pre-RA-sched", &v6, &v5, v8);
  __cxa_atexit(sub_1D47060, &unk_4FC1860, &qword_4A427C0);
  qword_4FC1828 = (__int64)"default";
  qword_4FC1848 = (__int64)sub_1D469E0;
  qword_4FC1820 = 0;
  qword_4FC1830 = 7;
  qword_4FC1838 = (__int64)"Best scheduler for the target";
  qword_4FC1840 = 29;
  sub_1E40390(&unk_4FC1B10, &qword_4FC1820);
  return __cxa_atexit(sub_1CFC0C0, &qword_4FC1820, &qword_4A427C0);
}
