// Function: ctor_238
// Address: 0x4ec670
//
int ctor_238()
{
  int v0; // eax
  int v1; // eax

  qword_4FB6980 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB698C &= 0xF000u;
  qword_4FB6990 = 0;
  qword_4FB6998 = 0;
  qword_4FB69A0 = 0;
  qword_4FB69A8 = 0;
  qword_4FB69B0 = 0;
  dword_4FB6988 = v0;
  qword_4FB69B8 = 0;
  qword_4FB69C8 = (__int64)qword_4FA01C0;
  qword_4FB69D8 = (__int64)&unk_4FB69F8;
  qword_4FB69E0 = (__int64)&unk_4FB69F8;
  qword_4FB69C0 = 0;
  qword_4FB69D0 = 0;
  qword_4FB6A28 = (__int64)&unk_49E74A8;
  qword_4FB69E8 = 4;
  qword_4FB6980 = (__int64)&unk_49EEAF0;
  dword_4FB69F0 = 0;
  qword_4FB6A38 = (__int64)&unk_49EEE10;
  byte_4FB6A18 = 0;
  dword_4FB6A20 = 0;
  byte_4FB6A34 = 1;
  dword_4FB6A30 = 0;
  sub_16B8280(&qword_4FB6980, "unroll-peel-max-count", 21);
  dword_4FB6A20 = 7;
  byte_4FB6A34 = 1;
  dword_4FB6A30 = 7;
  qword_4FB69B0 = 53;
  LOBYTE(word_4FB698C) = word_4FB698C & 0x9F | 0x20;
  qword_4FB69A8 = (__int64)"Max average trip count which will cause loop peeling.";
  sub_16B88A0(&qword_4FB6980);
  __cxa_atexit(sub_12EDE60, &qword_4FB6980, &qword_4A427C0);
  qword_4FB68A0 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FB68AC &= 0xF000u;
  qword_4FB68B0 = 0;
  qword_4FB68B8 = 0;
  qword_4FB68C0 = 0;
  qword_4FB68C8 = 0;
  qword_4FB68D0 = 0;
  dword_4FB68A8 = v1;
  qword_4FB6948 = (__int64)&unk_49E74A8;
  qword_4FB68E8 = (__int64)qword_4FA01C0;
  qword_4FB68F8 = (__int64)&unk_4FB6918;
  qword_4FB6900 = (__int64)&unk_4FB6918;
  qword_4FB68A0 = (__int64)&unk_49EEAF0;
  qword_4FB6958 = (__int64)&unk_49EEE10;
  qword_4FB68D8 = 0;
  qword_4FB68E0 = 0;
  qword_4FB68F0 = 0;
  qword_4FB6908 = 4;
  dword_4FB6910 = 0;
  byte_4FB6938 = 0;
  dword_4FB6940 = 0;
  byte_4FB6954 = 1;
  dword_4FB6950 = 0;
  sub_16B8280(&qword_4FB68A0, "unroll-force-peel-count", 23);
  dword_4FB6940 = 0;
  byte_4FB6954 = 1;
  dword_4FB6950 = 0;
  qword_4FB68D0 = 55;
  LOBYTE(word_4FB68AC) = word_4FB68AC & 0x9F | 0x20;
  qword_4FB68C8 = (__int64)"Force a peel count regardless of profiling information.";
  sub_16B88A0(&qword_4FB68A0);
  return __cxa_atexit(sub_12EDE60, &qword_4FB68A0, &qword_4A427C0);
}
