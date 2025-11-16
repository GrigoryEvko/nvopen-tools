// Function: ctor_177
// Address: 0x4d6500
//
int ctor_177()
{
  int v0; // eax
  int v1; // eax
  int v2; // eax
  char v4; // [rsp+13h] [rbp-4Dh] BYREF
  int v5; // [rsp+14h] [rbp-4Ch] BYREF
  char *v6; // [rsp+18h] [rbp-48h] BYREF
  const char *v7; // [rsp+20h] [rbp-40h] BYREF
  __int64 v8; // [rsp+28h] [rbp-38h]

  v7 = "Instrument memory accesses";
  v5 = 1;
  v8 = 26;
  v4 = 1;
  v6 = &v4;
  sub_17FC400(&unk_4FA6B00, "tsan-instrument-memory-accesses", &v6, &v7, &v5);
  __cxa_atexit(sub_12EDEC0, &unk_4FA6B00, &qword_4A427C0);
  v5 = 1;
  v7 = "Instrument function entry and exit";
  v8 = 34;
  v4 = 1;
  v6 = &v4;
  sub_17FC400(&unk_4FA6A20, "tsan-instrument-func-entry-exit", &v6, &v7, &v5);
  __cxa_atexit(sub_12EDEC0, &unk_4FA6A20, &qword_4A427C0);
  qword_4FA6940 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA694C &= 0xF000u;
  qword_4FA6950 = 0;
  qword_4FA6958 = 0;
  qword_4FA6960 = 0;
  qword_4FA6968 = 0;
  qword_4FA6970 = 0;
  dword_4FA6948 = v0;
  qword_4FA6978 = 0;
  qword_4FA6988 = (__int64)qword_4FA01C0;
  qword_4FA6998 = (__int64)&unk_4FA69B8;
  qword_4FA69A0 = (__int64)&unk_4FA69B8;
  qword_4FA6980 = 0;
  qword_4FA6990 = 0;
  word_4FA69F0 = 256;
  qword_4FA69E8 = (__int64)&unk_49E74E8;
  qword_4FA69A8 = 4;
  qword_4FA6940 = (__int64)&unk_49EEC70;
  byte_4FA69D8 = 0;
  qword_4FA69F8 = (__int64)&unk_49EEDB0;
  dword_4FA69B0 = 0;
  byte_4FA69E0 = 0;
  sub_16B8280(&qword_4FA6940, "tsan-handle-cxx-exceptions", 26);
  qword_4FA6968 = (__int64)"Handle C++ exceptions (insert cleanup blocks for unwinding)";
  word_4FA69F0 = 257;
  byte_4FA69E0 = 1;
  qword_4FA6970 = 59;
  LOBYTE(word_4FA694C) = word_4FA694C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FA6940);
  __cxa_atexit(sub_12EDEC0, &qword_4FA6940, &qword_4A427C0);
  qword_4FA6860 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA6910 = 256;
  word_4FA686C &= 0xF000u;
  qword_4FA6870 = 0;
  qword_4FA6878 = 0;
  qword_4FA6880 = 0;
  dword_4FA6868 = v1;
  qword_4FA6908 = (__int64)&unk_49E74E8;
  qword_4FA68A8 = (__int64)qword_4FA01C0;
  qword_4FA68B8 = (__int64)&unk_4FA68D8;
  qword_4FA68C0 = (__int64)&unk_4FA68D8;
  qword_4FA6860 = (__int64)&unk_49EEC70;
  qword_4FA6918 = (__int64)&unk_49EEDB0;
  qword_4FA6888 = 0;
  qword_4FA6890 = 0;
  qword_4FA6898 = 0;
  qword_4FA68A0 = 0;
  qword_4FA68B0 = 0;
  qword_4FA68C8 = 4;
  dword_4FA68D0 = 0;
  byte_4FA68F8 = 0;
  byte_4FA6900 = 0;
  sub_16B8280(&qword_4FA6860, "tsan-instrument-atomics", 23);
  qword_4FA6888 = (__int64)"Instrument atomics";
  word_4FA6910 = 257;
  byte_4FA6900 = 1;
  qword_4FA6890 = 18;
  LOBYTE(word_4FA686C) = word_4FA686C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FA6860);
  __cxa_atexit(sub_12EDEC0, &qword_4FA6860, &qword_4A427C0);
  qword_4FA6780 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_4FA6830 = 256;
  word_4FA678C &= 0xF000u;
  qword_4FA6790 = 0;
  qword_4FA6798 = 0;
  qword_4FA67A0 = 0;
  dword_4FA6788 = v2;
  qword_4FA6828 = (__int64)&unk_49E74E8;
  qword_4FA67C8 = (__int64)qword_4FA01C0;
  qword_4FA67D8 = (__int64)&unk_4FA67F8;
  qword_4FA67E0 = (__int64)&unk_4FA67F8;
  qword_4FA6780 = (__int64)&unk_49EEC70;
  qword_4FA6838 = (__int64)&unk_49EEDB0;
  qword_4FA67A8 = 0;
  qword_4FA67B0 = 0;
  qword_4FA67B8 = 0;
  qword_4FA67C0 = 0;
  qword_4FA67D0 = 0;
  qword_4FA67E8 = 4;
  dword_4FA67F0 = 0;
  byte_4FA6818 = 0;
  byte_4FA6820 = 0;
  sub_16B8280(&qword_4FA6780, "tsan-instrument-memintrinsics", 29);
  qword_4FA67A8 = (__int64)"Instrument memintrinsics (memset/memcpy/memmove)";
  byte_4FA6820 = 1;
  word_4FA6830 = 257;
  qword_4FA67B0 = 48;
  LOBYTE(word_4FA678C) = word_4FA678C & 0x9F | 0x20;
  sub_16B88A0(&qword_4FA6780);
  return __cxa_atexit(sub_12EDEC0, &qword_4FA6780, &qword_4A427C0);
}
