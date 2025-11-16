// Function: ctor_659
// Address: 0x59c7c0
//
int __fastcall ctor_659(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rcx
  int v15; // edx
  __int64 v16; // rbx
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  __int64 v20; // [rsp+8h] [rbp-38h]

  qword_503A020 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_503A070 = 0x100000000LL;
  dword_503A02C &= 0x8000u;
  word_503A030 = 0;
  qword_503A038 = 0;
  qword_503A040 = 0;
  dword_503A028 = v4;
  qword_503A048 = 0;
  qword_503A050 = 0;
  qword_503A058 = 0;
  qword_503A060 = 0;
  qword_503A068 = (__int64)&unk_503A078;
  qword_503A080 = 0;
  qword_503A088 = (__int64)&unk_503A0A0;
  qword_503A090 = 1;
  dword_503A098 = 0;
  byte_503A09C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_503A070;
  v7 = (unsigned int)qword_503A070 + 1LL;
  if ( v7 > HIDWORD(qword_503A070) )
  {
    sub_C8D5F0((char *)&unk_503A078 - 16, &unk_503A078, v7, 8);
    v6 = (unsigned int)qword_503A070;
  }
  *(_QWORD *)(qword_503A068 + 8 * v6) = v5;
  qword_503A0B0 = (__int64)&unk_49D9748;
  qword_503A020 = (__int64)&unk_49DC090;
  LODWORD(qword_503A070) = qword_503A070 + 1;
  qword_503A0A8 = 0;
  qword_503A0C0 = (__int64)&unk_49DC1D0;
  qword_503A0B8 = 0;
  qword_503A0E0 = (__int64)nullsub_23;
  qword_503A0D8 = (__int64)sub_984030;
  sub_C53080(&qword_503A020, "use-registers-for-deopt-values", 30);
  LOWORD(qword_503A0B8) = 256;
  LOBYTE(qword_503A0A8) = 0;
  qword_503A050 = 48;
  LOBYTE(dword_503A02C) = dword_503A02C & 0x9F | 0x20;
  qword_503A048 = (__int64)"Allow using registers for non pointer deopt args";
  sub_C53130(&qword_503A020);
  __cxa_atexit(sub_984900, &qword_503A020, &qword_4A427C0);
  qword_5039F40 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_503A020, v8, v9), 1u);
  qword_5039F90 = 0x100000000LL;
  dword_5039F4C &= 0x8000u;
  word_5039F50 = 0;
  qword_5039F58 = 0;
  qword_5039F60 = 0;
  dword_5039F48 = v10;
  qword_5039F68 = 0;
  qword_5039F70 = 0;
  qword_5039F78 = 0;
  qword_5039F80 = 0;
  qword_5039F88 = (__int64)&unk_5039F98;
  qword_5039FA0 = 0;
  qword_5039FA8 = (__int64)&unk_5039FC0;
  qword_5039FB0 = 1;
  dword_5039FB8 = 0;
  byte_5039FBC = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5039F90;
  if ( (unsigned __int64)(unsigned int)qword_5039F90 + 1 > HIDWORD(qword_5039F90) )
  {
    v20 = v11;
    sub_C8D5F0((char *)&unk_5039F98 - 16, &unk_5039F98, (unsigned int)qword_5039F90 + 1LL, 8);
    v12 = (unsigned int)qword_5039F90;
    v11 = v20;
  }
  *(_QWORD *)(qword_5039F88 + 8 * v12) = v11;
  qword_5039FD0 = (__int64)&unk_49D9748;
  qword_5039F40 = (__int64)&unk_49DC090;
  LODWORD(qword_5039F90) = qword_5039F90 + 1;
  qword_5039FC8 = 0;
  qword_5039FE0 = (__int64)&unk_49DC1D0;
  qword_5039FD8 = 0;
  qword_503A000 = (__int64)nullsub_23;
  qword_5039FF8 = (__int64)sub_984030;
  sub_C53080(&qword_5039F40, "use-registers-for-gc-values-in-landing-pad", 42);
  LOBYTE(qword_5039FC8) = 0;
  qword_5039F70 = 51;
  LOBYTE(dword_5039F4C) = dword_5039F4C & 0x9F | 0x20;
  LOWORD(qword_5039FD8) = 256;
  qword_5039F68 = (__int64)"Allow using registers for gc pointer in landing pad";
  sub_C53130(&qword_5039F40);
  __cxa_atexit(sub_984900, &qword_5039F40, &qword_4A427C0);
  qword_5039E60 = (__int64)&unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5039F40, v13, v14), 1u);
  byte_5039EDC = 1;
  qword_5039EB0 = 0x100000000LL;
  dword_5039E6C &= 0x8000u;
  qword_5039E78 = 0;
  qword_5039E80 = 0;
  qword_5039E88 = 0;
  dword_5039E68 = v15;
  word_5039E70 = 0;
  qword_5039E90 = 0;
  qword_5039E98 = 0;
  qword_5039EA0 = 0;
  qword_5039EA8 = (__int64)&unk_5039EB8;
  qword_5039EC0 = 0;
  qword_5039EC8 = (__int64)&unk_5039EE0;
  qword_5039ED0 = 1;
  dword_5039ED8 = 0;
  v16 = sub_C57470();
  v17 = (unsigned int)qword_5039EB0;
  v18 = (unsigned int)qword_5039EB0 + 1LL;
  if ( v18 > HIDWORD(qword_5039EB0) )
  {
    sub_C8D5F0((char *)&unk_5039EB8 - 16, &unk_5039EB8, v18, 8);
    v17 = (unsigned int)qword_5039EB0;
  }
  *(_QWORD *)(qword_5039EA8 + 8 * v17) = v16;
  LODWORD(qword_5039EB0) = qword_5039EB0 + 1;
  qword_5039EE8 = 0;
  qword_5039EF0 = (__int64)&unk_49D9728;
  qword_5039EF8 = 0;
  qword_5039E60 = (__int64)&unk_49DBF10;
  qword_5039F00 = (__int64)&unk_49DC290;
  qword_5039F20 = (__int64)nullsub_24;
  qword_5039F18 = (__int64)sub_984050;
  sub_C53080(&qword_5039E60, "max-registers-for-gc-values", 27);
  LODWORD(qword_5039EE8) = 0;
  BYTE4(qword_5039EF8) = 1;
  LODWORD(qword_5039EF8) = 0;
  qword_5039E90 = 59;
  LOBYTE(dword_5039E6C) = dword_5039E6C & 0x9F | 0x20;
  qword_5039E88 = (__int64)"Max number of VRegs allowed to pass GC pointer meta args in";
  sub_C53130(&qword_5039E60);
  return __cxa_atexit(sub_984970, &qword_5039E60, &qword_4A427C0);
}
