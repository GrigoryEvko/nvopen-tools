// Function: ctor_695
// Address: 0x5a7bf0
//
int __fastcall ctor_695(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // r15
  __int64 v12; // rax
  unsigned __int64 v13; // rdx

  qword_5040EC0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_5040F10 = 0x100000000LL;
  dword_5040ECC &= 0x8000u;
  word_5040ED0 = 0;
  qword_5040ED8 = 0;
  qword_5040EE0 = 0;
  dword_5040EC8 = v4;
  qword_5040EE8 = 0;
  qword_5040EF0 = 0;
  qword_5040EF8 = 0;
  qword_5040F00 = 0;
  qword_5040F08 = (__int64)&unk_5040F18;
  qword_5040F20 = 0;
  qword_5040F28 = (__int64)&unk_5040F40;
  qword_5040F30 = 1;
  dword_5040F38 = 0;
  byte_5040F3C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5040F10;
  v7 = (unsigned int)qword_5040F10 + 1LL;
  if ( v7 > HIDWORD(qword_5040F10) )
  {
    sub_C8D5F0((char *)&unk_5040F18 - 16, &unk_5040F18, v7, 8);
    v6 = (unsigned int)qword_5040F10;
  }
  *(_QWORD *)(qword_5040F08 + 8 * v6) = v5;
  qword_5040F50 = (__int64)&unk_49D9748;
  LODWORD(qword_5040F10) = qword_5040F10 + 1;
  qword_5040F48 = 0;
  qword_5040EC0 = (__int64)&unk_49DC090;
  qword_5040F60 = (__int64)&unk_49DC1D0;
  qword_5040F58 = 0;
  qword_5040F80 = (__int64)nullsub_23;
  qword_5040F78 = (__int64)sub_984030;
  sub_C53080(&qword_5040EC0, "disable-rsqrt-opt", 17);
  LOWORD(qword_5040F58) = 256;
  LOBYTE(qword_5040F48) = 0;
  qword_5040EF0 = 38;
  LOBYTE(dword_5040ECC) = dword_5040ECC & 0x9F | 0x20;
  qword_5040EE8 = (__int64)"Disable reciprocal sqrt optimization. ";
  sub_C53130(&qword_5040EC0);
  __cxa_atexit(sub_984900, &qword_5040EC0, &qword_4A427C0);
  qword_5040DE0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5040EC0, v8, v9), 1u);
  qword_5040E30 = 0x100000000LL;
  word_5040DF0 = 0;
  dword_5040DEC &= 0x8000u;
  qword_5040DF8 = 0;
  qword_5040E00 = 0;
  dword_5040DE8 = v10;
  qword_5040E08 = 0;
  qword_5040E10 = 0;
  qword_5040E18 = 0;
  qword_5040E20 = 0;
  qword_5040E28 = (__int64)&unk_5040E38;
  qword_5040E40 = 0;
  qword_5040E48 = (__int64)&unk_5040E60;
  qword_5040E50 = 1;
  dword_5040E58 = 0;
  byte_5040E5C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5040E30;
  v13 = (unsigned int)qword_5040E30 + 1LL;
  if ( v13 > HIDWORD(qword_5040E30) )
  {
    sub_C8D5F0((char *)&unk_5040E38 - 16, &unk_5040E38, v13, 8);
    v12 = (unsigned int)qword_5040E30;
  }
  *(_QWORD *)(qword_5040E28 + 8 * v12) = v11;
  qword_5040E70 = (__int64)&unk_49D9748;
  LODWORD(qword_5040E30) = qword_5040E30 + 1;
  qword_5040E68 = 0;
  qword_5040DE0 = (__int64)&unk_49DC090;
  qword_5040E80 = (__int64)&unk_49DC1D0;
  qword_5040E78 = 0;
  qword_5040EA0 = (__int64)nullsub_23;
  qword_5040E98 = (__int64)sub_984030;
  sub_C53080(&qword_5040DE0, "nvptx-rsqrt-approx-opt", 22);
  LOBYTE(qword_5040E68) = 1;
  LOWORD(qword_5040E78) = 257;
  qword_5040E10 = 35;
  LOBYTE(dword_5040DEC) = dword_5040DEC & 0x9F | 0x20;
  qword_5040E08 = (__int64)"Enable reciprocal sqrt optimization";
  sub_C53130(&qword_5040DE0);
  return __cxa_atexit(sub_984900, &qword_5040DE0, &qword_4A427C0);
}
