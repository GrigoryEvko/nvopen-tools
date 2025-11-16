// Function: ctor_687
// Address: 0x5a6280
//
int ctor_687()
{
  __int64 v0; // rdx
  __int64 v1; // rcx
  int v2; // edx
  __int64 v3; // rbx
  __int64 v4; // rax
  unsigned __int64 v5; // rdx

  qword_503FF08 = (__int64)"pbqp";
  qword_503FF10 = 4;
  qword_503FF18 = (__int64)"PBQP register allocator";
  qword_503FF20 = 23;
  qword_503FF28 = (__int64)sub_35B9BC0;
  qword_503FF00 = unk_5023860;
  unk_5023860 = &qword_503FF00;
  if ( qword_5023870 )
    (*(void (__fastcall **)(_QWORD *, const char *, __int64, __int64 (*)(void), const char *, __int64))(*qword_5023870 + 24LL))(
      qword_5023870,
      "pbqp",
      4,
      sub_35B9BC0,
      "PBQP register allocator",
      23);
  __cxa_atexit(sub_2F41140, &qword_503FF00, &qword_4A427C0);
  qword_503FE20 = (__int64)&unk_49DC150;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_2F41140, &qword_503FF00, v0, v1), 1u);
  byte_503FE9C = 1;
  qword_503FE70 = 0x100000000LL;
  dword_503FE2C &= 0x8000u;
  qword_503FE38 = 0;
  qword_503FE40 = 0;
  qword_503FE48 = 0;
  dword_503FE28 = v2;
  word_503FE30 = 0;
  qword_503FE50 = 0;
  qword_503FE58 = 0;
  qword_503FE60 = 0;
  qword_503FE68 = (__int64)&unk_503FE78;
  qword_503FE80 = 0;
  qword_503FE88 = (__int64)&unk_503FEA0;
  qword_503FE90 = 1;
  dword_503FE98 = 0;
  v3 = sub_C57470();
  v4 = (unsigned int)qword_503FE70;
  v5 = (unsigned int)qword_503FE70 + 1LL;
  if ( v5 > HIDWORD(qword_503FE70) )
  {
    sub_C8D5F0((char *)&unk_503FE78 - 16, &unk_503FE78, v5, 8);
    v4 = (unsigned int)qword_503FE70;
  }
  *(_QWORD *)(qword_503FE68 + 8 * v4) = v3;
  LODWORD(qword_503FE70) = qword_503FE70 + 1;
  qword_503FEA8 = 0;
  qword_503FEB0 = (__int64)&unk_49D9748;
  qword_503FEB8 = 0;
  qword_503FE20 = (__int64)&unk_49DC090;
  qword_503FEC0 = (__int64)&unk_49DC1D0;
  qword_503FEE0 = (__int64)nullsub_23;
  qword_503FED8 = (__int64)sub_984030;
  sub_C53080(&qword_503FE20, "pbqp-coalescing", 15);
  qword_503FE50 = 51;
  qword_503FE48 = (__int64)"Attempt coalescing during PBQP register allocation.";
  LOWORD(qword_503FEB8) = 256;
  LOBYTE(qword_503FEA8) = 0;
  LOBYTE(dword_503FE2C) = dword_503FE2C & 0x9F | 0x20;
  sub_C53130(&qword_503FE20);
  return __cxa_atexit(sub_984900, &qword_503FE20, &qword_4A427C0);
}
