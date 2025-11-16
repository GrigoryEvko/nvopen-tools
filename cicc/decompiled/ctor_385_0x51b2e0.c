// Function: ctor_385
// Address: 0x51b2e0
//
int ctor_385()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4FDC160 = (__int64)&qword_4FDC190;
  qword_4FDC168 = 1;
  qword_4FDC170 = 0;
  qword_4FDC178 = 0;
  dword_4FDC180 = 1065353216;
  qword_4FDC188 = 0;
  qword_4FDC190 = 0;
  __cxa_atexit(sub_8565C0, &qword_4FDC190 - 6, &qword_4A427C0);
  qword_4FDC080 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FDC0FC = 1;
  qword_4FDC0D0 = 0x100000000LL;
  dword_4FDC08C &= 0x8000u;
  qword_4FDC098 = 0;
  qword_4FDC0A0 = 0;
  qword_4FDC0A8 = 0;
  dword_4FDC088 = v0;
  word_4FDC090 = 0;
  qword_4FDC0B0 = 0;
  qword_4FDC0B8 = 0;
  qword_4FDC0C0 = 0;
  qword_4FDC0C8 = (__int64)&unk_4FDC0D8;
  qword_4FDC0E0 = 0;
  qword_4FDC0E8 = (__int64)&unk_4FDC100;
  qword_4FDC0F0 = 1;
  dword_4FDC0F8 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FDC0D0;
  v3 = (unsigned int)qword_4FDC0D0 + 1LL;
  if ( v3 > HIDWORD(qword_4FDC0D0) )
  {
    sub_C8D5F0((char *)&unk_4FDC0D8 - 16, &unk_4FDC0D8, v3, 8);
    v2 = (unsigned int)qword_4FDC0D0;
  }
  *(_QWORD *)(qword_4FDC0C8 + 8 * v2) = v1;
  LODWORD(qword_4FDC0D0) = qword_4FDC0D0 + 1;
  qword_4FDC108 = 0;
  qword_4FDC110 = (__int64)&unk_49D9748;
  qword_4FDC118 = 0;
  qword_4FDC080 = (__int64)&unk_49DC090;
  qword_4FDC120 = (__int64)&unk_49DC1D0;
  qword_4FDC140 = (__int64)nullsub_23;
  qword_4FDC138 = (__int64)sub_984030;
  sub_C53080(&qword_4FDC080, "only-simple-regions", 19);
  qword_4FDC0B0 = 47;
  qword_4FDC0A8 = (__int64)"Show only simple regions in the graphviz viewer";
  LOBYTE(qword_4FDC108) = 0;
  LOBYTE(dword_4FDC08C) = dword_4FDC08C & 0x9F | 0x20;
  LOWORD(qword_4FDC118) = 256;
  sub_C53130(&qword_4FDC080);
  return __cxa_atexit(sub_984900, &qword_4FDC080, &qword_4A427C0);
}
