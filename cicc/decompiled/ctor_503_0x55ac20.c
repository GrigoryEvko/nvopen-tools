// Function: ctor_503
// Address: 0x55ac20
//
int ctor_503()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_500A8A0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_500A91C = 1;
  qword_500A8F0 = 0x100000000LL;
  dword_500A8AC &= 0x8000u;
  qword_500A8B8 = 0;
  qword_500A8C0 = 0;
  qword_500A8C8 = 0;
  dword_500A8A8 = v0;
  word_500A8B0 = 0;
  qword_500A8D0 = 0;
  qword_500A8D8 = 0;
  qword_500A8E0 = 0;
  qword_500A8E8 = (__int64)&unk_500A8F8;
  qword_500A900 = 0;
  qword_500A908 = (__int64)&unk_500A920;
  qword_500A910 = 1;
  dword_500A918 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_500A8F0;
  v3 = (unsigned int)qword_500A8F0 + 1LL;
  if ( v3 > HIDWORD(qword_500A8F0) )
  {
    sub_C8D5F0((char *)&unk_500A8F8 - 16, &unk_500A8F8, v3, 8);
    v2 = (unsigned int)qword_500A8F0;
  }
  *(_QWORD *)(qword_500A8E8 + 8 * v2) = v1;
  LODWORD(qword_500A8F0) = qword_500A8F0 + 1;
  qword_500A928 = 0;
  qword_500A930 = (__int64)&unk_49D9748;
  qword_500A938 = 0;
  qword_500A8A0 = (__int64)&unk_49DC090;
  qword_500A940 = (__int64)&unk_49DC1D0;
  qword_500A960 = (__int64)nullsub_23;
  qword_500A958 = (__int64)sub_984030;
  sub_C53080(&qword_500A8A0, "loop-version-annotate-no-alias", 30);
  LOBYTE(qword_500A928) = 1;
  LOWORD(qword_500A938) = 257;
  qword_500A8D0 = 76;
  LOBYTE(dword_500A8AC) = dword_500A8AC & 0x9F | 0x20;
  qword_500A8C8 = (__int64)"Add no-alias annotation for instructions that are disambiguated by memchecks";
  sub_C53130(&qword_500A8A0);
  return __cxa_atexit(sub_984900, &qword_500A8A0, &qword_4A427C0);
}
