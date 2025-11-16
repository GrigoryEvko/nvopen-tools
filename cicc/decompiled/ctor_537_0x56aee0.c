// Function: ctor_537
// Address: 0x56aee0
//
int ctor_537()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_5015E40 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_5015E4C &= 0x8000u;
  word_5015E50 = 0;
  qword_5015E90 = 0x100000000LL;
  qword_5015E58 = 0;
  qword_5015E60 = 0;
  qword_5015E68 = 0;
  dword_5015E48 = v0;
  qword_5015E70 = 0;
  qword_5015E78 = 0;
  qword_5015E80 = 0;
  qword_5015E88 = (__int64)&unk_5015E98;
  qword_5015EA0 = 0;
  qword_5015EA8 = (__int64)&unk_5015EC0;
  qword_5015EB0 = 1;
  dword_5015EB8 = 0;
  byte_5015EBC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5015E90;
  v3 = (unsigned int)qword_5015E90 + 1LL;
  if ( v3 > HIDWORD(qword_5015E90) )
  {
    sub_C8D5F0((char *)&unk_5015E98 - 16, &unk_5015E98, v3, 8);
    v2 = (unsigned int)qword_5015E90;
  }
  *(_QWORD *)(qword_5015E88 + 8 * v2) = v1;
  LODWORD(qword_5015E90) = qword_5015E90 + 1;
  qword_5015EC8 = 0;
  qword_5015ED0 = (__int64)&unk_49DA090;
  qword_5015ED8 = 0;
  qword_5015E40 = (__int64)&unk_49DBF90;
  qword_5015EE0 = (__int64)&unk_49DC230;
  qword_5015F00 = (__int64)nullsub_58;
  qword_5015EF8 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_5015E40, "reuse-lmem-very-long-live-range", 31);
  LODWORD(qword_5015EC8) = 5000;
  BYTE4(qword_5015ED8) = 1;
  LODWORD(qword_5015ED8) = 5000;
  qword_5015E70 = 45;
  LOBYTE(dword_5015E4C) = dword_5015E4C & 0x9F | 0x20;
  qword_5015E68 = (__int64)"Define the threshold for very long live range";
  sub_C53130(&qword_5015E40);
  return __cxa_atexit(sub_B2B680, &qword_5015E40, &qword_4A427C0);
}
