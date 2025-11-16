// Function: ctor_075
// Address: 0x49b4b0
//
int ctor_075()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4F8D8E0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4F8D8EC &= 0x8000u;
  word_4F8D8F0 = 0;
  qword_4F8D930 = 0x100000000LL;
  qword_4F8D8F8 = 0;
  qword_4F8D900 = 0;
  qword_4F8D908 = 0;
  dword_4F8D8E8 = v0;
  qword_4F8D910 = 0;
  qword_4F8D918 = 0;
  qword_4F8D920 = 0;
  qword_4F8D928 = (__int64)&unk_4F8D938;
  qword_4F8D940 = 0;
  qword_4F8D948 = (__int64)&unk_4F8D960;
  qword_4F8D950 = 1;
  dword_4F8D958 = 0;
  byte_4F8D95C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F8D930;
  v3 = (unsigned int)qword_4F8D930 + 1LL;
  if ( v3 > HIDWORD(qword_4F8D930) )
  {
    sub_C8D5F0((char *)&unk_4F8D938 - 16, &unk_4F8D938, v3, 8);
    v2 = (unsigned int)qword_4F8D930;
  }
  *(_QWORD *)(qword_4F8D928 + 8 * v2) = v1;
  LODWORD(qword_4F8D930) = qword_4F8D930 + 1;
  qword_4F8D968 = 0;
  qword_4F8D970 = (__int64)&unk_49D9728;
  qword_4F8D978 = 0;
  qword_4F8D8E0 = (__int64)&unk_49DBF10;
  qword_4F8D980 = (__int64)&unk_49DC290;
  qword_4F8D9A0 = (__int64)nullsub_24;
  qword_4F8D998 = (__int64)sub_984050;
  sub_C53080(&qword_4F8D8E0, "alias-set-saturation-threshold", 30);
  LODWORD(qword_4F8D968) = 250;
  BYTE4(qword_4F8D978) = 1;
  LODWORD(qword_4F8D978) = 250;
  qword_4F8D910 = 86;
  LOBYTE(dword_4F8D8EC) = dword_4F8D8EC & 0x9F | 0x20;
  qword_4F8D908 = (__int64)"The maximum total number of memory locations alias sets may contain before degradation";
  sub_C53130(&qword_4F8D8E0);
  return __cxa_atexit(sub_984970, &qword_4F8D8E0, &qword_4A427C0);
}
