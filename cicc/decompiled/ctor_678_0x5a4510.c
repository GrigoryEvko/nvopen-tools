// Function: ctor_678
// Address: 0x5a4510
//
int __fastcall ctor_678(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_503F0A0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  dword_503F0AC &= 0x8000u;
  word_503F0B0 = 0;
  qword_503F0F0 = 0x100000000LL;
  qword_503F0B8 = 0;
  qword_503F0C0 = 0;
  qword_503F0C8 = 0;
  dword_503F0A8 = v4;
  qword_503F0D0 = 0;
  qword_503F0D8 = 0;
  qword_503F0E0 = 0;
  qword_503F0E8 = (__int64)&unk_503F0F8;
  qword_503F100 = 0;
  qword_503F108 = (__int64)&unk_503F120;
  qword_503F110 = 1;
  dword_503F118 = 0;
  byte_503F11C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_503F0F0;
  v7 = (unsigned int)qword_503F0F0 + 1LL;
  if ( v7 > HIDWORD(qword_503F0F0) )
  {
    sub_C8D5F0((char *)&unk_503F0F8 - 16, &unk_503F0F8, v7, 8);
    v6 = (unsigned int)qword_503F0F0;
  }
  *(_QWORD *)(qword_503F0E8 + 8 * v6) = v5;
  LODWORD(qword_503F0F0) = qword_503F0F0 + 1;
  qword_503F128 = 0;
  qword_503F130 = (__int64)&unk_49D9728;
  qword_503F138 = 0;
  qword_503F0A0 = (__int64)&unk_49DBF10;
  qword_503F140 = (__int64)&unk_49DC290;
  qword_503F160 = (__int64)nullsub_24;
  qword_503F158 = (__int64)sub_984050;
  sub_C53080(&qword_503F0A0, "canon-nth-function", 18);
  LODWORD(qword_503F128) = -1;
  BYTE4(qword_503F138) = 1;
  LODWORD(qword_503F138) = -1;
  qword_503F0E0 = 1;
  LOBYTE(dword_503F0AC) = dword_503F0AC & 0x9F | 0x20;
  qword_503F0D8 = (__int64)"N";
  qword_503F0C8 = (__int64)"Function number to canonicalize.";
  qword_503F0D0 = 32;
  sub_C53130(&qword_503F0A0);
  return __cxa_atexit(sub_984970, &qword_503F0A0, &qword_4A427C0);
}
