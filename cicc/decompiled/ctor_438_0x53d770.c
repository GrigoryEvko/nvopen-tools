// Function: ctor_438
// Address: 0x53d770
//
int ctor_438()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4FF9A40 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FF9A4C &= 0x8000u;
  word_4FF9A50 = 0;
  qword_4FF9A90 = 0x100000000LL;
  qword_4FF9A58 = 0;
  qword_4FF9A60 = 0;
  qword_4FF9A68 = 0;
  dword_4FF9A48 = v0;
  qword_4FF9A70 = 0;
  qword_4FF9A78 = 0;
  qword_4FF9A80 = 0;
  qword_4FF9A88 = (__int64)&unk_4FF9A98;
  qword_4FF9AA0 = 0;
  qword_4FF9AA8 = (__int64)&unk_4FF9AC0;
  qword_4FF9AB0 = 1;
  dword_4FF9AB8 = 0;
  byte_4FF9ABC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FF9A90;
  v3 = (unsigned int)qword_4FF9A90 + 1LL;
  if ( v3 > HIDWORD(qword_4FF9A90) )
  {
    sub_C8D5F0((char *)&unk_4FF9A98 - 16, &unk_4FF9A98, v3, 8);
    v2 = (unsigned int)qword_4FF9A90;
  }
  *(_QWORD *)(qword_4FF9A88 + 8 * v2) = v1;
  LODWORD(qword_4FF9A90) = qword_4FF9A90 + 1;
  qword_4FF9AC8 = 0;
  qword_4FF9AD0 = (__int64)&unk_49D9728;
  qword_4FF9AD8 = 0;
  qword_4FF9A40 = (__int64)&unk_49DBF10;
  qword_4FF9AE0 = (__int64)&unk_49DC290;
  qword_4FF9B00 = (__int64)nullsub_24;
  qword_4FF9AF8 = (__int64)sub_984050;
  sub_C53080(&qword_4FF9A40, "arc-opt-max-ptr-states", 22);
  qword_4FF9A70 = 57;
  LODWORD(qword_4FF9AC8) = 4095;
  BYTE4(qword_4FF9AD8) = 1;
  LODWORD(qword_4FF9AD8) = 4095;
  LOBYTE(dword_4FF9A4C) = dword_4FF9A4C & 0x9F | 0x20;
  qword_4FF9A68 = (__int64)"Maximum number of ptr states the optimizer keeps track of";
  sub_C53130(&qword_4FF9A40);
  return __cxa_atexit(sub_984970, &qword_4FF9A40, &qword_4A427C0);
}
