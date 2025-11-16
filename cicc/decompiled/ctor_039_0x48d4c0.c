// Function: ctor_039
// Address: 0x48d4c0
//
int ctor_039()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4F83B80 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4F83B8C &= 0x8000u;
  word_4F83B90 = 0;
  qword_4F83BD0 = 0x100000000LL;
  qword_4F83B98 = 0;
  qword_4F83BA0 = 0;
  qword_4F83BA8 = 0;
  dword_4F83B88 = v0;
  qword_4F83BB0 = 0;
  qword_4F83BB8 = 0;
  qword_4F83BC0 = 0;
  qword_4F83BC8 = (__int64)&unk_4F83BD8;
  qword_4F83BE0 = 0;
  qword_4F83BE8 = (__int64)&unk_4F83C00;
  qword_4F83BF0 = 1;
  dword_4F83BF8 = 0;
  byte_4F83BFC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F83BD0;
  v3 = (unsigned int)qword_4F83BD0 + 1LL;
  if ( v3 > HIDWORD(qword_4F83BD0) )
  {
    sub_C8D5F0((char *)&unk_4F83BD8 - 16, &unk_4F83BD8, v3, 8);
    v2 = (unsigned int)qword_4F83BD0;
  }
  *(_QWORD *)(qword_4F83BC8 + 8 * v2) = v1;
  LODWORD(qword_4F83BD0) = qword_4F83BD0 + 1;
  qword_4F83C08 = 0;
  qword_4F83C10 = (__int64)&unk_49DC110;
  qword_4F83C18 = 0;
  qword_4F83B80 = (__int64)&unk_49D97F0;
  qword_4F83C20 = (__int64)&unk_49DC200;
  qword_4F83C40 = (__int64)nullsub_26;
  qword_4F83C38 = (__int64)sub_9C26D0;
  sub_C53080(&qword_4F83B80, "remarks-section", 15);
  qword_4F83BB0 = 133;
  qword_4F83BA8 = (__int64)"Emit a section containing remark diagnostics metadata. By default, this is enabled for the fo"
                           "llowing formats: yaml-strtab, bitstream.";
  LODWORD(qword_4F83C08) = 0;
  BYTE4(qword_4F83C18) = 1;
  LODWORD(qword_4F83C18) = 0;
  LOBYTE(dword_4F83B8C) = dword_4F83B8C & 0x9F | 0x20;
  sub_C53130(&qword_4F83B80);
  return __cxa_atexit(sub_9C44F0, &qword_4F83B80, &qword_4A427C0);
}
