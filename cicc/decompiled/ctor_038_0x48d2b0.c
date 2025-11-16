// Function: ctor_038
// Address: 0x48d2b0
//
int ctor_038()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4F83AA0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F83B1C = 1;
  qword_4F83AF0 = 0x100000000LL;
  dword_4F83AAC &= 0x8000u;
  qword_4F83AB8 = 0;
  qword_4F83AC0 = 0;
  qword_4F83AC8 = 0;
  dword_4F83AA8 = v0;
  word_4F83AB0 = 0;
  qword_4F83AD0 = 0;
  qword_4F83AD8 = 0;
  qword_4F83AE0 = 0;
  qword_4F83AE8 = (__int64)&unk_4F83AF8;
  qword_4F83B00 = 0;
  qword_4F83B08 = (__int64)&unk_4F83B20;
  qword_4F83B10 = 1;
  dword_4F83B18 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F83AF0;
  v3 = (unsigned int)qword_4F83AF0 + 1LL;
  if ( v3 > HIDWORD(qword_4F83AF0) )
  {
    sub_C8D5F0((char *)&unk_4F83AF8 - 16, &unk_4F83AF8, v3, 8);
    v2 = (unsigned int)qword_4F83AF0;
  }
  *(_QWORD *)(qword_4F83AE8 + 8 * v2) = v1;
  LODWORD(qword_4F83AF0) = qword_4F83AF0 + 1;
  qword_4F83B28 = 0;
  qword_4F83B30 = (__int64)&unk_49D9748;
  qword_4F83B38 = 0;
  qword_4F83AA0 = (__int64)&unk_49DC090;
  qword_4F83B40 = (__int64)&unk_49DC1D0;
  qword_4F83B60 = (__int64)nullsub_23;
  qword_4F83B58 = (__int64)sub_984030;
  sub_C53080(&qword_4F83AA0, "profile-isfs", 12);
  LOBYTE(qword_4F83B28) = 0;
  qword_4F83AD0 = 42;
  LOBYTE(dword_4F83AAC) = dword_4F83AAC & 0x9F | 0x20;
  LOWORD(qword_4F83B38) = 256;
  qword_4F83AC8 = (__int64)"Profile uses flow sensitive discriminators";
  sub_C53130(&qword_4F83AA0);
  return __cxa_atexit(sub_984900, &qword_4F83AA0, &qword_4A427C0);
}
