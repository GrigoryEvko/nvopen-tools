// Function: ctor_069
// Address: 0x4988c0
//
int ctor_069()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_4F8BCC0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F8BD3C = 1;
  qword_4F8BD10 = 0x100000000LL;
  dword_4F8BCCC &= 0x8000u;
  qword_4F8BCD8 = 0;
  qword_4F8BCE0 = 0;
  qword_4F8BCE8 = 0;
  dword_4F8BCC8 = v0;
  word_4F8BCD0 = 0;
  qword_4F8BCF0 = 0;
  qword_4F8BCF8 = 0;
  qword_4F8BD00 = 0;
  qword_4F8BD08 = (__int64)&unk_4F8BD18;
  qword_4F8BD20 = 0;
  qword_4F8BD28 = (__int64)&unk_4F8BD40;
  qword_4F8BD30 = 1;
  dword_4F8BD38 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F8BD10;
  v3 = (unsigned int)qword_4F8BD10 + 1LL;
  if ( v3 > HIDWORD(qword_4F8BD10) )
  {
    sub_C8D5F0((char *)&unk_4F8BD18 - 16, &unk_4F8BD18, v3, 8);
    v2 = (unsigned int)qword_4F8BD10;
  }
  *(_QWORD *)(qword_4F8BD08 + 8 * v2) = v1;
  qword_4F8BD48 = &byte_4F8BD58;
  qword_4F8BD70 = (__int64)&byte_4F8BD80;
  LODWORD(qword_4F8BD10) = qword_4F8BD10 + 1;
  qword_4F8BD50 = 0;
  qword_4F8BD68 = (__int64)&unk_49DC130;
  byte_4F8BD58 = 0;
  byte_4F8BD80 = 0;
  qword_4F8BCC0 = (__int64)&unk_49DC010;
  qword_4F8BD98 = (__int64)&unk_49DC350;
  qword_4F8BD78 = 0;
  qword_4F8BDB8 = (__int64)nullsub_92;
  byte_4F8BD90 = 0;
  qword_4F8BDB0 = (__int64)sub_BC4D70;
  sub_C53080(&qword_4F8BCC0, "internalize-public-api-file", 27);
  qword_4F8BD00 = 8;
  qword_4F8BCF8 = (__int64)"filename";
  qword_4F8BCE8 = (__int64)"A file containing list of symbol names to preserve";
  qword_4F8BCF0 = 50;
  sub_C53130(&qword_4F8BCC0);
  __cxa_atexit(sub_BC5A40, &qword_4F8BCC0, &qword_4A427C0);
  qword_4F8BBC0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F8BBD8 = 0;
  qword_4F8BBE0 = 0;
  qword_4F8BBE8 = 0;
  qword_4F8BBF0 = 0;
  dword_4F8BBCC = dword_4F8BBCC & 0x8000 | 1;
  word_4F8BBD0 = 0;
  qword_4F8BC10 = 0x100000000LL;
  dword_4F8BBC8 = v4;
  qword_4F8BBF8 = 0;
  qword_4F8BC00 = 0;
  qword_4F8BC08 = (__int64)&unk_4F8BC18;
  qword_4F8BC20 = 0;
  qword_4F8BC28 = (__int64)&unk_4F8BC40;
  qword_4F8BC30 = 1;
  dword_4F8BC38 = 0;
  byte_4F8BC3C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F8BC10;
  v7 = (unsigned int)qword_4F8BC10 + 1LL;
  if ( v7 > HIDWORD(qword_4F8BC10) )
  {
    sub_C8D5F0((char *)&unk_4F8BC18 - 16, &unk_4F8BC18, v7, 8);
    v6 = (unsigned int)qword_4F8BC10;
  }
  *(_QWORD *)(qword_4F8BC08 + 8 * v6) = v5;
  LODWORD(qword_4F8BC10) = qword_4F8BC10 + 1;
  qword_4F8BC48 = 0;
  qword_4F8BBC0 = (__int64)&unk_49DAD08;
  qword_4F8BC98 = (__int64)&unk_49DC350;
  qword_4F8BC50 = 0;
  qword_4F8BCB8 = (__int64)nullsub_81;
  qword_4F8BC58 = 0;
  qword_4F8BCB0 = (__int64)sub_BB8600;
  qword_4F8BC60 = 0;
  qword_4F8BC68 = 0;
  qword_4F8BC70 = 0;
  byte_4F8BC78 = 0;
  qword_4F8BC80 = 0;
  qword_4F8BC88 = 0;
  qword_4F8BC90 = 0;
  sub_C53080(&qword_4F8BBC0, "internalize-public-api-list", 27);
  BYTE1(dword_4F8BBCC) |= 2u;
  qword_4F8BBF8 = (__int64)"list";
  qword_4F8BC00 = 4;
  qword_4F8BBE8 = (__int64)"A list of symbol names to preserve";
  qword_4F8BBF0 = 34;
  sub_C53130(&qword_4F8BBC0);
  return __cxa_atexit(sub_BB89D0, &qword_4F8BBC0, &qword_4A427C0);
}
