// Function: ctor_411
// Address: 0x52f350
//
int ctor_411()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_4FEE3E0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FEE45C = 1;
  qword_4FEE430 = 0x100000000LL;
  dword_4FEE3EC &= 0x8000u;
  qword_4FEE3F8 = 0;
  qword_4FEE400 = 0;
  qword_4FEE408 = 0;
  dword_4FEE3E8 = v0;
  word_4FEE3F0 = 0;
  qword_4FEE410 = 0;
  qword_4FEE418 = 0;
  qword_4FEE420 = 0;
  qword_4FEE428 = (__int64)&unk_4FEE438;
  qword_4FEE440 = 0;
  qword_4FEE448 = (__int64)&unk_4FEE460;
  qword_4FEE450 = 1;
  dword_4FEE458 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FEE430;
  v3 = (unsigned int)qword_4FEE430 + 1LL;
  if ( v3 > HIDWORD(qword_4FEE430) )
  {
    sub_C8D5F0((char *)&unk_4FEE438 - 16, &unk_4FEE438, v3, 8);
    v2 = (unsigned int)qword_4FEE430;
  }
  *(_QWORD *)(qword_4FEE428 + 8 * v2) = v1;
  LODWORD(qword_4FEE430) = qword_4FEE430 + 1;
  qword_4FEE468 = 0;
  qword_4FEE470 = (__int64)&unk_49D9748;
  qword_4FEE478 = 0;
  qword_4FEE3E0 = (__int64)&unk_49DC090;
  qword_4FEE480 = (__int64)&unk_49DC1D0;
  qword_4FEE4A0 = (__int64)nullsub_23;
  qword_4FEE498 = (__int64)sub_984030;
  sub_C53080(&qword_4FEE3E0, "tysan-writes-always-set-type", 28);
  qword_4FEE410 = 26;
  qword_4FEE408 = (__int64)"Writes always set the type";
  LOBYTE(qword_4FEE468) = 0;
  LOBYTE(dword_4FEE3EC) = dword_4FEE3EC & 0x9F | 0x20;
  LOWORD(qword_4FEE478) = 256;
  sub_C53130(&qword_4FEE3E0);
  return __cxa_atexit(sub_984900, &qword_4FEE3E0, &qword_4A427C0);
}
