// Function: ctor_476
// Address: 0x54e650
//
int ctor_476()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_50043E0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_50043EC &= 0x8000u;
  word_50043F0 = 0;
  qword_5004430 = 0x100000000LL;
  qword_50043F8 = 0;
  qword_5004400 = 0;
  qword_5004408 = 0;
  dword_50043E8 = v0;
  qword_5004410 = 0;
  qword_5004418 = 0;
  qword_5004420 = 0;
  qword_5004428 = (__int64)&unk_5004438;
  qword_5004440 = 0;
  qword_5004448 = (__int64)&unk_5004460;
  qword_5004450 = 1;
  dword_5004458 = 0;
  byte_500445C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5004430;
  v3 = (unsigned int)qword_5004430 + 1LL;
  if ( v3 > HIDWORD(qword_5004430) )
  {
    sub_C8D5F0((char *)&unk_5004438 - 16, &unk_5004438, v3, 8);
    v2 = (unsigned int)qword_5004430;
  }
  *(_QWORD *)(qword_5004428 + 8 * v2) = v1;
  LODWORD(qword_5004430) = qword_5004430 + 1;
  qword_5004468 = 0;
  qword_5004470 = (__int64)&unk_49D9748;
  qword_5004478 = 0;
  qword_50043E0 = (__int64)&unk_49DC090;
  qword_5004480 = (__int64)&unk_49DC1D0;
  qword_50044A0 = (__int64)nullsub_23;
  qword_5004498 = (__int64)sub_984030;
  sub_C53080(&qword_50043E0, "enable-memcpyopt-without-libcalls", 33);
  qword_5004410 = 48;
  LOBYTE(dword_50043EC) = dword_50043EC & 0x9F | 0x20;
  qword_5004408 = (__int64)"Enable memcpyopt even when libcalls are disabled";
  sub_C53130(&qword_50043E0);
  return __cxa_atexit(sub_984900, &qword_50043E0, &qword_4A427C0);
}
