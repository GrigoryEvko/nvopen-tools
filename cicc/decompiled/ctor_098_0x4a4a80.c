// Function: ctor_098
// Address: 0x4a4a80
//
int ctor_098()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int result; // eax

  qword_4F923A0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F9241C = 1;
  qword_4F923F0 = 0x100000000LL;
  dword_4F923AC &= 0x8000u;
  qword_4F923B8 = 0;
  qword_4F923C0 = 0;
  qword_4F923C8 = 0;
  dword_4F923A8 = v0;
  word_4F923B0 = 0;
  qword_4F923D0 = 0;
  qword_4F923D8 = 0;
  qword_4F923E0 = 0;
  qword_4F923E8 = (__int64)&unk_4F923F8;
  qword_4F92400 = 0;
  qword_4F92408 = (__int64)&unk_4F92420;
  qword_4F92410 = 1;
  dword_4F92418 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F923F0;
  v3 = (unsigned int)qword_4F923F0 + 1LL;
  if ( v3 > HIDWORD(qword_4F923F0) )
  {
    sub_C8D5F0((char *)&unk_4F923F8 - 16, &unk_4F923F8, v3, 8);
    v2 = (unsigned int)qword_4F923F0;
  }
  *(_QWORD *)(qword_4F923E8 + 8 * v2) = v1;
  LODWORD(qword_4F923F0) = qword_4F923F0 + 1;
  qword_4F92428 = 0;
  qword_4F92430 = (__int64)&unk_49D9748;
  qword_4F92438 = 0;
  qword_4F923A0 = (__int64)&unk_49DC090;
  qword_4F92440 = (__int64)&unk_49DC1D0;
  qword_4F92460 = (__int64)nullsub_23;
  qword_4F92458 = (__int64)sub_984030;
  sub_C53080(&qword_4F923A0, "allow-incomplete-ir", 19);
  LOBYTE(qword_4F92428) = 0;
  LOWORD(qword_4F92438) = 256;
  qword_4F923D0 = 91;
  LOBYTE(dword_4F923AC) = dword_4F923AC & 0x9F | 0x20;
  qword_4F923C8 = (__int64)"Allow incomplete IR on a best effort basis (references to unknown metadata will be dropped)";
  sub_C53130(&qword_4F923A0);
  result = __cxa_atexit(sub_984900, &qword_4F923A0, &qword_4A427C0);
  qword_4F92390 = -8;
  return result;
}
