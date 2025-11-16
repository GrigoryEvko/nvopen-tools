// Function: ctor_553
// Address: 0x56fd20
//
int ctor_553()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_501E140 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_501E1BC = 1;
  qword_501E190 = 0x100000000LL;
  dword_501E14C &= 0x8000u;
  qword_501E158 = 0;
  qword_501E160 = 0;
  qword_501E168 = 0;
  dword_501E148 = v0;
  word_501E150 = 0;
  qword_501E170 = 0;
  qword_501E178 = 0;
  qword_501E180 = 0;
  qword_501E188 = (__int64)&unk_501E198;
  qword_501E1A0 = 0;
  qword_501E1A8 = (__int64)&unk_501E1C0;
  qword_501E1B0 = 1;
  dword_501E1B8 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_501E190;
  v3 = (unsigned int)qword_501E190 + 1LL;
  if ( v3 > HIDWORD(qword_501E190) )
  {
    sub_C8D5F0((char *)&unk_501E198 - 16, &unk_501E198, v3, 8);
    v2 = (unsigned int)qword_501E190;
  }
  *(_QWORD *)(qword_501E188 + 8 * v2) = v1;
  LODWORD(qword_501E190) = qword_501E190 + 1;
  qword_501E1C8 = 0;
  qword_501E1D0 = (__int64)&unk_49D9748;
  qword_501E1D8 = 0;
  qword_501E140 = (__int64)&unk_49DC090;
  qword_501E1E0 = (__int64)&unk_49DC1D0;
  qword_501E200 = (__int64)nullsub_23;
  qword_501E1F8 = (__int64)sub_984030;
  sub_C53080(&qword_501E140, "disable-cgdata-for-merging", 26);
  qword_501E170 = 90;
  LOBYTE(qword_501E1C8) = 0;
  LOBYTE(dword_501E14C) = dword_501E14C & 0x9F | 0x20;
  qword_501E168 = (__int64)"Disable codegen data for function merging. Local merging is still enabled within a module.";
  LOWORD(qword_501E1D8) = 256;
  sub_C53130(&qword_501E140);
  return __cxa_atexit(sub_984900, &qword_501E140, &qword_4A427C0);
}
