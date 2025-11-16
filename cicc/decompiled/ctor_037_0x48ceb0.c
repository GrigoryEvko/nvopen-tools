// Function: ctor_037
// Address: 0x48ceb0
//
int ctor_037()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_4F839C0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F83A3C = 1;
  qword_4F83A10 = 0x100000000LL;
  dword_4F839CC &= 0x8000u;
  qword_4F839D8 = 0;
  qword_4F839E0 = 0;
  qword_4F839E8 = 0;
  dword_4F839C8 = v0;
  word_4F839D0 = 0;
  qword_4F839F0 = 0;
  qword_4F839F8 = 0;
  qword_4F83A00 = 0;
  qword_4F83A08 = (__int64)&unk_4F83A18;
  qword_4F83A20 = 0;
  qword_4F83A28 = (__int64)&unk_4F83A40;
  qword_4F83A30 = 1;
  dword_4F83A38 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F83A10;
  v3 = (unsigned int)qword_4F83A10 + 1LL;
  if ( v3 > HIDWORD(qword_4F83A10) )
  {
    sub_C8D5F0((char *)&unk_4F83A18 - 16, &unk_4F83A18, v3, 8);
    v2 = (unsigned int)qword_4F83A10;
  }
  *(_QWORD *)(qword_4F83A08 + 8 * v2) = v1;
  LODWORD(qword_4F83A10) = qword_4F83A10 + 1;
  byte_4F83A60 = 0;
  qword_4F83A50 = (__int64)&unk_49DB998;
  qword_4F83A48 = 0;
  qword_4F83A58 = 0;
  qword_4F839C0 = (__int64)&unk_49DB9B8;
  qword_4F83A68 = (__int64)&unk_49DC2C0;
  qword_4F83A88 = (__int64)nullsub_121;
  qword_4F83A80 = (__int64)sub_C1A370;
  sub_C53080(&qword_4F839C0, "profile-symbol-list-cutoff", 26);
  qword_4F83A48 = -1;
  byte_4F83A60 = 1;
  qword_4F83A58 = -1;
  qword_4F839F0 = 118;
  LOBYTE(dword_4F839CC) = dword_4F839CC & 0x9F | 0x20;
  qword_4F839E8 = (__int64)"Cutoff value about how many symbols in profile symbol list will be used. This is very useful "
                           "for performance debugging";
  sub_C53130(&qword_4F839C0);
  __cxa_atexit(sub_C1A610, &qword_4F839C0, &qword_4A427C0);
  qword_4F838E0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4F838EC &= 0x8000u;
  word_4F838F0 = 0;
  qword_4F83930 = 0x100000000LL;
  qword_4F838F8 = 0;
  qword_4F83900 = 0;
  qword_4F83908 = 0;
  dword_4F838E8 = v4;
  qword_4F83910 = 0;
  qword_4F83918 = 0;
  qword_4F83920 = 0;
  qword_4F83928 = (__int64)&unk_4F83938;
  qword_4F83940 = 0;
  qword_4F83948 = (__int64)&unk_4F83960;
  qword_4F83950 = 1;
  dword_4F83958 = 0;
  byte_4F8395C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F83930;
  v7 = (unsigned int)qword_4F83930 + 1LL;
  if ( v7 > HIDWORD(qword_4F83930) )
  {
    sub_C8D5F0((char *)&unk_4F83938 - 16, &unk_4F83938, v7, 8);
    v6 = (unsigned int)qword_4F83930;
  }
  *(_QWORD *)(qword_4F83928 + 8 * v6) = v5;
  LODWORD(qword_4F83930) = qword_4F83930 + 1;
  qword_4F83968 = 0;
  qword_4F83970 = (__int64)&unk_49D9748;
  qword_4F83978 = 0;
  qword_4F838E0 = (__int64)&unk_49DC090;
  qword_4F83980 = (__int64)&unk_49DC1D0;
  qword_4F839A0 = (__int64)nullsub_23;
  qword_4F83998 = (__int64)sub_984030;
  sub_C53080(&qword_4F838E0, "generate-merged-base-profiles", 29);
  qword_4F83910 = 144;
  qword_4F83908 = (__int64)"When generating nested context-sensitive profiles, always generate extra base profile for fun"
                           "ction with all its context profiles merged into it.";
  sub_C53130(&qword_4F838E0);
  return __cxa_atexit(sub_984900, &qword_4F838E0, &qword_4A427C0);
}
