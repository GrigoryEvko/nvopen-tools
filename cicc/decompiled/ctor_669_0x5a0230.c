// Function: ctor_669
// Address: 0x5a0230
//
int __fastcall ctor_669(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_503BCC0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_503BD3C = 1;
  qword_503BD10 = 0x100000000LL;
  dword_503BCCC &= 0x8000u;
  qword_503BCD8 = 0;
  qword_503BCE0 = 0;
  qword_503BCE8 = 0;
  dword_503BCC8 = v4;
  word_503BCD0 = 0;
  qword_503BCF0 = 0;
  qword_503BCF8 = 0;
  qword_503BD00 = 0;
  qword_503BD08 = (__int64)&unk_503BD18;
  qword_503BD20 = 0;
  qword_503BD28 = (__int64)&unk_503BD40;
  qword_503BD30 = 1;
  dword_503BD38 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_503BD10;
  v7 = (unsigned int)qword_503BD10 + 1LL;
  if ( v7 > HIDWORD(qword_503BD10) )
  {
    sub_C8D5F0((char *)&unk_503BD18 - 16, &unk_503BD18, v7, 8);
    v6 = (unsigned int)qword_503BD10;
  }
  *(_QWORD *)(qword_503BD08 + 8 * v6) = v5;
  LODWORD(qword_503BD10) = qword_503BD10 + 1;
  qword_503BD48 = 0;
  qword_503BD50 = (__int64)&unk_49D9748;
  qword_503BD58 = 0;
  qword_503BCC0 = (__int64)&unk_49DC090;
  qword_503BD60 = (__int64)&unk_49DC1D0;
  qword_503BD80 = (__int64)nullsub_23;
  qword_503BD78 = (__int64)sub_984030;
  sub_C53080(&qword_503BCC0, "restrict-statepoint-remat", 25);
  LOBYTE(qword_503BD48) = 0;
  LOWORD(qword_503BD58) = 256;
  qword_503BCF0 = 38;
  LOBYTE(dword_503BCCC) = dword_503BCCC & 0x9F | 0x20;
  qword_503BCE8 = (__int64)"Restrict remat for statepoint operands";
  sub_C53130(&qword_503BCC0);
  return __cxa_atexit(sub_984900, &qword_503BCC0, &qword_4A427C0);
}
