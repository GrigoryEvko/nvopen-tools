// Function: ctor_088
// Address: 0x4a0950
//
int ctor_088()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_4F902E0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F90330 = 0x100000000LL;
  dword_4F902EC &= 0x8000u;
  word_4F902F0 = 0;
  qword_4F902F8 = 0;
  qword_4F90300 = 0;
  dword_4F902E8 = v0;
  qword_4F90308 = 0;
  qword_4F90310 = 0;
  qword_4F90318 = 0;
  qword_4F90320 = 0;
  qword_4F90328 = (__int64)&unk_4F90338;
  qword_4F90340 = 0;
  qword_4F90348 = (__int64)&unk_4F90360;
  qword_4F90350 = 1;
  dword_4F90358 = 0;
  byte_4F9035C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F90330;
  v3 = (unsigned int)qword_4F90330 + 1LL;
  if ( v3 > HIDWORD(qword_4F90330) )
  {
    sub_C8D5F0((char *)&unk_4F90338 - 16, &unk_4F90338, v3, 8);
    v2 = (unsigned int)qword_4F90330;
  }
  *(_QWORD *)(qword_4F90328 + 8 * v2) = v1;
  qword_4F90370 = (__int64)&unk_49D9748;
  LODWORD(qword_4F90330) = qword_4F90330 + 1;
  qword_4F90368 = 0;
  qword_4F902E0 = (__int64)&unk_49DC090;
  qword_4F90380 = (__int64)&unk_49DC1D0;
  qword_4F90378 = 0;
  qword_4F903A0 = (__int64)nullsub_23;
  qword_4F90398 = (__int64)sub_984030;
  sub_C53080(&qword_4F902E0, "disable-converting-i32", 22);
  LOWORD(qword_4F90378) = 257;
  LOBYTE(qword_4F90368) = 1;
  qword_4F90310 = 41;
  LOBYTE(dword_4F902EC) = dword_4F902EC & 0x9F | 0x20;
  qword_4F90308 = (__int64)"Disable converting i32 opts into i16 opts";
  sub_C53130(&qword_4F902E0);
  __cxa_atexit(sub_984900, &qword_4F902E0, &qword_4A427C0);
  qword_4F90200 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F90250 = 0x100000000LL;
  word_4F90210 = 0;
  dword_4F9020C &= 0x8000u;
  qword_4F90218 = 0;
  qword_4F90220 = 0;
  dword_4F90208 = v4;
  qword_4F90228 = 0;
  qword_4F90230 = 0;
  qword_4F90238 = 0;
  qword_4F90240 = 0;
  qword_4F90248 = (__int64)&unk_4F90258;
  qword_4F90260 = 0;
  qword_4F90268 = (__int64)&unk_4F90280;
  qword_4F90270 = 1;
  dword_4F90278 = 0;
  byte_4F9027C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F90250;
  v7 = (unsigned int)qword_4F90250 + 1LL;
  if ( v7 > HIDWORD(qword_4F90250) )
  {
    sub_C8D5F0((char *)&unk_4F90258 - 16, &unk_4F90258, v7, 8);
    v6 = (unsigned int)qword_4F90250;
  }
  *(_QWORD *)(qword_4F90248 + 8 * v6) = v5;
  qword_4F90290 = (__int64)&unk_49D9748;
  LODWORD(qword_4F90250) = qword_4F90250 + 1;
  qword_4F90288 = 0;
  qword_4F90200 = (__int64)&unk_49DC090;
  qword_4F902A0 = (__int64)&unk_49DC1D0;
  qword_4F90298 = 0;
  qword_4F902C0 = (__int64)nullsub_23;
  qword_4F902B8 = (__int64)sub_984030;
  sub_C53080(&qword_4F90200, "instcombine-disable-bitcast-phi-opt", 35);
  LOBYTE(qword_4F90288) = 0;
  LOWORD(qword_4F90298) = 256;
  qword_4F90230 = 56;
  LOBYTE(dword_4F9020C) = dword_4F9020C & 0x9F | 0x20;
  qword_4F90228 = (__int64)"Disable bitcast vector to scalar optimization around Phi";
  sub_C53130(&qword_4F90200);
  return __cxa_atexit(sub_984900, &qword_4F90200, &qword_4A427C0);
}
