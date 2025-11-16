// Function: ctor_530
// Address: 0x567620
//
int ctor_530()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // r15
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v12; // [rsp+8h] [rbp-38h]

  qword_5013BC0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5013C10 = 0x100000000LL;
  dword_5013BCC &= 0x8000u;
  word_5013BD0 = 0;
  qword_5013BD8 = 0;
  qword_5013BE0 = 0;
  dword_5013BC8 = v0;
  qword_5013BE8 = 0;
  qword_5013BF0 = 0;
  qword_5013BF8 = 0;
  qword_5013C00 = 0;
  qword_5013C08 = (__int64)&unk_5013C18;
  qword_5013C20 = 0;
  qword_5013C28 = (__int64)&unk_5013C40;
  qword_5013C30 = 1;
  dword_5013C38 = 0;
  byte_5013C3C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_5013C10;
  v3 = (unsigned int)qword_5013C10 + 1LL;
  if ( v3 > HIDWORD(qword_5013C10) )
  {
    sub_C8D5F0((char *)&unk_5013C18 - 16, &unk_5013C18, v3, 8);
    v2 = (unsigned int)qword_5013C10;
  }
  *(_QWORD *)(qword_5013C08 + 8 * v2) = v1;
  qword_5013C50 = (__int64)&unk_49D9748;
  qword_5013BC0 = (__int64)&unk_49DC090;
  qword_5013C60 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5013C10) = qword_5013C10 + 1;
  qword_5013C80 = (__int64)nullsub_23;
  qword_5013C48 = 0;
  qword_5013C78 = (__int64)sub_984030;
  qword_5013C58 = 0;
  sub_C53080(&qword_5013BC0, "lsa-opt", 7);
  LOWORD(qword_5013C58) = 257;
  LOBYTE(qword_5013C48) = 1;
  qword_5013BF0 = 47;
  LOBYTE(dword_5013BCC) = dword_5013BCC & 0x9F | 0x20;
  qword_5013BE8 = (__int64)"Optimize copying of struct args to local memory";
  sub_C53130(&qword_5013BC0);
  __cxa_atexit(sub_984900, &qword_5013BC0, &qword_4A427C0);
  qword_5013AE0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5013B30 = 0x100000000LL;
  dword_5013AEC &= 0x8000u;
  qword_5013B28 = (__int64)&unk_5013B38;
  word_5013AF0 = 0;
  qword_5013AF8 = 0;
  dword_5013AE8 = v4;
  qword_5013B00 = 0;
  qword_5013B08 = 0;
  qword_5013B10 = 0;
  qword_5013B18 = 0;
  qword_5013B20 = 0;
  qword_5013B40 = 0;
  qword_5013B48 = (__int64)&unk_5013B60;
  qword_5013B50 = 1;
  dword_5013B58 = 0;
  byte_5013B5C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5013B30;
  if ( (unsigned __int64)(unsigned int)qword_5013B30 + 1 > HIDWORD(qword_5013B30) )
  {
    v12 = v5;
    sub_C8D5F0((char *)&unk_5013B38 - 16, &unk_5013B38, (unsigned int)qword_5013B30 + 1LL, 8);
    v6 = (unsigned int)qword_5013B30;
    v5 = v12;
  }
  *(_QWORD *)(qword_5013B28 + 8 * v6) = v5;
  qword_5013B70 = (__int64)&unk_49D9748;
  qword_5013AE0 = (__int64)&unk_49DC090;
  qword_5013B80 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5013B30) = qword_5013B30 + 1;
  qword_5013BA0 = (__int64)nullsub_23;
  qword_5013B68 = 0;
  qword_5013B98 = (__int64)sub_984030;
  qword_5013B78 = 0;
  sub_C53080(&qword_5013AE0, "lower-read-only-devicefn-byval", 30);
  LOWORD(qword_5013B78) = 256;
  LOBYTE(qword_5013B68) = 0;
  qword_5013B10 = 60;
  LOBYTE(dword_5013AEC) = dword_5013AEC & 0x9F | 0x20;
  qword_5013B08 = (__int64)"Handling byval attribute of args to device functions as well";
  sub_C53130(&qword_5013AE0);
  __cxa_atexit(sub_984900, &qword_5013AE0, &qword_4A427C0);
  qword_5013A00 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_5013A50 = 0x100000000LL;
  dword_5013A0C &= 0x8000u;
  word_5013A10 = 0;
  qword_5013A48 = (__int64)&unk_5013A58;
  qword_5013A18 = 0;
  dword_5013A08 = v7;
  qword_5013A20 = 0;
  qword_5013A28 = 0;
  qword_5013A30 = 0;
  qword_5013A38 = 0;
  qword_5013A40 = 0;
  qword_5013A60 = 0;
  qword_5013A68 = (__int64)&unk_5013A80;
  qword_5013A70 = 1;
  dword_5013A78 = 0;
  byte_5013A7C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_5013A50;
  v10 = (unsigned int)qword_5013A50 + 1LL;
  if ( v10 > HIDWORD(qword_5013A50) )
  {
    sub_C8D5F0((char *)&unk_5013A58 - 16, &unk_5013A58, v10, 8);
    v9 = (unsigned int)qword_5013A50;
  }
  *(_QWORD *)(qword_5013A48 + 8 * v9) = v8;
  qword_5013A90 = (__int64)&unk_49D9748;
  qword_5013A00 = (__int64)&unk_49DC090;
  qword_5013AA0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5013A50) = qword_5013A50 + 1;
  qword_5013AC0 = (__int64)nullsub_23;
  qword_5013A88 = 0;
  qword_5013AB8 = (__int64)sub_984030;
  qword_5013A98 = 0;
  sub_C53080(&qword_5013A00, "hoist-load-param", 16);
  LOBYTE(qword_5013A88) = 0;
  LOWORD(qword_5013A98) = 256;
  qword_5013A30 = 40;
  LOBYTE(dword_5013A0C) = dword_5013A0C & 0x9F | 0x20;
  qword_5013A28 = (__int64)"Generate all ld.param in the entry block";
  sub_C53130(&qword_5013A00);
  return __cxa_atexit(sub_984900, &qword_5013A00, &qword_4A427C0);
}
