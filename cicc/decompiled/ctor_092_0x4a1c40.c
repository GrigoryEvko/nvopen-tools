// Function: ctor_092
// Address: 0x4a1c40
//
int ctor_092()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_4F90BA0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F90BF0 = 0x100000000LL;
  word_4F90BB0 = 0;
  dword_4F90BAC &= 0x8000u;
  qword_4F90BB8 = 0;
  qword_4F90BC0 = 0;
  dword_4F90BA8 = v0;
  qword_4F90BC8 = 0;
  qword_4F90BD0 = 0;
  qword_4F90BD8 = 0;
  qword_4F90BE0 = 0;
  qword_4F90BE8 = (__int64)&unk_4F90BF8;
  qword_4F90C00 = 0;
  qword_4F90C08 = (__int64)&unk_4F90C20;
  qword_4F90C10 = 1;
  dword_4F90C18 = 0;
  byte_4F90C1C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F90BF0;
  v3 = (unsigned int)qword_4F90BF0 + 1LL;
  if ( v3 > HIDWORD(qword_4F90BF0) )
  {
    sub_C8D5F0((char *)&unk_4F90BF8 - 16, &unk_4F90BF8, v3, 8);
    v2 = (unsigned int)qword_4F90BF0;
  }
  *(_QWORD *)(qword_4F90BE8 + 8 * v2) = v1;
  LODWORD(qword_4F90BF0) = qword_4F90BF0 + 1;
  qword_4F90C28 = 0;
  qword_4F90C30 = (__int64)&unk_49D9748;
  qword_4F90C38 = 0;
  qword_4F90BA0 = (__int64)&unk_49DC090;
  qword_4F90C40 = (__int64)&unk_49DC1D0;
  qword_4F90C60 = (__int64)nullsub_23;
  qword_4F90C58 = (__int64)sub_984030;
  sub_C53080(&qword_4F90BA0, "instcombine-verify-known-bits", 29);
  qword_4F90BD0 = 72;
  qword_4F90BC8 = (__int64)"Verify that computeKnownBits() and SimplifyDemandedBits() are consistent";
  LOBYTE(qword_4F90C28) = 0;
  LOBYTE(dword_4F90BAC) = dword_4F90BAC & 0x9F | 0x20;
  LOWORD(qword_4F90C38) = 256;
  sub_C53130(&qword_4F90BA0);
  __cxa_atexit(sub_984900, &qword_4F90BA0, &qword_4A427C0);
  qword_4F90AC0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F90B3C = 1;
  qword_4F90B10 = 0x100000000LL;
  dword_4F90ACC &= 0x8000u;
  qword_4F90AD8 = 0;
  qword_4F90AE0 = 0;
  qword_4F90AE8 = 0;
  dword_4F90AC8 = v4;
  word_4F90AD0 = 0;
  qword_4F90AF0 = 0;
  qword_4F90AF8 = 0;
  qword_4F90B00 = 0;
  qword_4F90B08 = (__int64)&unk_4F90B18;
  qword_4F90B20 = 0;
  qword_4F90B28 = (__int64)&unk_4F90B40;
  qword_4F90B30 = 1;
  dword_4F90B38 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F90B10;
  v7 = (unsigned int)qword_4F90B10 + 1LL;
  if ( v7 > HIDWORD(qword_4F90B10) )
  {
    sub_C8D5F0((char *)&unk_4F90B18 - 16, &unk_4F90B18, v7, 8);
    v6 = (unsigned int)qword_4F90B10;
  }
  *(_QWORD *)(qword_4F90B08 + 8 * v6) = v5;
  LODWORD(qword_4F90B10) = qword_4F90B10 + 1;
  qword_4F90B48 = 0;
  qword_4F90B50 = (__int64)&unk_49D9728;
  qword_4F90B58 = 0;
  qword_4F90AC0 = (__int64)&unk_49DBF10;
  qword_4F90B60 = (__int64)&unk_49DC290;
  qword_4F90B80 = (__int64)nullsub_24;
  qword_4F90B78 = (__int64)sub_984050;
  sub_C53080(&qword_4F90AC0, "instcombine-simplify-vector-elts-depth", 38);
  qword_4F90AF0 = 67;
  qword_4F90AE8 = (__int64)"Depth limit when simplifying vector instructions and their operands";
  LODWORD(qword_4F90B48) = 10;
  BYTE4(qword_4F90B58) = 1;
  LODWORD(qword_4F90B58) = 10;
  LOBYTE(dword_4F90ACC) = dword_4F90ACC & 0x9F | 0x20;
  sub_C53130(&qword_4F90AC0);
  return __cxa_atexit(sub_984970, &qword_4F90AC0, &qword_4A427C0);
}
