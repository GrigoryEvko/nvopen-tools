// Function: ctor_011
// Address: 0x453100
//
int ctor_011()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v12; // [rsp+8h] [rbp-38h]

  qword_4F80060 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F800B0 = 0x100000000LL;
  dword_4F8006C &= 0x8000u;
  word_4F80070 = 0;
  qword_4F80078 = 0;
  qword_4F80080 = 0;
  dword_4F80068 = v0;
  qword_4F80088 = 0;
  qword_4F80090 = 0;
  qword_4F80098 = 0;
  qword_4F800A0 = 0;
  qword_4F800A8 = (__int64)&unk_4F800B8;
  qword_4F800C0 = 0;
  qword_4F800C8 = (__int64)&unk_4F800E0;
  qword_4F800D0 = 1;
  dword_4F800D8 = 0;
  byte_4F800DC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F800B0;
  v3 = (unsigned int)qword_4F800B0 + 1LL;
  if ( v3 > HIDWORD(qword_4F800B0) )
  {
    sub_C8D5F0((char *)&unk_4F800B8 - 16, &unk_4F800B8, v3, 8);
    v2 = (unsigned int)qword_4F800B0;
  }
  *(_QWORD *)(qword_4F800A8 + 8 * v2) = v1;
  qword_4F800F0 = (__int64)&unk_49D9748;
  qword_4F80060 = (__int64)&unk_49DC090;
  LODWORD(qword_4F800B0) = qword_4F800B0 + 1;
  qword_4F800E8 = 0;
  qword_4F80100 = (__int64)&unk_49DC1D0;
  qword_4F800F8 = 0;
  qword_4F80120 = (__int64)nullsub_23;
  qword_4F80118 = (__int64)sub_984030;
  sub_C53080(&qword_4F80060, "imply-fp-condition", 18);
  LOWORD(qword_4F800F8) = 257;
  LOBYTE(qword_4F800E8) = 1;
  LOBYTE(dword_4F8006C) = dword_4F8006C & 0x9F | 0x20;
  sub_C53130(&qword_4F80060);
  __cxa_atexit(sub_984900, &qword_4F80060, &qword_4A427C0);
  qword_4F7FF80 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F7FFD0 = 0x100000000LL;
  dword_4F7FF8C &= 0x8000u;
  word_4F7FF90 = 0;
  qword_4F7FF98 = 0;
  qword_4F7FFA0 = 0;
  dword_4F7FF88 = v4;
  qword_4F7FFA8 = 0;
  qword_4F7FFB0 = 0;
  qword_4F7FFB8 = 0;
  qword_4F7FFC0 = 0;
  qword_4F7FFC8 = (__int64)&unk_4F7FFD8;
  qword_4F7FFE0 = 0;
  qword_4F7FFE8 = (__int64)&unk_4F80000;
  qword_4F7FFF0 = 1;
  dword_4F7FFF8 = 0;
  byte_4F7FFFC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F7FFD0;
  if ( (unsigned __int64)(unsigned int)qword_4F7FFD0 + 1 > HIDWORD(qword_4F7FFD0) )
  {
    v12 = v5;
    sub_C8D5F0((char *)&unk_4F7FFD8 - 16, &unk_4F7FFD8, (unsigned int)qword_4F7FFD0 + 1LL, 8);
    v6 = (unsigned int)qword_4F7FFD0;
    v5 = v12;
  }
  *(_QWORD *)(qword_4F7FFC8 + 8 * v6) = v5;
  qword_4F80010 = (__int64)&unk_49D9748;
  qword_4F7FF80 = (__int64)&unk_49DC090;
  LODWORD(qword_4F7FFD0) = qword_4F7FFD0 + 1;
  qword_4F80008 = 0;
  qword_4F80020 = (__int64)&unk_49DC1D0;
  qword_4F80018 = 0;
  qword_4F80040 = (__int64)nullsub_23;
  qword_4F80038 = (__int64)sub_984030;
  sub_C53080(&qword_4F7FF80, "extend-compute-range-analysis", 29);
  LOBYTE(qword_4F80008) = 0;
  qword_4F7FFB0 = 68;
  LOBYTE(dword_4F7FF8C) = dword_4F7FF8C & 0x9F | 0x20;
  LOWORD(qword_4F80018) = 256;
  qword_4F7FFA8 = (__int64)"Extend computeConstantRange for BinaryOperator and InsertElementInst";
  sub_C53130(&qword_4F7FF80);
  __cxa_atexit(sub_984900, &qword_4F7FF80, &qword_4A427C0);
  qword_4F7FEA0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F7FF1C = 1;
  qword_4F7FEF0 = 0x100000000LL;
  dword_4F7FEAC &= 0x8000u;
  qword_4F7FEB8 = 0;
  qword_4F7FEC0 = 0;
  qword_4F7FEC8 = 0;
  dword_4F7FEA8 = v7;
  word_4F7FEB0 = 0;
  qword_4F7FED0 = 0;
  qword_4F7FED8 = 0;
  qword_4F7FEE0 = 0;
  qword_4F7FEE8 = (__int64)&unk_4F7FEF8;
  qword_4F7FF00 = 0;
  qword_4F7FF08 = (__int64)&unk_4F7FF20;
  qword_4F7FF10 = 1;
  dword_4F7FF18 = 0;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4F7FEF0;
  v10 = (unsigned int)qword_4F7FEF0 + 1LL;
  if ( v10 > HIDWORD(qword_4F7FEF0) )
  {
    sub_C8D5F0((char *)&unk_4F7FEF8 - 16, &unk_4F7FEF8, v10, 8);
    v9 = (unsigned int)qword_4F7FEF0;
  }
  *(_QWORD *)(qword_4F7FEE8 + 8 * v9) = v8;
  LODWORD(qword_4F7FEF0) = qword_4F7FEF0 + 1;
  qword_4F7FF28 = 0;
  qword_4F7FF30 = (__int64)&unk_49D9728;
  qword_4F7FF38 = 0;
  qword_4F7FEA0 = (__int64)&unk_49DBF10;
  qword_4F7FF40 = (__int64)&unk_49DC290;
  qword_4F7FF60 = (__int64)nullsub_24;
  qword_4F7FF58 = (__int64)sub_984050;
  sub_C53080(&qword_4F7FEA0, "dom-conditions-max-uses", 23);
  LODWORD(qword_4F7FF28) = 20;
  BYTE4(qword_4F7FF38) = 1;
  LODWORD(qword_4F7FF38) = 20;
  LOBYTE(dword_4F7FEAC) = dword_4F7FEAC & 0x9F | 0x20;
  sub_C53130(&qword_4F7FEA0);
  return __cxa_atexit(sub_984970, &qword_4F7FEA0, &qword_4A427C0);
}
