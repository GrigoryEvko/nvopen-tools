// Function: ctor_713
// Address: 0x5bf0b0
//
int __fastcall ctor_713(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_5051600 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_505167C = 1;
  qword_5051650 = 0x100000000LL;
  dword_505160C &= 0x8000u;
  qword_5051618 = 0;
  qword_5051620 = 0;
  qword_5051628 = 0;
  dword_5051608 = v4;
  word_5051610 = 0;
  qword_5051630 = 0;
  qword_5051638 = 0;
  qword_5051640 = 0;
  qword_5051648 = (__int64)&unk_5051658;
  qword_5051660 = 0;
  qword_5051668 = (__int64)&unk_5051680;
  qword_5051670 = 1;
  dword_5051678 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5051650;
  v7 = (unsigned int)qword_5051650 + 1LL;
  if ( v7 > HIDWORD(qword_5051650) )
  {
    sub_C8D5F0((char *)&unk_5051658 - 16, &unk_5051658, v7, 8);
    v6 = (unsigned int)qword_5051650;
  }
  *(_QWORD *)(qword_5051648 + 8 * v6) = v5;
  LODWORD(qword_5051650) = qword_5051650 + 1;
  qword_5051688 = 0;
  qword_5051690 = (__int64)&unk_49D9748;
  qword_5051698 = 0;
  qword_5051600 = (__int64)&unk_49DC090;
  qword_50516A0 = (__int64)&unk_49DC1D0;
  qword_50516C0 = (__int64)nullsub_23;
  qword_50516B8 = (__int64)sub_984030;
  sub_C53080(&qword_5051600, "jumptable-in-function-section", 29);
  LOBYTE(qword_5051688) = 0;
  qword_5051630 = 38;
  LOBYTE(dword_505160C) = dword_505160C & 0x9F | 0x20;
  LOWORD(qword_5051698) = 256;
  qword_5051628 = (__int64)"Putting Jump Table in function section";
  sub_C53130(&qword_5051600);
  return __cxa_atexit(sub_984900, &qword_5051600, &qword_4A427C0);
}
