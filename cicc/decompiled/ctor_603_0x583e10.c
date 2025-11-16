// Function: ctor_603
// Address: 0x583e10
//
int __fastcall ctor_603(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx

  qword_502A580 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_502A5D0 = 0x100000000LL;
  word_502A590 = 0;
  dword_502A58C &= 0x8000u;
  qword_502A598 = 0;
  qword_502A5A0 = 0;
  dword_502A588 = v4;
  qword_502A5A8 = 0;
  qword_502A5B0 = 0;
  qword_502A5B8 = 0;
  qword_502A5C0 = 0;
  qword_502A5C8 = (__int64)&unk_502A5D8;
  qword_502A5E0 = 0;
  qword_502A5E8 = (__int64)&unk_502A600;
  qword_502A5F0 = 1;
  dword_502A5F8 = 0;
  byte_502A5FC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_502A5D0;
  v7 = (unsigned int)qword_502A5D0 + 1LL;
  if ( v7 > HIDWORD(qword_502A5D0) )
  {
    sub_C8D5F0((char *)&unk_502A5D8 - 16, &unk_502A5D8, v7, 8);
    v6 = (unsigned int)qword_502A5D0;
  }
  *(_QWORD *)(qword_502A5C8 + 8 * v6) = v5;
  LODWORD(qword_502A5D0) = qword_502A5D0 + 1;
  qword_502A608 = 0;
  qword_502A610 = (__int64)&unk_49D9748;
  qword_502A618 = 0;
  qword_502A580 = (__int64)&unk_49DC090;
  qword_502A620 = (__int64)&unk_49DC1D0;
  qword_502A640 = (__int64)nullsub_23;
  qword_502A638 = (__int64)sub_984030;
  sub_C53080(&qword_502A580, "twoaddr-reschedule", 18);
  qword_502A5B0 = 46;
  qword_502A5A8 = (__int64)"Coalesce copies by rescheduling (default=true)";
  LOWORD(qword_502A618) = 257;
  LOBYTE(qword_502A608) = 1;
  LOBYTE(dword_502A58C) = dword_502A58C & 0x9F | 0x20;
  sub_C53130(&qword_502A580);
  __cxa_atexit(sub_984900, &qword_502A580, &qword_4A427C0);
  qword_502A4A0 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_502A580, v8, v9), 1u);
  byte_502A51C = 1;
  qword_502A4F0 = 0x100000000LL;
  dword_502A4AC &= 0x8000u;
  qword_502A4B8 = 0;
  qword_502A4C0 = 0;
  qword_502A4C8 = 0;
  dword_502A4A8 = v10;
  word_502A4B0 = 0;
  qword_502A4D0 = 0;
  qword_502A4D8 = 0;
  qword_502A4E0 = 0;
  qword_502A4E8 = (__int64)&unk_502A4F8;
  qword_502A500 = 0;
  qword_502A508 = (__int64)&unk_502A520;
  qword_502A510 = 1;
  dword_502A518 = 0;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_502A4F0;
  v13 = (unsigned int)qword_502A4F0 + 1LL;
  if ( v13 > HIDWORD(qword_502A4F0) )
  {
    sub_C8D5F0((char *)&unk_502A4F8 - 16, &unk_502A4F8, v13, 8);
    v12 = (unsigned int)qword_502A4F0;
  }
  *(_QWORD *)(qword_502A4E8 + 8 * v12) = v11;
  LODWORD(qword_502A4F0) = qword_502A4F0 + 1;
  qword_502A528 = 0;
  qword_502A530 = (__int64)&unk_49D9728;
  qword_502A538 = 0;
  qword_502A4A0 = (__int64)&unk_49DBF10;
  qword_502A540 = (__int64)&unk_49DC290;
  qword_502A560 = (__int64)nullsub_24;
  qword_502A558 = (__int64)sub_984050;
  sub_C53080(&qword_502A4A0, "dataflow-edge-limit", 19);
  LODWORD(qword_502A528) = 3;
  BYTE4(qword_502A538) = 1;
  LODWORD(qword_502A538) = 3;
  qword_502A4D0 = 94;
  LOBYTE(dword_502A4AC) = dword_502A4AC & 0x9F | 0x20;
  qword_502A4C8 = (__int64)"Maximum number of dataflow edges to traverse when evaluating the benefit of commuting operands";
  sub_C53130(&qword_502A4A0);
  return __cxa_atexit(sub_984970, &qword_502A4A0, &qword_4A427C0);
}
