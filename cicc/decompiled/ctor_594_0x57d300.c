// Function: ctor_594
// Address: 0x57d300
//
int __fastcall ctor_594(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_5025FC0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  dword_5025FCC &= 0x8000u;
  word_5025FD0 = 0;
  qword_5026010 = 0x100000000LL;
  qword_5025FD8 = 0;
  qword_5025FE0 = 0;
  qword_5025FE8 = 0;
  dword_5025FC8 = v4;
  qword_5025FF0 = 0;
  qword_5025FF8 = 0;
  qword_5026000 = 0;
  qword_5026008 = (__int64)&unk_5026018;
  qword_5026020 = 0;
  qword_5026028 = (__int64)&unk_5026040;
  qword_5026030 = 1;
  dword_5026038 = 0;
  byte_502603C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5026010;
  v7 = (unsigned int)qword_5026010 + 1LL;
  if ( v7 > HIDWORD(qword_5026010) )
  {
    sub_C8D5F0((char *)&unk_5026018 - 16, &unk_5026018, v7, 8);
    v6 = (unsigned int)qword_5026010;
  }
  *(_QWORD *)(qword_5026008 + 8 * v6) = v5;
  LODWORD(qword_5026010) = qword_5026010 + 1;
  qword_5026048 = 0;
  qword_5026050 = (__int64)&unk_49DA090;
  qword_5026058 = 0;
  qword_5025FC0 = (__int64)&unk_49DBF90;
  qword_5026060 = (__int64)&unk_49DC230;
  qword_5026080 = (__int64)nullsub_58;
  qword_5026078 = (__int64)sub_B2B5F0;
  sub_C53080(&qword_5025FC0, "stackmap-version", 16);
  LODWORD(qword_5026048) = 3;
  BYTE4(qword_5026058) = 1;
  LODWORD(qword_5026058) = 3;
  qword_5025FF0 = 51;
  LOBYTE(dword_5025FCC) = dword_5025FCC & 0x9F | 0x20;
  qword_5025FE8 = (__int64)"Specify the stackmap encoding version (default = 3)";
  sub_C53130(&qword_5025FC0);
  return __cxa_atexit(sub_B2B680, &qword_5025FC0, &qword_4A427C0);
}
