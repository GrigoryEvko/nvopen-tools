// Function: ctor_608
// Address: 0x585b20
//
int __fastcall ctor_608(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_502B5E0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  byte_502B65C = 1;
  qword_502B630 = 0x100000000LL;
  dword_502B5EC &= 0x8000u;
  qword_502B5F8 = 0;
  qword_502B600 = 0;
  qword_502B608 = 0;
  dword_502B5E8 = v4;
  word_502B5F0 = 0;
  qword_502B610 = 0;
  qword_502B618 = 0;
  qword_502B620 = 0;
  qword_502B628 = (__int64)&unk_502B638;
  qword_502B640 = 0;
  qword_502B648 = (__int64)&unk_502B660;
  qword_502B650 = 1;
  dword_502B658 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_502B630;
  v7 = (unsigned int)qword_502B630 + 1LL;
  if ( v7 > HIDWORD(qword_502B630) )
  {
    sub_C8D5F0((char *)&unk_502B638 - 16, &unk_502B638, v7, 8);
    v6 = (unsigned int)qword_502B630;
  }
  *(_QWORD *)(qword_502B628 + 8 * v6) = v5;
  LODWORD(qword_502B630) = qword_502B630 + 1;
  qword_502B668 = 0;
  qword_502B670 = (__int64)&unk_49D9748;
  qword_502B678 = 0;
  qword_502B5E0 = (__int64)&unk_49DC090;
  qword_502B680 = (__int64)&unk_49DC1D0;
  qword_502B6A0 = (__int64)nullsub_23;
  qword_502B698 = (__int64)sub_984030;
  sub_C53080(&qword_502B5E0, "nvptx-no-f16-math", 17);
  qword_502B610 = 51;
  LOBYTE(qword_502B668) = 0;
  LOBYTE(dword_502B5EC) = dword_502B5EC & 0x9F | 0x20;
  qword_502B608 = (__int64)"NVPTX Specific: Disable generation of f16 math ops.";
  LOWORD(qword_502B678) = 256;
  sub_C53130(&qword_502B5E0);
  return __cxa_atexit(sub_984900, &qword_502B5E0, &qword_4A427C0);
}
