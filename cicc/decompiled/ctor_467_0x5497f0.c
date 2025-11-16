// Function: ctor_467
// Address: 0x5497f0
//
int ctor_467()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_5000560 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50005B0 = 0x100000000LL;
  word_5000570 = 0;
  dword_500056C &= 0x8000u;
  qword_5000578 = 0;
  qword_5000580 = 0;
  dword_5000568 = v0;
  qword_5000588 = 0;
  qword_5000590 = 0;
  qword_5000598 = 0;
  qword_50005A0 = 0;
  qword_50005A8 = (__int64)&unk_50005B8;
  qword_50005C0 = 0;
  qword_50005C8 = (__int64)&unk_50005E0;
  qword_50005D0 = 1;
  dword_50005D8 = 0;
  byte_50005DC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_50005B0;
  v3 = (unsigned int)qword_50005B0 + 1LL;
  if ( v3 > HIDWORD(qword_50005B0) )
  {
    sub_C8D5F0((char *)&unk_50005B8 - 16, &unk_50005B8, v3, 8);
    v2 = (unsigned int)qword_50005B0;
  }
  *(_QWORD *)(qword_50005A8 + 8 * v2) = v1;
  LODWORD(qword_50005B0) = qword_50005B0 + 1;
  qword_50005E8 = 0;
  qword_50005F0 = (__int64)&unk_49D9728;
  qword_50005F8 = 0;
  qword_5000560 = (__int64)&unk_49DBF10;
  qword_5000600 = (__int64)&unk_49DC290;
  qword_5000620 = (__int64)nullsub_24;
  qword_5000618 = (__int64)sub_984050;
  sub_C53080(&qword_5000560, "rotation-max-header-size", 24);
  LODWORD(qword_50005E8) = 16;
  BYTE4(qword_50005F8) = 1;
  LODWORD(qword_50005F8) = 16;
  qword_5000590 = 59;
  LOBYTE(dword_500056C) = dword_500056C & 0x9F | 0x20;
  qword_5000588 = (__int64)"The default maximum header size for automatic loop rotation";
  sub_C53130(&qword_5000560);
  __cxa_atexit(sub_984970, &qword_5000560, &qword_4A427C0);
  qword_5000480 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_50004FC = 1;
  qword_50004D0 = 0x100000000LL;
  dword_500048C &= 0x8000u;
  qword_5000498 = 0;
  qword_50004A0 = 0;
  qword_50004A8 = 0;
  dword_5000488 = v4;
  word_5000490 = 0;
  qword_50004B0 = 0;
  qword_50004B8 = 0;
  qword_50004C0 = 0;
  qword_50004C8 = (__int64)&unk_50004D8;
  qword_50004E0 = 0;
  qword_50004E8 = (__int64)&unk_5000500;
  qword_50004F0 = 1;
  dword_50004F8 = 0;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50004D0;
  v7 = (unsigned int)qword_50004D0 + 1LL;
  if ( v7 > HIDWORD(qword_50004D0) )
  {
    sub_C8D5F0((char *)&unk_50004D8 - 16, &unk_50004D8, v7, 8);
    v6 = (unsigned int)qword_50004D0;
  }
  *(_QWORD *)(qword_50004C8 + 8 * v6) = v5;
  LODWORD(qword_50004D0) = qword_50004D0 + 1;
  qword_5000508 = 0;
  qword_5000510 = (__int64)&unk_49D9748;
  qword_5000518 = 0;
  qword_5000480 = (__int64)&unk_49DC090;
  qword_5000520 = (__int64)&unk_49DC1D0;
  qword_5000540 = (__int64)nullsub_23;
  qword_5000538 = (__int64)sub_984030;
  sub_C53080(&qword_5000480, "rotation-prepare-for-lto", 24);
  LOBYTE(qword_5000508) = 0;
  LOWORD(qword_5000518) = 256;
  qword_50004B0 = 92;
  LOBYTE(dword_500048C) = dword_500048C & 0x9F | 0x20;
  qword_50004A8 = (__int64)"Run loop-rotation in the prepare-for-lto stage. This option should be used for testing only.";
  sub_C53130(&qword_5000480);
  return __cxa_atexit(sub_984900, &qword_5000480, &qword_4A427C0);
}
