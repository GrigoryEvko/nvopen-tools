// Function: ctor_454
// Address: 0x543800
//
int ctor_454()
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

  qword_4FFD660 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFD6B0 = 0x100000000LL;
  dword_4FFD66C &= 0x8000u;
  word_4FFD670 = 0;
  qword_4FFD678 = 0;
  qword_4FFD680 = 0;
  dword_4FFD668 = v0;
  qword_4FFD688 = 0;
  qword_4FFD690 = 0;
  qword_4FFD698 = 0;
  qword_4FFD6A0 = 0;
  qword_4FFD6A8 = (__int64)&unk_4FFD6B8;
  qword_4FFD6C0 = 0;
  qword_4FFD6C8 = (__int64)&unk_4FFD6E0;
  qword_4FFD6D0 = 1;
  dword_4FFD6D8 = 0;
  byte_4FFD6DC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FFD6B0;
  v3 = (unsigned int)qword_4FFD6B0 + 1LL;
  if ( v3 > HIDWORD(qword_4FFD6B0) )
  {
    sub_C8D5F0((char *)&unk_4FFD6B8 - 16, &unk_4FFD6B8, v3, 8);
    v2 = (unsigned int)qword_4FFD6B0;
  }
  *(_QWORD *)(qword_4FFD6A8 + 8 * v2) = v1;
  qword_4FFD6F0 = (__int64)&unk_49D9748;
  qword_4FFD660 = (__int64)&unk_49DC090;
  qword_4FFD700 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FFD6B0) = qword_4FFD6B0 + 1;
  qword_4FFD720 = (__int64)nullsub_23;
  qword_4FFD6E8 = 0;
  qword_4FFD718 = (__int64)sub_984030;
  qword_4FFD6F8 = 0;
  sub_C53080(&qword_4FFD660, "ias-track-indir-load", 20);
  LOWORD(qword_4FFD6F8) = 257;
  LOBYTE(qword_4FFD6E8) = 1;
  qword_4FFD690 = 52;
  LOBYTE(dword_4FFD66C) = dword_4FFD66C & 0x9F | 0x20;
  qword_4FFD688 = (__int64)"Enable tracking indirect loads in InferAddressSpaces";
  sub_C53130(&qword_4FFD660);
  __cxa_atexit(sub_984900, &qword_4FFD660, &qword_4A427C0);
  qword_4FFD580 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFD5D0 = 0x100000000LL;
  dword_4FFD58C &= 0x8000u;
  qword_4FFD5C8 = (__int64)&unk_4FFD5D8;
  word_4FFD590 = 0;
  qword_4FFD598 = 0;
  dword_4FFD588 = v4;
  qword_4FFD5A0 = 0;
  qword_4FFD5A8 = 0;
  qword_4FFD5B0 = 0;
  qword_4FFD5B8 = 0;
  qword_4FFD5C0 = 0;
  qword_4FFD5E0 = 0;
  qword_4FFD5E8 = (__int64)&unk_4FFD600;
  qword_4FFD5F0 = 1;
  dword_4FFD5F8 = 0;
  byte_4FFD5FC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FFD5D0;
  if ( (unsigned __int64)(unsigned int)qword_4FFD5D0 + 1 > HIDWORD(qword_4FFD5D0) )
  {
    v12 = v5;
    sub_C8D5F0((char *)&unk_4FFD5D8 - 16, &unk_4FFD5D8, (unsigned int)qword_4FFD5D0 + 1LL, 8);
    v6 = (unsigned int)qword_4FFD5D0;
    v5 = v12;
  }
  *(_QWORD *)(qword_4FFD5C8 + 8 * v6) = v5;
  qword_4FFD610 = (__int64)&unk_49D9748;
  qword_4FFD580 = (__int64)&unk_49DC090;
  qword_4FFD620 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FFD5D0) = qword_4FFD5D0 + 1;
  qword_4FFD640 = (__int64)nullsub_23;
  qword_4FFD608 = 0;
  qword_4FFD638 = (__int64)sub_984030;
  qword_4FFD618 = 0;
  sub_C53080(&qword_4FFD580, "ias-track-int2ptr", 17);
  LOWORD(qword_4FFD618) = 257;
  LOBYTE(qword_4FFD608) = 1;
  qword_4FFD5B0 = 46;
  LOBYTE(dword_4FFD58C) = dword_4FFD58C & 0x9F | 0x20;
  qword_4FFD5A8 = (__int64)"Enable tracking IntToPtr in InferAddressSpaces";
  sub_C53130(&qword_4FFD580);
  __cxa_atexit(sub_984900, &qword_4FFD580, &qword_4A427C0);
  qword_4FFD4A0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FFD4F0 = 0x100000000LL;
  dword_4FFD4AC &= 0x8000u;
  word_4FFD4B0 = 0;
  qword_4FFD4E8 = (__int64)&unk_4FFD4F8;
  qword_4FFD4B8 = 0;
  dword_4FFD4A8 = v7;
  qword_4FFD4C0 = 0;
  qword_4FFD4C8 = 0;
  qword_4FFD4D0 = 0;
  qword_4FFD4D8 = 0;
  qword_4FFD4E0 = 0;
  qword_4FFD500 = 0;
  qword_4FFD508 = (__int64)&unk_4FFD520;
  qword_4FFD510 = 1;
  dword_4FFD518 = 0;
  byte_4FFD51C = 1;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_4FFD4F0;
  v10 = (unsigned int)qword_4FFD4F0 + 1LL;
  if ( v10 > HIDWORD(qword_4FFD4F0) )
  {
    sub_C8D5F0((char *)&unk_4FFD4F8 - 16, &unk_4FFD4F8, v10, 8);
    v9 = (unsigned int)qword_4FFD4F0;
  }
  *(_QWORD *)(qword_4FFD4E8 + 8 * v9) = v8;
  qword_4FFD530 = (__int64)&unk_49D9748;
  qword_4FFD4A0 = (__int64)&unk_49DC090;
  qword_4FFD540 = (__int64)&unk_49DC1D0;
  LODWORD(qword_4FFD4F0) = qword_4FFD4F0 + 1;
  qword_4FFD560 = (__int64)nullsub_23;
  qword_4FFD528 = 0;
  qword_4FFD558 = (__int64)sub_984030;
  qword_4FFD538 = 0;
  sub_C53080(&qword_4FFD4A0, "assume-default-is-flat-addrspace", 32);
  LOBYTE(qword_4FFD528) = 0;
  LOWORD(qword_4FFD538) = 256;
  qword_4FFD4D0 = 96;
  LOBYTE(dword_4FFD4AC) = dword_4FFD4AC & 0x9F | 0x40;
  qword_4FFD4C8 = (__int64)"The default address space is assumed as the flat address space. This is mainly for test purpose.";
  sub_C53130(&qword_4FFD4A0);
  return __cxa_atexit(sub_984900, &qword_4FFD4A0, &qword_4A427C0);
}
