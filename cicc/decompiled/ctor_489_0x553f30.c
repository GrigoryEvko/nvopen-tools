// Function: ctor_489
// Address: 0x553f30
//
int ctor_489()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_5007880 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50078D0 = 0x100000000LL;
  dword_500788C &= 0x8000u;
  word_5007890 = 0;
  qword_5007898 = 0;
  qword_50078A0 = 0;
  dword_5007888 = v0;
  qword_50078A8 = 0;
  qword_50078B0 = 0;
  qword_50078B8 = 0;
  qword_50078C0 = 0;
  qword_50078C8 = (__int64)&unk_50078D8;
  qword_50078E0 = 0;
  qword_50078E8 = (__int64)&unk_5007900;
  qword_50078F0 = 1;
  dword_50078F8 = 0;
  byte_50078FC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_50078D0;
  v3 = (unsigned int)qword_50078D0 + 1LL;
  if ( v3 > HIDWORD(qword_50078D0) )
  {
    sub_C8D5F0((char *)&unk_50078D8 - 16, &unk_50078D8, v3, 8);
    v2 = (unsigned int)qword_50078D0;
  }
  *(_QWORD *)(qword_50078C8 + 8 * v2) = v1;
  qword_5007910 = (__int64)&unk_49D9748;
  LODWORD(qword_50078D0) = qword_50078D0 + 1;
  qword_5007908 = 0;
  qword_5007880 = (__int64)&unk_49DC090;
  qword_5007920 = (__int64)&unk_49DC1D0;
  qword_5007918 = 0;
  qword_5007940 = (__int64)nullsub_23;
  qword_5007938 = (__int64)sub_984030;
  sub_C53080(&qword_5007880, "structurizecfg-skip-uniform-regions", 35);
  LOWORD(qword_5007918) = 256;
  LOBYTE(qword_5007908) = 0;
  qword_50078B0 = 59;
  LOBYTE(dword_500788C) = dword_500788C & 0x9F | 0x20;
  qword_50078A8 = (__int64)"Force whether the StructurizeCFG pass skips uniform regions";
  sub_C53130(&qword_5007880);
  __cxa_atexit(sub_984900, &qword_5007880, &qword_4A427C0);
  qword_50077A0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_50077F0 = 0x100000000LL;
  word_50077B0 = 0;
  dword_50077AC &= 0x8000u;
  qword_50077B8 = 0;
  qword_50077C0 = 0;
  dword_50077A8 = v4;
  qword_50077C8 = 0;
  qword_50077D0 = 0;
  qword_50077D8 = 0;
  qword_50077E0 = 0;
  qword_50077E8 = (__int64)&unk_50077F8;
  qword_5007800 = 0;
  qword_5007808 = (__int64)&unk_5007820;
  qword_5007810 = 1;
  dword_5007818 = 0;
  byte_500781C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50077F0;
  v7 = (unsigned int)qword_50077F0 + 1LL;
  if ( v7 > HIDWORD(qword_50077F0) )
  {
    sub_C8D5F0((char *)&unk_50077F8 - 16, &unk_50077F8, v7, 8);
    v6 = (unsigned int)qword_50077F0;
  }
  *(_QWORD *)(qword_50077E8 + 8 * v6) = v5;
  qword_5007830 = (__int64)&unk_49D9748;
  LODWORD(qword_50077F0) = qword_50077F0 + 1;
  qword_5007828 = 0;
  qword_50077A0 = (__int64)&unk_49DC090;
  qword_5007840 = (__int64)&unk_49DC1D0;
  qword_5007838 = 0;
  qword_5007860 = (__int64)nullsub_23;
  qword_5007858 = (__int64)sub_984030;
  sub_C53080(&qword_50077A0, "structurizecfg-relaxed-uniform-regions", 38);
  qword_50077D0 = 35;
  LOBYTE(qword_5007828) = 1;
  LOBYTE(dword_50077AC) = dword_50077AC & 0x9F | 0x20;
  qword_50077C8 = (__int64)"Allow relaxed uniform region checks";
  LOWORD(qword_5007838) = 257;
  sub_C53130(&qword_50077A0);
  return __cxa_atexit(sub_984900, &qword_50077A0, &qword_4A427C0);
}
