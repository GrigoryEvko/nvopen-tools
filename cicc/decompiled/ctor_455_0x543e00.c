// Function: ctor_455
// Address: 0x543e00
//
int ctor_455()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_4FFD820 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FFD89C = 1;
  qword_4FFD870 = 0x100000000LL;
  dword_4FFD82C &= 0x8000u;
  qword_4FFD838 = 0;
  qword_4FFD840 = 0;
  qword_4FFD848 = 0;
  dword_4FFD828 = v0;
  word_4FFD830 = 0;
  qword_4FFD850 = 0;
  qword_4FFD858 = 0;
  qword_4FFD860 = 0;
  qword_4FFD868 = (__int64)&unk_4FFD878;
  qword_4FFD880 = 0;
  qword_4FFD888 = (__int64)&unk_4FFD8A0;
  qword_4FFD890 = 1;
  dword_4FFD898 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FFD870;
  v3 = (unsigned int)qword_4FFD870 + 1LL;
  if ( v3 > HIDWORD(qword_4FFD870) )
  {
    sub_C8D5F0((char *)&unk_4FFD878 - 16, &unk_4FFD878, v3, 8);
    v2 = (unsigned int)qword_4FFD870;
  }
  *(_QWORD *)(qword_4FFD868 + 8 * v2) = v1;
  qword_4FFD8B0 = (__int64)&unk_49D9728;
  LODWORD(qword_4FFD870) = qword_4FFD870 + 1;
  qword_4FFD8A8 = 0;
  qword_4FFD820 = (__int64)&unk_49DBF10;
  qword_4FFD8C0 = (__int64)&unk_49DC290;
  qword_4FFD8B8 = 0;
  qword_4FFD8E0 = (__int64)nullsub_24;
  qword_4FFD8D8 = (__int64)sub_984050;
  sub_C53080(&qword_4FFD820, "jump-table-to-switch-size-threshold", 35);
  qword_4FFD850 = 75;
  LODWORD(qword_4FFD8A8) = 10;
  BYTE4(qword_4FFD8B8) = 1;
  LODWORD(qword_4FFD8B8) = 10;
  LOBYTE(dword_4FFD82C) = dword_4FFD82C & 0x9F | 0x20;
  qword_4FFD848 = (__int64)"Only split jump tables with size less or equal than JumpTableSizeThreshold.";
  sub_C53130(&qword_4FFD820);
  __cxa_atexit(sub_984970, &qword_4FFD820, &qword_4A427C0);
  qword_4FFD740 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FFD74C &= 0x8000u;
  word_4FFD750 = 0;
  qword_4FFD790 = 0x100000000LL;
  qword_4FFD758 = 0;
  qword_4FFD760 = 0;
  qword_4FFD768 = 0;
  dword_4FFD748 = v4;
  qword_4FFD770 = 0;
  qword_4FFD778 = 0;
  qword_4FFD780 = 0;
  qword_4FFD788 = (__int64)&unk_4FFD798;
  qword_4FFD7A0 = 0;
  qword_4FFD7A8 = (__int64)&unk_4FFD7C0;
  qword_4FFD7B0 = 1;
  dword_4FFD7B8 = 0;
  byte_4FFD7BC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FFD790;
  v7 = (unsigned int)qword_4FFD790 + 1LL;
  if ( v7 > HIDWORD(qword_4FFD790) )
  {
    sub_C8D5F0((char *)&unk_4FFD798 - 16, &unk_4FFD798, v7, 8);
    v6 = (unsigned int)qword_4FFD790;
  }
  *(_QWORD *)(qword_4FFD788 + 8 * v6) = v5;
  qword_4FFD7D0 = (__int64)&unk_49D9728;
  LODWORD(qword_4FFD790) = qword_4FFD790 + 1;
  qword_4FFD7C8 = 0;
  qword_4FFD740 = (__int64)&unk_49DBF10;
  qword_4FFD7E0 = (__int64)&unk_49DC290;
  qword_4FFD7D8 = 0;
  qword_4FFD800 = (__int64)nullsub_24;
  qword_4FFD7F8 = (__int64)sub_984050;
  sub_C53080(&qword_4FFD740, "jump-table-to-switch-function-size-threshold", 44);
  qword_4FFD770 = 94;
  LODWORD(qword_4FFD7C8) = 50;
  BYTE4(qword_4FFD7D8) = 1;
  LODWORD(qword_4FFD7D8) = 50;
  LOBYTE(dword_4FFD74C) = dword_4FFD74C & 0x9F | 0x20;
  qword_4FFD768 = (__int64)"Only split jump tables containing functions whose sizes are less or equal than this threshold.";
  sub_C53130(&qword_4FFD740);
  return __cxa_atexit(sub_984970, &qword_4FFD740, &qword_4A427C0);
}
