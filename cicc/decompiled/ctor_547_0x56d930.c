// Function: ctor_547
// Address: 0x56d930
//
int ctor_547()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx

  qword_501D140 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_501D14C &= 0x8000u;
  word_501D150 = 0;
  qword_501D190 = 0x100000000LL;
  qword_501D158 = 0;
  qword_501D160 = 0;
  qword_501D168 = 0;
  dword_501D148 = v0;
  qword_501D170 = 0;
  qword_501D178 = 0;
  qword_501D180 = 0;
  qword_501D188 = (__int64)&unk_501D198;
  qword_501D1A0 = 0;
  qword_501D1A8 = (__int64)&unk_501D1C0;
  qword_501D1B0 = 1;
  dword_501D1B8 = 0;
  byte_501D1BC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_501D190;
  v3 = (unsigned int)qword_501D190 + 1LL;
  if ( v3 > HIDWORD(qword_501D190) )
  {
    sub_C8D5F0((char *)&unk_501D198 - 16, &unk_501D198, v3, 8);
    v2 = (unsigned int)qword_501D190;
  }
  *(_QWORD *)(qword_501D188 + 8 * v2) = v1;
  LODWORD(qword_501D190) = qword_501D190 + 1;
  qword_501D1C8 = 0;
  qword_501D1D0 = (__int64)&unk_49D9748;
  qword_501D1D8 = 0;
  qword_501D140 = (__int64)&unk_49DC090;
  qword_501D1E0 = (__int64)&unk_49DC1D0;
  qword_501D200 = (__int64)nullsub_23;
  qword_501D1F8 = (__int64)sub_984030;
  sub_C53080(&qword_501D140, "view-edge-bundles", 17);
  qword_501D170 = 42;
  LOBYTE(dword_501D14C) = dword_501D14C & 0x9F | 0x20;
  qword_501D168 = (__int64)"Pop up a window to show edge bundle graphs";
  sub_C53130(&qword_501D140);
  return __cxa_atexit(sub_984900, &qword_501D140, &qword_4A427C0);
}
