// Function: ctor_563
// Address: 0x572660
//
int ctor_563()
{
  __int64 v0; // rax
  __int64 v1; // r12
  int v2; // edx
  __int64 v3; // r12
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  int v6; // edx
  __int64 v7; // rbx
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  _QWORD v11[2]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v12[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v13[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v14[8]; // [rsp+30h] [rbp-40h] BYREF

  v0 = sub_C60B10();
  v13[0] = v14;
  v1 = v0;
  sub_2E44A00(v13, "Controls which register COPYs are forwarded");
  v11[0] = v12;
  sub_2E44A00(v11, "machine-cp-fwd");
  sub_CF9810(v1, v11, v13);
  if ( (_QWORD *)v11[0] != v12 )
    j_j___libc_free_0(v11[0], v12[0] + 1LL);
  if ( (_QWORD *)v13[0] != v14 )
    j_j___libc_free_0(v13[0], v14[0] + 1LL);
  qword_501F480 = (__int64)&unk_49DC150;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_501F4D0 = 0x100000000LL;
  word_501F490 = 0;
  dword_501F48C &= 0x8000u;
  qword_501F498 = 0;
  qword_501F4A0 = 0;
  dword_501F488 = v2;
  qword_501F4A8 = 0;
  qword_501F4B0 = 0;
  qword_501F4B8 = 0;
  qword_501F4C0 = 0;
  qword_501F4C8 = (__int64)&unk_501F4D8;
  qword_501F4E0 = 0;
  qword_501F4E8 = (__int64)&unk_501F500;
  qword_501F4F0 = 1;
  dword_501F4F8 = 0;
  byte_501F4FC = 1;
  v3 = sub_C57470();
  v4 = (unsigned int)qword_501F4D0;
  v5 = (unsigned int)qword_501F4D0 + 1LL;
  if ( v5 > HIDWORD(qword_501F4D0) )
  {
    sub_C8D5F0((char *)&unk_501F4D8 - 16, &unk_501F4D8, v5, 8);
    v4 = (unsigned int)qword_501F4D0;
  }
  *(_QWORD *)(qword_501F4C8 + 8 * v4) = v3;
  LODWORD(qword_501F4D0) = qword_501F4D0 + 1;
  qword_501F508 = 0;
  qword_501F510 = (__int64)&unk_49D9748;
  qword_501F518 = 0;
  qword_501F480 = (__int64)&unk_49DC090;
  qword_501F520 = (__int64)&unk_49DC1D0;
  qword_501F540 = (__int64)nullsub_23;
  qword_501F538 = (__int64)sub_984030;
  sub_C53080(&qword_501F480, "mcp-use-is-copy-instr", 21);
  LOBYTE(qword_501F508) = 0;
  LOWORD(qword_501F518) = 256;
  LOBYTE(dword_501F48C) = dword_501F48C & 0x9F | 0x20;
  sub_C53130(&qword_501F480);
  __cxa_atexit(sub_984900, &qword_501F480, &qword_4A427C0);
  qword_501F3A0 = (__int64)&unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_501F41C = 1;
  qword_501F3F0 = 0x100000000LL;
  dword_501F3AC &= 0x8000u;
  qword_501F3B8 = 0;
  qword_501F3C0 = 0;
  qword_501F3C8 = 0;
  dword_501F3A8 = v6;
  word_501F3B0 = 0;
  qword_501F3D0 = 0;
  qword_501F3D8 = 0;
  qword_501F3E0 = 0;
  qword_501F3E8 = (__int64)&unk_501F3F8;
  qword_501F400 = 0;
  qword_501F408 = (__int64)&unk_501F420;
  qword_501F410 = 1;
  dword_501F418 = 0;
  v7 = sub_C57470();
  v8 = (unsigned int)qword_501F3F0;
  v9 = (unsigned int)qword_501F3F0 + 1LL;
  if ( v9 > HIDWORD(qword_501F3F0) )
  {
    sub_C8D5F0((char *)&unk_501F3F8 - 16, &unk_501F3F8, v9, 8);
    v8 = (unsigned int)qword_501F3F0;
  }
  *(_QWORD *)(qword_501F3E8 + 8 * v8) = v7;
  LODWORD(qword_501F3F0) = qword_501F3F0 + 1;
  qword_501F428 = 0;
  qword_501F430 = (__int64)&unk_49DC110;
  qword_501F438 = 0;
  qword_501F3A0 = (__int64)&unk_49D97F0;
  qword_501F440 = (__int64)&unk_49DC200;
  qword_501F460 = (__int64)nullsub_26;
  qword_501F458 = (__int64)sub_9C26D0;
  sub_C53080(&qword_501F3A0, "enable-spill-copy-elim", 22);
  LOBYTE(dword_501F3AC) = dword_501F3AC & 0x9F | 0x20;
  sub_C53130(&qword_501F3A0);
  return __cxa_atexit(sub_9C44F0, &qword_501F3A0, &qword_4A427C0);
}
