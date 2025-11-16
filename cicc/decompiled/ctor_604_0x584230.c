// Function: ctor_604
// Address: 0x584230
//
int ctor_604()
{
  __int64 v0; // rdx
  __int64 v1; // rcx
  int v2; // edx
  __int64 v3; // rbx
  __int64 v4; // rax
  unsigned __int64 v5; // rdx
  char v7; // [rsp+3h] [rbp-4Dh] BYREF
  int v8; // [rsp+4h] [rbp-4Ch] BYREF
  char *v9; // [rsp+8h] [rbp-48h] BYREF
  const char *v10; // [rsp+10h] [rbp-40h] BYREF
  __int64 v11; // [rsp+18h] [rbp-38h]

  v10 = "Clone multicolor basic blocks but do not demote cross scopes";
  v7 = 0;
  v9 = &v7;
  v11 = 60;
  v8 = 1;
  sub_3012570(&unk_502A840, "disable-demotion", &v8, &v10, &v9);
  __cxa_atexit(sub_984900, &unk_502A840, &qword_4A427C0);
  v7 = 0;
  v10 = "Do not remove implausible terminators or other similar cleanups";
  v9 = &v7;
  v11 = 63;
  v8 = 1;
  sub_3012570(&unk_502A760, "disable-cleanups", &v8, &v10, &v9);
  __cxa_atexit(sub_984900, &unk_502A760, &qword_4A427C0);
  qword_502A680 = (__int64)&unk_49DC150;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &unk_502A760, v0, v1), 1u);
  byte_502A6FC = 1;
  qword_502A6D0 = 0x100000000LL;
  dword_502A68C &= 0x8000u;
  qword_502A698 = 0;
  qword_502A6A0 = 0;
  qword_502A6A8 = 0;
  dword_502A688 = v2;
  word_502A690 = 0;
  qword_502A6B0 = 0;
  qword_502A6B8 = 0;
  qword_502A6C0 = 0;
  qword_502A6C8 = (__int64)&unk_502A6D8;
  qword_502A6E0 = 0;
  qword_502A6E8 = (__int64)&unk_502A700;
  qword_502A6F0 = 1;
  dword_502A6F8 = 0;
  v3 = sub_C57470();
  v4 = (unsigned int)qword_502A6D0;
  v5 = (unsigned int)qword_502A6D0 + 1LL;
  if ( v5 > HIDWORD(qword_502A6D0) )
  {
    sub_C8D5F0((char *)&unk_502A6D8 - 16, &unk_502A6D8, v5, 8);
    v4 = (unsigned int)qword_502A6D0;
  }
  *(_QWORD *)(qword_502A6C8 + 8 * v4) = v3;
  LODWORD(qword_502A6D0) = qword_502A6D0 + 1;
  qword_502A708 = 0;
  qword_502A710 = (__int64)&unk_49D9748;
  qword_502A718 = 0;
  qword_502A680 = (__int64)&unk_49DC090;
  qword_502A720 = (__int64)&unk_49DC1D0;
  qword_502A740 = (__int64)nullsub_23;
  qword_502A738 = (__int64)sub_984030;
  sub_C53080(&qword_502A680, "demote-catchswitch-only", 23);
  qword_502A6B0 = 41;
  LOBYTE(qword_502A708) = 0;
  LOBYTE(dword_502A68C) = dword_502A68C & 0x9F | 0x20;
  qword_502A6A8 = (__int64)"Demote catchswitch BBs only (for wasm EH)";
  LOWORD(qword_502A718) = 256;
  sub_C53130(&qword_502A680);
  return __cxa_atexit(sub_984900, &qword_502A680, &qword_4A427C0);
}
