// Function: ctor_504
// Address: 0x55ae30
//
int ctor_504()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // rdx
  int v7; // edx
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v12; // [rsp+8h] [rbp-58h]
  int v13; // [rsp+1Ch] [rbp-44h] BYREF
  const char *v14; // [rsp+20h] [rbp-40h] BYREF
  __int64 v15; // [rsp+28h] [rbp-38h]

  qword_500AD80 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_500ADD0 = 0x100000000LL;
  dword_500AD8C &= 0x8000u;
  word_500AD90 = 0;
  qword_500AD98 = 0;
  qword_500ADA0 = 0;
  dword_500AD88 = v0;
  qword_500ADA8 = 0;
  qword_500ADB0 = 0;
  qword_500ADB8 = 0;
  qword_500ADC0 = 0;
  qword_500ADC8 = (__int64)&unk_500ADD8;
  qword_500ADE0 = 0;
  qword_500ADE8 = (__int64)&unk_500AE00;
  qword_500ADF0 = 1;
  dword_500ADF8 = 0;
  byte_500ADFC = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_500ADD0;
  v3 = (unsigned int)qword_500ADD0 + 1LL;
  if ( v3 > HIDWORD(qword_500ADD0) )
  {
    sub_C8D5F0((char *)&unk_500ADD8 - 16, &unk_500ADD8, v3, 8);
    v2 = (unsigned int)qword_500ADD0;
  }
  *(_QWORD *)(qword_500ADC8 + 8 * v2) = v1;
  qword_500AE08 = (__int64)&byte_500AE18;
  qword_500AE30 = (__int64)&byte_500AE40;
  qword_500AE28 = (__int64)&unk_49DC130;
  qword_500AD80 = (__int64)&unk_49DC010;
  LODWORD(qword_500ADD0) = qword_500ADD0 + 1;
  qword_500AE10 = 0;
  qword_500AE58 = (__int64)&unk_49DC350;
  byte_500AE18 = 0;
  qword_500AE78 = (__int64)nullsub_92;
  qword_500AE38 = 0;
  qword_500AE70 = (__int64)sub_BC4D70;
  byte_500AE40 = 0;
  byte_500AE50 = 0;
  sub_C53080(&qword_500AD80, "rename-exclude-function-prefixes", 32);
  qword_500ADB0 = 74;
  qword_500ADA8 = (__int64)"Prefixes for functions that don't need to be renamed, separated by a comma";
  LOBYTE(dword_500AD8C) = dword_500AD8C & 0x9F | 0x20;
  sub_C53130(&qword_500AD80);
  __cxa_atexit(sub_BC5A40, &qword_500AD80, &qword_4A427C0);
  qword_500AC80 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_500ACD0 = 0x100000000LL;
  word_500AC90 = 0;
  dword_500AC8C &= 0x8000u;
  qword_500AC98 = 0;
  qword_500ACA0 = 0;
  dword_500AC88 = v4;
  qword_500ACA8 = 0;
  qword_500ACB0 = 0;
  qword_500ACB8 = 0;
  qword_500ACC0 = 0;
  qword_500ACC8 = (__int64)&unk_500ACD8;
  qword_500ACE0 = 0;
  qword_500ACE8 = (__int64)&unk_500AD00;
  qword_500ACF0 = 1;
  dword_500ACF8 = 0;
  byte_500ACFC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_500ACD0;
  if ( (unsigned __int64)(unsigned int)qword_500ACD0 + 1 > HIDWORD(qword_500ACD0) )
  {
    v12 = v5;
    sub_C8D5F0((char *)&unk_500ACD8 - 16, &unk_500ACD8, (unsigned int)qword_500ACD0 + 1LL, 8);
    v6 = (unsigned int)qword_500ACD0;
    v5 = v12;
  }
  *(_QWORD *)(qword_500ACC8 + 8 * v6) = v5;
  qword_500AD08 = (__int64)&byte_500AD18;
  qword_500AD30 = (__int64)&byte_500AD40;
  qword_500AD28 = (__int64)&unk_49DC130;
  qword_500AC80 = (__int64)&unk_49DC010;
  LODWORD(qword_500ACD0) = qword_500ACD0 + 1;
  qword_500AD10 = 0;
  qword_500AD58 = (__int64)&unk_49DC350;
  byte_500AD18 = 0;
  qword_500AD78 = (__int64)nullsub_92;
  qword_500AD38 = 0;
  qword_500AD70 = (__int64)sub_BC4D70;
  byte_500AD40 = 0;
  byte_500AD50 = 0;
  sub_C53080(&qword_500AC80, "rename-exclude-alias-prefixes", 29);
  qword_500ACB0 = 72;
  qword_500ACA8 = (__int64)"Prefixes for aliases that don't need to be renamed, separated by a comma";
  LOBYTE(dword_500AC8C) = dword_500AC8C & 0x9F | 0x20;
  sub_C53130(&qword_500AC80);
  __cxa_atexit(sub_BC5A40, &qword_500AC80, &qword_4A427C0);
  v14 = "Prefixes for global values that don't need to be renamed, separated by a comma";
  v13 = 1;
  v15 = 78;
  sub_2A3DD90(&unk_500AB80, "rename-exclude-global-prefixes", &v14, &v13);
  __cxa_atexit(sub_BC5A40, &unk_500AB80, &qword_4A427C0);
  v14 = "Prefixes for structs that don't need to be renamed, separated by a comma";
  v13 = 1;
  v15 = 72;
  sub_2A3DD90(&unk_500AA80, "rename-exclude-struct-prefixes", &v14, &v13);
  __cxa_atexit(sub_BC5A40, &unk_500AA80, &qword_4A427C0);
  qword_500A9A0 = (__int64)&unk_49DC150;
  v7 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_500AA1C = 1;
  qword_500A9F0 = 0x100000000LL;
  dword_500A9AC &= 0x8000u;
  qword_500A9B8 = 0;
  qword_500A9C0 = 0;
  qword_500A9C8 = 0;
  dword_500A9A8 = v7;
  word_500A9B0 = 0;
  qword_500A9D0 = 0;
  qword_500A9D8 = 0;
  qword_500A9E0 = 0;
  qword_500A9E8 = (__int64)&unk_500A9F8;
  qword_500AA00 = 0;
  qword_500AA08 = (__int64)&unk_500AA20;
  qword_500AA10 = 1;
  dword_500AA18 = 0;
  v8 = sub_C57470();
  v9 = (unsigned int)qword_500A9F0;
  v10 = (unsigned int)qword_500A9F0 + 1LL;
  if ( v10 > HIDWORD(qword_500A9F0) )
  {
    sub_C8D5F0((char *)&unk_500A9F8 - 16, &unk_500A9F8, v10, 8);
    v9 = (unsigned int)qword_500A9F0;
  }
  *(_QWORD *)(qword_500A9E8 + 8 * v9) = v8;
  LODWORD(qword_500A9F0) = qword_500A9F0 + 1;
  qword_500AA28 = 0;
  qword_500AA30 = (__int64)&unk_49D9748;
  qword_500AA38 = 0;
  qword_500A9A0 = (__int64)&unk_49DC090;
  qword_500AA40 = (__int64)&unk_49DC1D0;
  qword_500AA60 = (__int64)nullsub_23;
  qword_500AA58 = (__int64)sub_984030;
  sub_C53080(&qword_500A9A0, "rename-only-inst", 16);
  LOBYTE(qword_500AA28) = 0;
  LOWORD(qword_500AA38) = 256;
  qword_500A9C8 = (__int64)"only rename the instructions in the function";
  qword_500A9D0 = 44;
  LOBYTE(dword_500A9AC) = dword_500A9AC & 0x9F | 0x20;
  sub_C53130(&qword_500A9A0);
  return __cxa_atexit(sub_984900, &qword_500A9A0, &qword_4A427C0);
}
