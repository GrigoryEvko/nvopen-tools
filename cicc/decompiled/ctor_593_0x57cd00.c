// Function: ctor_593
// Address: 0x57cd00
//
int __fastcall ctor_593(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rcx
  int v15; // edx
  __int64 v16; // r15
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  __int64 v20; // [rsp+8h] [rbp-38h]

  qword_5025EE0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_5025F30 = 0x100000000LL;
  dword_5025EEC &= 0x8000u;
  word_5025EF0 = 0;
  qword_5025EF8 = 0;
  qword_5025F00 = 0;
  dword_5025EE8 = v4;
  qword_5025F08 = 0;
  qword_5025F10 = 0;
  qword_5025F18 = 0;
  qword_5025F20 = 0;
  qword_5025F28 = (__int64)&unk_5025F38;
  qword_5025F40 = 0;
  qword_5025F48 = (__int64)&unk_5025F60;
  qword_5025F50 = 1;
  dword_5025F58 = 0;
  byte_5025F5C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5025F30;
  v7 = (unsigned int)qword_5025F30 + 1LL;
  if ( v7 > HIDWORD(qword_5025F30) )
  {
    sub_C8D5F0((char *)&unk_5025F38 - 16, &unk_5025F38, v7, 8);
    v6 = (unsigned int)qword_5025F30;
  }
  *(_QWORD *)(qword_5025F28 + 8 * v6) = v5;
  qword_5025F70 = (__int64)&unk_49D9748;
  qword_5025EE0 = (__int64)&unk_49DC090;
  qword_5025F80 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5025F30) = qword_5025F30 + 1;
  qword_5025FA0 = (__int64)nullsub_23;
  qword_5025F68 = 0;
  qword_5025F98 = (__int64)sub_984030;
  qword_5025F78 = 0;
  sub_C53080(&qword_5025EE0, "no-stack-coloring", 17);
  LOWORD(qword_5025F78) = 256;
  LOBYTE(qword_5025F68) = 0;
  qword_5025F10 = 22;
  LOBYTE(dword_5025EEC) = dword_5025EEC & 0x9F | 0x20;
  qword_5025F08 = (__int64)"Disable stack coloring";
  sub_C53130(&qword_5025EE0);
  __cxa_atexit(sub_984900, &qword_5025EE0, &qword_4A427C0);
  qword_5025E00 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5025EE0, v8, v9), 1u);
  qword_5025E50 = 0x100000000LL;
  dword_5025E0C &= 0x8000u;
  qword_5025E48 = (__int64)&unk_5025E58;
  word_5025E10 = 0;
  qword_5025E18 = 0;
  dword_5025E08 = v10;
  qword_5025E20 = 0;
  qword_5025E28 = 0;
  qword_5025E30 = 0;
  qword_5025E38 = 0;
  qword_5025E40 = 0;
  qword_5025E60 = 0;
  qword_5025E68 = (__int64)&unk_5025E80;
  qword_5025E70 = 1;
  dword_5025E78 = 0;
  byte_5025E7C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5025E50;
  if ( (unsigned __int64)(unsigned int)qword_5025E50 + 1 > HIDWORD(qword_5025E50) )
  {
    v20 = v11;
    sub_C8D5F0((char *)&unk_5025E58 - 16, &unk_5025E58, (unsigned int)qword_5025E50 + 1LL, 8);
    v12 = (unsigned int)qword_5025E50;
    v11 = v20;
  }
  *(_QWORD *)(qword_5025E48 + 8 * v12) = v11;
  qword_5025E90 = (__int64)&unk_49D9748;
  qword_5025E00 = (__int64)&unk_49DC090;
  qword_5025EA0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5025E50) = qword_5025E50 + 1;
  qword_5025EC0 = (__int64)nullsub_23;
  qword_5025E88 = 0;
  qword_5025EB8 = (__int64)sub_984030;
  qword_5025E98 = 0;
  sub_C53080(&qword_5025E00, "protect-from-escaped-allocas", 28);
  LOWORD(qword_5025E98) = 256;
  LOBYTE(qword_5025E88) = 0;
  qword_5025E30 = 46;
  LOBYTE(dword_5025E0C) = dword_5025E0C & 0x9F | 0x20;
  qword_5025E28 = (__int64)"Do not optimize lifetime zones that are broken";
  sub_C53130(&qword_5025E00);
  __cxa_atexit(sub_984900, &qword_5025E00, &qword_4A427C0);
  qword_5025D20 = (__int64)&unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5025E00, v13, v14), 1u);
  qword_5025D70 = 0x100000000LL;
  dword_5025D2C &= 0x8000u;
  word_5025D30 = 0;
  qword_5025D68 = (__int64)&unk_5025D78;
  qword_5025D38 = 0;
  dword_5025D28 = v15;
  qword_5025D40 = 0;
  qword_5025D48 = 0;
  qword_5025D50 = 0;
  qword_5025D58 = 0;
  qword_5025D60 = 0;
  qword_5025D80 = 0;
  qword_5025D88 = (__int64)&unk_5025DA0;
  qword_5025D90 = 1;
  dword_5025D98 = 0;
  byte_5025D9C = 1;
  v16 = sub_C57470();
  v17 = (unsigned int)qword_5025D70;
  v18 = (unsigned int)qword_5025D70 + 1LL;
  if ( v18 > HIDWORD(qword_5025D70) )
  {
    sub_C8D5F0((char *)&unk_5025D78 - 16, &unk_5025D78, v18, 8);
    v17 = (unsigned int)qword_5025D70;
  }
  *(_QWORD *)(qword_5025D68 + 8 * v17) = v16;
  qword_5025DB0 = (__int64)&unk_49D9748;
  qword_5025D20 = (__int64)&unk_49DC090;
  qword_5025DC0 = (__int64)&unk_49DC1D0;
  LODWORD(qword_5025D70) = qword_5025D70 + 1;
  qword_5025DE0 = (__int64)nullsub_23;
  qword_5025DA8 = 0;
  qword_5025DD8 = (__int64)sub_984030;
  qword_5025DB8 = 0;
  sub_C53080(&qword_5025D20, "stackcoloring-lifetime-start-on-first-use", 41);
  LOBYTE(qword_5025DA8) = 1;
  LOWORD(qword_5025DB8) = 257;
  qword_5025D50 = 68;
  LOBYTE(dword_5025D2C) = dword_5025D2C & 0x9F | 0x20;
  qword_5025D48 = (__int64)"Treat stack lifetimes as starting on first use, not on START marker.";
  sub_C53130(&qword_5025D20);
  return __cxa_atexit(sub_984900, &qword_5025D20, &qword_4A427C0);
}
