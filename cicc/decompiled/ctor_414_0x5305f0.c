// Function: ctor_414
// Address: 0x5305f0
//
int ctor_414()
{
  int v0; // edx
  __int64 v1; // r12
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_4FEFB20 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4FEFB9C = 1;
  qword_4FEFB70 = 0x100000000LL;
  dword_4FEFB2C &= 0x8000u;
  qword_4FEFB38 = 0;
  qword_4FEFB40 = 0;
  qword_4FEFB48 = 0;
  dword_4FEFB28 = v0;
  word_4FEFB30 = 0;
  qword_4FEFB50 = 0;
  qword_4FEFB58 = 0;
  qword_4FEFB60 = 0;
  qword_4FEFB68 = (__int64)&unk_4FEFB78;
  qword_4FEFB80 = 0;
  qword_4FEFB88 = (__int64)&unk_4FEFBA0;
  qword_4FEFB90 = 1;
  dword_4FEFB98 = 0;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FEFB70;
  v3 = (unsigned int)qword_4FEFB70 + 1LL;
  if ( v3 > HIDWORD(qword_4FEFB70) )
  {
    sub_C8D5F0((char *)&unk_4FEFB78 - 16, &unk_4FEFB78, v3, 8);
    v2 = (unsigned int)qword_4FEFB70;
  }
  *(_QWORD *)(qword_4FEFB68 + 8 * v2) = v1;
  qword_4FEFBA8 = (__int64)&byte_4FEFBB8;
  qword_4FEFBD0 = (__int64)&byte_4FEFBE0;
  LODWORD(qword_4FEFB70) = qword_4FEFB70 + 1;
  qword_4FEFBB0 = 0;
  qword_4FEFBC8 = (__int64)&unk_49DC130;
  byte_4FEFBB8 = 0;
  byte_4FEFBE0 = 0;
  qword_4FEFB20 = (__int64)&unk_49DC010;
  qword_4FEFBD8 = 0;
  byte_4FEFBF0 = 0;
  qword_4FEFBF8 = (__int64)&unk_49DC350;
  qword_4FEFC18 = (__int64)nullsub_92;
  qword_4FEFC10 = (__int64)sub_BC4D70;
  sub_C53080(&qword_4FEFB20, "extract-blocks-file", 19);
  qword_4FEFB60 = 8;
  qword_4FEFB58 = (__int64)"filename";
  qword_4FEFB48 = (__int64)"A file containing list of basic blocks to extract";
  qword_4FEFB50 = 49;
  LOBYTE(dword_4FEFB2C) = dword_4FEFB2C & 0x9F | 0x20;
  sub_C53130(&qword_4FEFB20);
  __cxa_atexit(sub_BC5A40, &qword_4FEFB20, &qword_4A427C0);
  qword_4FEFA40 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FEFA4C &= 0x8000u;
  word_4FEFA50 = 0;
  qword_4FEFA90 = 0x100000000LL;
  qword_4FEFA58 = 0;
  qword_4FEFA60 = 0;
  qword_4FEFA68 = 0;
  dword_4FEFA48 = v4;
  qword_4FEFA70 = 0;
  qword_4FEFA78 = 0;
  qword_4FEFA80 = 0;
  qword_4FEFA88 = (__int64)&unk_4FEFA98;
  qword_4FEFAA0 = 0;
  qword_4FEFAA8 = (__int64)&unk_4FEFAC0;
  qword_4FEFAB0 = 1;
  dword_4FEFAB8 = 0;
  byte_4FEFABC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FEFA90;
  v7 = (unsigned int)qword_4FEFA90 + 1LL;
  if ( v7 > HIDWORD(qword_4FEFA90) )
  {
    sub_C8D5F0((char *)&unk_4FEFA98 - 16, &unk_4FEFA98, v7, 8);
    v6 = (unsigned int)qword_4FEFA90;
  }
  *(_QWORD *)(qword_4FEFA88 + 8 * v6) = v5;
  LODWORD(qword_4FEFA90) = qword_4FEFA90 + 1;
  qword_4FEFAC8 = 0;
  qword_4FEFAD0 = (__int64)&unk_49D9748;
  qword_4FEFAD8 = 0;
  qword_4FEFA40 = (__int64)&unk_49DC090;
  qword_4FEFAE0 = (__int64)&unk_49DC1D0;
  qword_4FEFB00 = (__int64)nullsub_23;
  qword_4FEFAF8 = (__int64)sub_984030;
  sub_C53080(&qword_4FEFA40, "extract-blocks-erase-funcs", 26);
  qword_4FEFA70 = 28;
  qword_4FEFA68 = (__int64)"Erase the existing functions";
  LOBYTE(dword_4FEFA4C) = dword_4FEFA4C & 0x9F | 0x20;
  sub_C53130(&qword_4FEFA40);
  return __cxa_atexit(sub_984900, &qword_4FEFA40, &qword_4A427C0);
}
