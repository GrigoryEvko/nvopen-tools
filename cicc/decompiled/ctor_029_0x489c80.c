// Function: ctor_029
// Address: 0x489c80
//
int ctor_029()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  _QWORD v10[2]; // [rsp+0h] [rbp-40h] BYREF
  _BYTE v11[48]; // [rsp+10h] [rbp-30h] BYREF

  qword_4F81EC0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F81F10 = 0x100000000LL;
  dword_4F81ECC &= 0x8000u;
  word_4F81ED0 = 0;
  qword_4F81ED8 = 0;
  qword_4F81EE0 = 0;
  dword_4F81EC8 = v0;
  qword_4F81EE8 = 0;
  qword_4F81EF0 = 0;
  qword_4F81EF8 = 0;
  qword_4F81F00 = 0;
  qword_4F81F08 = (__int64)&unk_4F81F18;
  qword_4F81F20 = 0;
  qword_4F81F28 = (__int64)&unk_4F81F40;
  qword_4F81F30 = 1;
  dword_4F81F38 = 0;
  byte_4F81F3C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4F81F10;
  v3 = (unsigned int)qword_4F81F10 + 1LL;
  if ( v3 > HIDWORD(qword_4F81F10) )
  {
    sub_C8D5F0((char *)&unk_4F81F18 - 16, &unk_4F81F18, v3, 8);
    v2 = (unsigned int)qword_4F81F10;
  }
  *(_QWORD *)(qword_4F81F08 + 8 * v2) = v1;
  qword_4F81F50 = (__int64)&unk_49D9748;
  LODWORD(qword_4F81F10) = qword_4F81F10 + 1;
  qword_4F81F48 = 0;
  qword_4F81EC0 = (__int64)&unk_49DC090;
  qword_4F81F60 = (__int64)&unk_49DC1D0;
  qword_4F81F58 = 0;
  qword_4F81F80 = (__int64)nullsub_23;
  qword_4F81F78 = (__int64)sub_984030;
  sub_C53080(&qword_4F81EC0, "propagate-attrs", 15);
  LOWORD(qword_4F81F58) = 257;
  LOBYTE(qword_4F81F48) = 1;
  qword_4F81EF0 = 29;
  LOBYTE(dword_4F81ECC) = dword_4F81ECC & 0x9F | 0x20;
  qword_4F81EE8 = (__int64)"Propagate attributes in index";
  sub_C53130(&qword_4F81EC0);
  __cxa_atexit(sub_984900, &qword_4F81EC0, &qword_4A427C0);
  qword_4F81DE0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4F81E30 = 0x100000000LL;
  word_4F81DF0 = 0;
  dword_4F81DEC &= 0x8000u;
  qword_4F81DF8 = 0;
  qword_4F81E00 = 0;
  dword_4F81DE8 = v4;
  qword_4F81E08 = 0;
  qword_4F81E10 = 0;
  qword_4F81E18 = 0;
  qword_4F81E20 = 0;
  qword_4F81E28 = (__int64)&unk_4F81E38;
  qword_4F81E40 = 0;
  qword_4F81E48 = (__int64)&unk_4F81E60;
  qword_4F81E50 = 1;
  dword_4F81E58 = 0;
  byte_4F81E5C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4F81E30;
  v7 = (unsigned int)qword_4F81E30 + 1LL;
  if ( v7 > HIDWORD(qword_4F81E30) )
  {
    sub_C8D5F0((char *)&unk_4F81E38 - 16, &unk_4F81E38, v7, 8);
    v6 = (unsigned int)qword_4F81E30;
  }
  *(_QWORD *)(qword_4F81E28 + 8 * v6) = v5;
  qword_4F81E70 = (__int64)&unk_49D9748;
  LODWORD(qword_4F81E30) = qword_4F81E30 + 1;
  qword_4F81E68 = 0;
  qword_4F81DE0 = (__int64)&unk_49DC090;
  qword_4F81E80 = (__int64)&unk_49DC1D0;
  qword_4F81E78 = 0;
  qword_4F81EA0 = (__int64)nullsub_23;
  qword_4F81E98 = (__int64)sub_984030;
  sub_C53080(&qword_4F81DE0, "import-constants-with-refs", 26);
  LOBYTE(qword_4F81E68) = 1;
  LOWORD(qword_4F81E78) = 257;
  qword_4F81E10 = 48;
  LOBYTE(dword_4F81DEC) = dword_4F81DEC & 0x9F | 0x20;
  qword_4F81E08 = (__int64)"Import constant global variables with references";
  sub_C53130(&qword_4F81DE0);
  __cxa_atexit(sub_984900, &qword_4F81DE0, &qword_4A427C0);
  v10[0] = v11;
  v10[1] = 0;
  sub_BAE7B0(&unk_4F81D60, v10);
  if ( (_BYTE *)v10[0] != v11 )
    _libc_free(v10[0], v10, v8);
  return __cxa_atexit(sub_9C53D0, &unk_4F81D60, &qword_4A427C0);
}
