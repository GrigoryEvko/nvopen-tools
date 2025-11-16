// Function: ctor_424
// Address: 0x5341e0
//
int ctor_424()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v4; // edx
  __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // rdx

  qword_4FF1EC0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF1F10 = 0x100000000LL;
  dword_4FF1ECC &= 0x8000u;
  word_4FF1ED0 = 0;
  qword_4FF1ED8 = 0;
  qword_4FF1EE0 = 0;
  dword_4FF1EC8 = v0;
  qword_4FF1EE8 = 0;
  qword_4FF1EF0 = 0;
  qword_4FF1EF8 = 0;
  qword_4FF1F00 = 0;
  qword_4FF1F08 = (__int64)&unk_4FF1F18;
  qword_4FF1F20 = 0;
  qword_4FF1F28 = (__int64)&unk_4FF1F40;
  qword_4FF1F30 = 1;
  dword_4FF1F38 = 0;
  byte_4FF1F3C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FF1F10;
  v3 = (unsigned int)qword_4FF1F10 + 1LL;
  if ( v3 > HIDWORD(qword_4FF1F10) )
  {
    sub_C8D5F0((char *)&unk_4FF1F18 - 16, &unk_4FF1F18, v3, 8);
    v2 = (unsigned int)qword_4FF1F10;
  }
  *(_QWORD *)(qword_4FF1F08 + 8 * v2) = v1;
  qword_4FF1F50 = (__int64)&unk_49D9748;
  LODWORD(qword_4FF1F10) = qword_4FF1F10 + 1;
  qword_4FF1F48 = 0;
  qword_4FF1EC0 = (__int64)&unk_49DC090;
  qword_4FF1F60 = (__int64)&unk_49DC1D0;
  qword_4FF1F58 = 0;
  qword_4FF1F80 = (__int64)nullsub_23;
  qword_4FF1F78 = (__int64)sub_984030;
  sub_C53080(&qword_4FF1EC0, "enable-linkonceodr-ir-outlining", 31);
  LOWORD(qword_4FF1F58) = 256;
  LOBYTE(qword_4FF1F48) = 0;
  qword_4FF1EF0 = 47;
  LOBYTE(dword_4FF1ECC) = dword_4FF1ECC & 0x9F | 0x20;
  qword_4FF1EE8 = (__int64)"Enable the IR outliner on linkonceodr functions";
  sub_C53130(&qword_4FF1EC0);
  __cxa_atexit(sub_984900, &qword_4FF1EC0, &qword_4A427C0);
  qword_4FF1DE0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  qword_4FF1E30 = 0x100000000LL;
  word_4FF1DF0 = 0;
  dword_4FF1DEC &= 0x8000u;
  qword_4FF1DF8 = 0;
  qword_4FF1E00 = 0;
  dword_4FF1DE8 = v4;
  qword_4FF1E08 = 0;
  qword_4FF1E10 = 0;
  qword_4FF1E18 = 0;
  qword_4FF1E20 = 0;
  qword_4FF1E28 = (__int64)&unk_4FF1E38;
  qword_4FF1E40 = 0;
  qword_4FF1E48 = (__int64)&unk_4FF1E60;
  qword_4FF1E50 = 1;
  dword_4FF1E58 = 0;
  byte_4FF1E5C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_4FF1E30;
  v7 = (unsigned int)qword_4FF1E30 + 1LL;
  if ( v7 > HIDWORD(qword_4FF1E30) )
  {
    sub_C8D5F0((char *)&unk_4FF1E38 - 16, &unk_4FF1E38, v7, 8);
    v6 = (unsigned int)qword_4FF1E30;
  }
  *(_QWORD *)(qword_4FF1E28 + 8 * v6) = v5;
  qword_4FF1E70 = (__int64)&unk_49D9748;
  LODWORD(qword_4FF1E30) = qword_4FF1E30 + 1;
  qword_4FF1E68 = 0;
  qword_4FF1DE0 = (__int64)&unk_49DC090;
  qword_4FF1E80 = (__int64)&unk_49DC1D0;
  qword_4FF1E78 = 0;
  qword_4FF1EA0 = (__int64)nullsub_23;
  qword_4FF1E98 = (__int64)sub_984030;
  sub_C53080(&qword_4FF1DE0, "ir-outlining-no-cost", 20);
  LOBYTE(qword_4FF1E68) = 0;
  LOWORD(qword_4FF1E78) = 256;
  qword_4FF1E10 = 92;
  LOBYTE(dword_4FF1DEC) = dword_4FF1DEC & 0x9F | 0x40;
  qword_4FF1E08 = (__int64)"Debug option to outline greedily, without restriction that calculated benefit outweighs cost";
  sub_C53130(&qword_4FF1DE0);
  return __cxa_atexit(sub_984900, &qword_4FF1DE0, &qword_4A427C0);
}
