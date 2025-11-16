// Function: ctor_726
// Address: 0x5c3530
//
__int64 ctor_726()
{
  int v0; // eax
  int v1; // ecx
  int v2; // ecx
  int v3; // esi
  int v4; // edx
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 result; // rax
  _QWORD v8[2]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD v9[2]; // [rsp+40h] [rbp-60h] BYREF
  _QWORD v10[2]; // [rsp+50h] [rbp-50h] BYREF
  _QWORD v11[8]; // [rsp+60h] [rbp-40h] BYREF

  qword_5054C00 = (__int64)&unk_49EED30;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  word_5054C0C &= 0xF000u;
  qword_5054C48 = (__int64)qword_4FA01C0;
  qword_5054C10 = 0;
  qword_5054C18 = 0;
  qword_5054C20 = 0;
  dword_5054C08 = v0;
  qword_5054C58 = (__int64)&unk_5054C78;
  qword_5054C60 = (__int64)&unk_5054C78;
  qword_5054C28 = 0;
  qword_5054C30 = 0;
  qword_5054CA8 = (__int64)&unk_49E74E8;
  word_5054CB0 = 256;
  qword_5054C38 = 0;
  qword_5054C40 = 0;
  qword_5054C00 = (__int64)&unk_49EEC70;
  qword_5054CB8 = (__int64)&unk_49EEDB0;
  qword_5054C50 = 0;
  qword_5054C68 = 4;
  dword_5054C70 = 0;
  byte_5054C98 = 0;
  byte_5054CA0 = 0;
  sub_16B8280(&qword_5054C00, "balance-dot-chain", 17);
  word_5054CB0 = 256;
  qword_5054C28 = (__int64)"Balance the chain of dot operations";
  byte_5054CA0 = 0;
  qword_5054C30 = 35;
  LOBYTE(word_5054C0C) = word_5054C0C & 0x9F | 0x20;
  sub_16B88A0(&qword_5054C00);
  __cxa_atexit(sub_12EDEC0, &qword_5054C00, &qword_4A427C0);
  qword_5054B20 = (__int64)&unk_49EED30;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  qword_5054B78 = (__int64)&unk_5054B98;
  qword_5054B80 = (__int64)&unk_5054B98;
  word_5054B2C &= 0xF000u;
  qword_5054B30 = 0;
  qword_5054BC8 = (__int64)&unk_49E74A8;
  qword_5054B20 = (__int64)&unk_49EEAF0;
  qword_5054BD8 = (__int64)&unk_49EEE10;
  dword_5054B28 = v1;
  qword_5054B68 = (__int64)qword_4FA01C0;
  qword_5054B38 = 0;
  qword_5054B40 = 0;
  qword_5054B48 = 0;
  qword_5054B50 = 0;
  qword_5054B58 = 0;
  qword_5054B60 = 0;
  qword_5054B70 = 0;
  qword_5054B88 = 4;
  dword_5054B90 = 0;
  byte_5054BB8 = 0;
  dword_5054BC0 = 0;
  byte_5054BD4 = 1;
  dword_5054BD0 = 0;
  sub_16B8280(&qword_5054B20, "max-chain-width", 15);
  qword_5054B48 = (__int64)"The width of the tree to use while balancing dot chain";
  dword_5054BC0 = 2;
  byte_5054BD4 = 1;
  dword_5054BD0 = 2;
  LOBYTE(word_5054B2C) = word_5054B2C & 0x9F | 0x20;
  qword_5054B50 = 54;
  sub_16B88A0(&qword_5054B20);
  __cxa_atexit(sub_12EDE60, &qword_5054B20, &qword_4A427C0);
  qword_5054A40 = (__int64)&unk_49EED30;
  v2 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  qword_5054A98 = (__int64)&unk_5054AB8;
  qword_5054AA0 = (__int64)&unk_5054AB8;
  word_5054A4C &= 0xF000u;
  qword_5054AE8 = (__int64)&unk_49E74A8;
  qword_5054A40 = (__int64)&unk_49EEAF0;
  dword_5054A48 = v2;
  qword_5054AF8 = (__int64)&unk_49EEE10;
  qword_5054A88 = (__int64)qword_4FA01C0;
  qword_5054A50 = 0;
  qword_5054A58 = 0;
  qword_5054A60 = 0;
  qword_5054A68 = 0;
  qword_5054A70 = 0;
  qword_5054A78 = 0;
  qword_5054A80 = 0;
  qword_5054A90 = 0;
  qword_5054AA8 = 4;
  dword_5054AB0 = 0;
  byte_5054AD8 = 0;
  dword_5054AE0 = 0;
  byte_5054AF4 = 1;
  dword_5054AF0 = 0;
  sub_16B8280((char *)&unk_5054AB8 - 120, "max-chain-length", 16);
  qword_5054A68 = (__int64)"Max Length of the chain of operations selected for idpa generation";
  dword_5054AE0 = 64;
  byte_5054AF4 = 1;
  dword_5054AF0 = 64;
  LOBYTE(word_5054A4C) = word_5054A4C & 0x9F | 0x20;
  qword_5054A70 = 66;
  sub_16B88A0(&qword_5054A40);
  __cxa_atexit(sub_12EDE60, &qword_5054A40, &qword_4A427C0);
  qword_5054960 = (__int64)&unk_49EED30;
  v3 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  qword_50549B8 = (__int64)&unk_50549D8;
  qword_50549C0 = (__int64)&unk_50549D8;
  word_5054A10 = 256;
  word_505496C &= 0xF000u;
  qword_5054A08 = (__int64)&unk_49E74E8;
  dword_5054968 = v3;
  qword_50549A8 = (__int64)qword_4FA01C0;
  qword_5054960 = (__int64)&unk_49EEC70;
  qword_5054A18 = (__int64)&unk_49EEDB0;
  qword_5054970 = 0;
  qword_5054978 = 0;
  qword_5054980 = 0;
  qword_5054988 = 0;
  qword_5054990 = 0;
  qword_5054998 = 0;
  qword_50549A0 = 0;
  qword_50549B0 = 0;
  qword_50549C8 = 4;
  dword_50549D0 = 0;
  byte_50549F8 = 0;
  byte_5054A00 = 0;
  sub_16B8280(&qword_5054960, "aggressive-no-sink", 18);
  word_5054A10 = 257;
  qword_5054988 = (__int64)"Sink all generated instructions";
  byte_5054A00 = 1;
  qword_5054990 = 31;
  LOBYTE(word_505496C) = word_505496C & 0x9F | 0x20;
  sub_16B88A0(&qword_5054960);
  __cxa_atexit(sub_12EDEC0, &qword_5054960, &qword_4A427C0);
  qword_5054880 = (__int64)&unk_49EED30;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  qword_50548D8 = (__int64)&unk_50548F8;
  qword_50548E0 = (__int64)&unk_50548F8;
  word_5054930 = 256;
  qword_5054880 = (__int64)&unk_49EEC70;
  qword_5054928 = (__int64)&unk_49E74E8;
  dword_5054888 = v4;
  qword_50548C8 = (__int64)qword_4FA01C0;
  qword_5054938 = (__int64)&unk_49EEDB0;
  word_505488C &= 0xF000u;
  qword_5054890 = 0;
  qword_5054898 = 0;
  qword_50548A0 = 0;
  qword_50548A8 = 0;
  qword_50548B0 = 0;
  qword_50548B8 = 0;
  qword_50548C0 = 0;
  qword_50548D0 = 0;
  qword_50548E8 = 4;
  dword_50548F0 = 0;
  byte_5054918 = 0;
  byte_5054920 = 0;
  sub_16B8280(&qword_5054880, "enable-dot", 10);
  word_5054930 = 257;
  qword_50548A8 = (__int64)"Enable Dot Transformation";
  qword_50548B0 = 25;
  byte_5054920 = 1;
  sub_16B88A0(&qword_5054880);
  __cxa_atexit(sub_12EDEC0, &qword_5054880, &qword_4A427C0);
  v5 = sub_16BAF20();
  v10[0] = v11;
  v6 = v5;
  sub_3959810(v10, "Controls dot transformations.");
  v8[0] = v9;
  sub_3959810(v8, "dot-counter");
  result = sub_14C9E50(v6, v8, v10);
  if ( (_QWORD *)v8[0] != v9 )
    result = j_j___libc_free_0(v8[0], v9[0] + 1LL);
  if ( (_QWORD *)v10[0] != v11 )
    return j_j___libc_free_0(v10[0], v11[0] + 1LL);
  return result;
}
