// Function: ctor_705
// Address: 0x5bd430
//
__int64 __fastcall ctor_705(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  int v9; // edx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rcx
  int v14; // edx
  __int64 v15; // rbx
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r12
  __int64 result; // rax
  __int64 v21; // [rsp+8h] [rbp-78h]
  _QWORD v22[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v23[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD v24[2]; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v25[8]; // [rsp+40h] [rbp-40h] BYREF

  qword_5050B60 = &unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  *(_DWORD *)&word_5050B6C = word_5050B6C & 0x8000;
  qword_5050BA8[1] = 0x100000000LL;
  unk_5050B68 = v4;
  unk_5050B70 = 0;
  unk_5050B78 = 0;
  unk_5050B80 = 0;
  unk_5050B88 = 0;
  unk_5050B90 = 0;
  unk_5050B98 = 0;
  unk_5050BA0 = 0;
  qword_5050BA8[0] = &qword_5050BA8[2];
  qword_5050BA8[3] = 0;
  qword_5050BA8[4] = &qword_5050BA8[7];
  qword_5050BA8[5] = 1;
  LODWORD(qword_5050BA8[6]) = 0;
  BYTE4(qword_5050BA8[6]) = 1;
  v5 = sub_C57470();
  v6 = LODWORD(qword_5050BA8[1]);
  if ( (unsigned __int64)LODWORD(qword_5050BA8[1]) + 1 > HIDWORD(qword_5050BA8[1]) )
  {
    sub_C8D5F0(qword_5050BA8, &qword_5050BA8[2], LODWORD(qword_5050BA8[1]) + 1LL, 8);
    v6 = LODWORD(qword_5050BA8[1]);
  }
  *(_QWORD *)(qword_5050BA8[0] + 8 * v6) = v5;
  qword_5050BA8[9] = &unk_49D9748;
  ++LODWORD(qword_5050BA8[1]);
  qword_5050BA8[8] = 0;
  qword_5050B60 = &unk_49DC090;
  qword_5050BA8[10] = 0;
  qword_5050BA8[11] = &unk_49DC1D0;
  qword_5050BA8[15] = nullsub_23;
  qword_5050BA8[14] = sub_984030;
  sub_C53080(&qword_5050B60, "cssa-coalesce", 13);
  LOWORD(qword_5050BA8[10]) = 256;
  LOBYTE(qword_5050BA8[8]) = 0;
  LOBYTE(word_5050B6C) = word_5050B6C & 0x9F | 0x20;
  sub_C53130(&qword_5050B60);
  __cxa_atexit(sub_984900, &qword_5050B60, &qword_4A427C0);
  qword_5050A80 = &unk_49DC150;
  v9 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5050B60, v7, v8), 1u);
  *(_DWORD *)&word_5050A8C = word_5050A8C & 0x8000;
  qword_5050AC8[1] = 0x100000000LL;
  unk_5050A88 = v9;
  unk_5050A90 = 0;
  unk_5050A98 = 0;
  unk_5050AA0 = 0;
  unk_5050AA8 = 0;
  unk_5050AB0 = 0;
  unk_5050AB8 = 0;
  unk_5050AC0 = 0;
  qword_5050AC8[0] = &qword_5050AC8[2];
  qword_5050AC8[3] = 0;
  qword_5050AC8[4] = &qword_5050AC8[7];
  qword_5050AC8[5] = 1;
  LODWORD(qword_5050AC8[6]) = 0;
  BYTE4(qword_5050AC8[6]) = 1;
  v10 = sub_C57470();
  v11 = LODWORD(qword_5050AC8[1]);
  if ( (unsigned __int64)LODWORD(qword_5050AC8[1]) + 1 > HIDWORD(qword_5050AC8[1]) )
  {
    v21 = v10;
    sub_C8D5F0(qword_5050AC8, &qword_5050AC8[2], LODWORD(qword_5050AC8[1]) + 1LL, 8);
    v11 = LODWORD(qword_5050AC8[1]);
    v10 = v21;
  }
  *(_QWORD *)(qword_5050AC8[0] + 8 * v11) = v10;
  ++LODWORD(qword_5050AC8[1]);
  qword_5050AC8[8] = 0;
  qword_5050AC8[9] = &unk_49D9728;
  qword_5050AC8[10] = 0;
  qword_5050A80 = &unk_49DBF10;
  qword_5050AC8[11] = &unk_49DC290;
  qword_5050AC8[15] = nullsub_24;
  qword_5050AC8[14] = sub_984050;
  sub_C53080(&qword_5050A80, "cssa-verbosity", 14);
  BYTE4(qword_5050AC8[10]) = 1;
  LODWORD(qword_5050AC8[8]) = 0;
  LODWORD(qword_5050AC8[10]) = 0;
  LOBYTE(word_5050A8C) = word_5050A8C & 0x9F | 0x20;
  sub_C53130(&qword_5050A80);
  __cxa_atexit(sub_984970, &qword_5050A80, &qword_4A427C0);
  qword_50509A0 = (__int64)&unk_49DC150;
  v14 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984970, &qword_5050A80, v12, v13), 1u);
  byte_5050A1C = 1;
  qword_50509F0 = 0x100000000LL;
  dword_50509AC &= 0x8000u;
  qword_50509B8 = 0;
  qword_50509C0 = 0;
  qword_50509C8 = 0;
  dword_50509A8 = v14;
  word_50509B0 = 0;
  qword_50509D0 = 0;
  qword_50509D8 = 0;
  qword_50509E0 = 0;
  qword_50509E8 = (__int64)&unk_50509F8;
  qword_5050A00 = 0;
  qword_5050A08 = (__int64)&unk_5050A20;
  qword_5050A10 = 1;
  dword_5050A18 = 0;
  v15 = sub_C57470();
  v16 = (unsigned int)qword_50509F0;
  v17 = (unsigned int)qword_50509F0 + 1LL;
  if ( v17 > HIDWORD(qword_50509F0) )
  {
    sub_C8D5F0((char *)&unk_50509F8 - 16, &unk_50509F8, v17, 8);
    v16 = (unsigned int)qword_50509F0;
  }
  *(_QWORD *)(qword_50509E8 + 8 * v16) = v15;
  qword_5050A30 = (__int64)&unk_49D9748;
  LODWORD(qword_50509F0) = qword_50509F0 + 1;
  qword_5050A28 = 0;
  qword_50509A0 = (__int64)&unk_49DC090;
  qword_5050A38 = 0;
  qword_5050A40 = (__int64)&unk_49DC1D0;
  qword_5050A60 = (__int64)nullsub_23;
  qword_5050A58 = (__int64)sub_984030;
  sub_C53080(&qword_50509A0, "dump-before-cssa", 16);
  LOBYTE(qword_5050A28) = 0;
  LOWORD(qword_5050A38) = 256;
  LOBYTE(dword_50509AC) = dword_50509AC & 0x9F | 0x20;
  sub_C53130(&qword_50509A0);
  __cxa_atexit(sub_984900, &qword_50509A0, &qword_4A427C0);
  v18 = sub_C60B10();
  v24[0] = v25;
  v19 = v18;
  sub_371D1D0(v24, "Controls which specific operands of phis in the module are coalesced");
  v22[0] = v23;
  sub_371D1D0(v22, "coalescing-counter");
  result = sub_CF9810(v19, v22, v24);
  if ( (_QWORD *)v22[0] != v23 )
    result = j_j___libc_free_0(v22[0], v23[0] + 1LL);
  if ( (_QWORD *)v24[0] != v25 )
    return j_j___libc_free_0(v24[0], v25[0] + 1LL);
  return result;
}
