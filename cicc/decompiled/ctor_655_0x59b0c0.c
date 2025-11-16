// Function: ctor_655
// Address: 0x59b0c0
//
int __fastcall ctor_655(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  int v17; // edx
  __int64 v18; // r13
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  const char *v22; // [rsp+0h] [rbp-50h] BYREF
  char v23; // [rsp+20h] [rbp-30h]
  char v24; // [rsp+21h] [rbp-2Fh]

  qword_5039340 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_5039390 = 0x100000000LL;
  dword_503934C &= 0x8000u;
  word_5039350 = 0;
  qword_5039358 = 0;
  qword_5039360 = 0;
  dword_5039348 = v4;
  qword_5039368 = 0;
  qword_5039370 = 0;
  qword_5039378 = 0;
  qword_5039380 = 0;
  qword_5039388 = (__int64)&unk_5039398;
  qword_50393A0 = 0;
  qword_50393A8 = (__int64)&unk_50393C0;
  qword_50393B0 = 1;
  dword_50393B8 = 0;
  byte_50393BC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5039390;
  v7 = (unsigned int)qword_5039390 + 1LL;
  if ( v7 > HIDWORD(qword_5039390) )
  {
    sub_C8D5F0((char *)&unk_5039398 - 16, &unk_5039398, v7, 8);
    v6 = (unsigned int)qword_5039390;
  }
  *(_QWORD *)(qword_5039388 + 8 * v6) = v5;
  LODWORD(qword_5039390) = qword_5039390 + 1;
  qword_50393C8 = 0;
  qword_50393D0 = (__int64)&unk_49D9748;
  qword_50393D8 = 0;
  qword_5039340 = (__int64)&unk_49DC090;
  qword_50393E0 = (__int64)&unk_49DC1D0;
  qword_5039400 = (__int64)nullsub_23;
  qword_50393F8 = (__int64)sub_984030;
  sub_C53080(&qword_5039340, "insert-assert-align", 19);
  qword_5039368 = (__int64)"Insert the experimental `assertalign` node.";
  LOWORD(qword_50393D8) = 257;
  LOBYTE(qword_50393C8) = 1;
  qword_5039370 = 43;
  LOBYTE(dword_503934C) = dword_503934C & 0x9F | 0x40;
  sub_C53130(&qword_5039340);
  __cxa_atexit(sub_984900, &qword_5039340, &qword_4A427C0);
  qword_5039260 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &qword_5039340, v8, v9), 1u);
  qword_50392B0 = 0x100000000LL;
  word_5039270 = 0;
  dword_503926C &= 0x8000u;
  qword_5039278 = 0;
  qword_5039280 = 0;
  dword_5039268 = v10;
  qword_5039288 = 0;
  qword_5039290 = 0;
  qword_5039298 = 0;
  qword_50392A0 = 0;
  qword_50392A8 = (__int64)&unk_50392B8;
  qword_50392C0 = 0;
  qword_50392C8 = (__int64)&unk_50392E0;
  qword_50392D0 = 1;
  dword_50392D8 = 0;
  byte_50392DC = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_50392B0;
  v13 = (unsigned int)qword_50392B0 + 1LL;
  if ( v13 > HIDWORD(qword_50392B0) )
  {
    sub_C8D5F0((char *)&unk_50392B8 - 16, &unk_50392B8, v13, 8);
    v12 = (unsigned int)qword_50392B0;
  }
  *(_QWORD *)(qword_50392A8 + 8 * v12) = v11;
  qword_50392F0 = (__int64)&unk_49D9728;
  LODWORD(qword_50392B0) = qword_50392B0 + 1;
  byte_50392FC = 0;
  qword_5039260 = (__int64)&unk_49DDF20;
  qword_5039300 = (__int64)&unk_49DC290;
  qword_50392E8 = 0;
  qword_5039320 = (__int64)nullsub_186;
  qword_5039318 = (__int64)sub_D320E0;
  sub_C53080(&qword_5039260, "limit-float-precision", 21);
  qword_5039290 = 63;
  qword_5039288 = (__int64)"Generate low-precision inline sequences for some float libcalls";
  if ( qword_50392E8 )
  {
    v14 = sub_CEADF0();
    v24 = 1;
    v22 = "cl::location(x) specified more than once!";
    v23 = 3;
    sub_C53280(&qword_5039260, &v22, 0, 0, v14);
  }
  else
  {
    qword_50392E8 = (__int64)&dword_5039408;
  }
  LOBYTE(dword_503926C) = dword_503926C & 0x9F | 0x20;
  *(_DWORD *)qword_50392E8 = 0;
  byte_50392FC = 1;
  dword_50392F8 = 0;
  sub_C53130(&qword_5039260);
  __cxa_atexit(sub_D32600, &qword_5039260, &qword_4A427C0);
  qword_5039180 = (__int64)&unk_49DC150;
  v17 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_D32600, &qword_5039260, v15, v16), 1u);
  dword_503918C &= 0x8000u;
  word_5039190 = 0;
  qword_50391D0 = 0x100000000LL;
  qword_5039198 = 0;
  qword_50391A0 = 0;
  qword_50391A8 = 0;
  dword_5039188 = v17;
  qword_50391B0 = 0;
  qword_50391B8 = 0;
  qword_50391C0 = 0;
  qword_50391C8 = (__int64)&unk_50391D8;
  qword_50391E0 = 0;
  qword_50391E8 = (__int64)&unk_5039200;
  qword_50391F0 = 1;
  dword_50391F8 = 0;
  byte_50391FC = 1;
  v18 = sub_C57470();
  v19 = (unsigned int)qword_50391D0;
  v20 = (unsigned int)qword_50391D0 + 1LL;
  if ( v20 > HIDWORD(qword_50391D0) )
  {
    sub_C8D5F0((char *)&unk_50391D8 - 16, &unk_50391D8, v20, 8);
    v19 = (unsigned int)qword_50391D0;
  }
  *(_QWORD *)(qword_50391C8 + 8 * v19) = v18;
  qword_5039210 = (__int64)&unk_49D9728;
  LODWORD(qword_50391D0) = qword_50391D0 + 1;
  qword_5039208 = 0;
  qword_5039180 = (__int64)&unk_49DBF10;
  qword_5039220 = (__int64)&unk_49DC290;
  qword_5039218 = 0;
  qword_5039240 = (__int64)nullsub_24;
  qword_5039238 = (__int64)sub_984050;
  sub_C53080(&qword_5039180, "switch-peel-threshold", 21);
  LODWORD(qword_5039208) = 66;
  BYTE4(qword_5039218) = 1;
  LODWORD(qword_5039218) = 66;
  qword_50391B0 = 133;
  LOBYTE(dword_503918C) = dword_503918C & 0x9F | 0x20;
  qword_50391A8 = (__int64)"Set the case probability threshold for peeling the case from a switch statement. A value grea"
                           "ter than 100 will void this optimization";
  sub_C53130(&qword_5039180);
  return __cxa_atexit(sub_984970, &qword_5039180, &qword_4A427C0);
}
