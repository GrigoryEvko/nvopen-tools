// Function: ctor_667
// Address: 0x59ef90
//
int __fastcall ctor_667(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
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
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rcx
  int v20; // edx
  __int64 v21; // rbx
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rcx
  int v26; // edx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // rcx
  int v31; // edx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rdx
  __int64 v35; // rcx
  int v36; // edx
  __int64 v37; // rbx
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  __int64 v41; // [rsp+8h] [rbp-48h]
  __int64 v42; // [rsp+8h] [rbp-48h]
  __int64 v43; // [rsp+8h] [rbp-48h]
  __int64 v44; // [rsp+8h] [rbp-48h]
  char v45; // [rsp+13h] [rbp-3Dh] BYREF
  int v46; // [rsp+14h] [rbp-3Ch] BYREF
  _QWORD v47[7]; // [rsp+18h] [rbp-38h] BYREF

  qword_503BA20 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  dword_503BA2C &= 0x8000u;
  word_503BA30 = 0;
  qword_503BA70 = 0x100000000LL;
  qword_503BA38 = 0;
  qword_503BA40 = 0;
  qword_503BA48 = 0;
  dword_503BA28 = v4;
  qword_503BA50 = 0;
  qword_503BA58 = 0;
  qword_503BA60 = 0;
  qword_503BA68 = (__int64)&unk_503BA78;
  qword_503BA80 = 0;
  qword_503BA88 = (__int64)&unk_503BAA0;
  qword_503BA90 = 1;
  dword_503BA98 = 0;
  byte_503BA9C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_503BA70;
  v7 = (unsigned int)qword_503BA70 + 1LL;
  if ( v7 > HIDWORD(qword_503BA70) )
  {
    sub_C8D5F0((char *)&unk_503BA78 - 16, &unk_503BA78, v7, 8);
    v6 = (unsigned int)qword_503BA70;
  }
  *(_QWORD *)(qword_503BA68 + 8 * v6) = v5;
  LODWORD(qword_503BA70) = qword_503BA70 + 1;
  qword_503BAA8 = 0;
  qword_503BAB0 = (__int64)&unk_49DA090;
  qword_503BA20 = (__int64)&unk_49DBF90;
  qword_503BAC0 = (__int64)&unk_49DC230;
  qword_503BAE0 = (__int64)nullsub_58;
  qword_503BAD8 = (__int64)sub_B2B5F0;
  qword_503BAB8 = 0;
  sub_C53080(&qword_503BA20, "ifcvt-fn-start", 14);
  LODWORD(qword_503BAA8) = -1;
  BYTE4(qword_503BAB8) = 1;
  LODWORD(qword_503BAB8) = -1;
  LOBYTE(dword_503BA2C) = dword_503BA2C & 0x9F | 0x20;
  sub_C53130(&qword_503BA20);
  __cxa_atexit(sub_B2B680, &qword_503BA20, &qword_4A427C0);
  qword_503B940 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_B2B680, &qword_503BA20, v8, v9), 1u);
  dword_503B94C &= 0x8000u;
  word_503B950 = 0;
  qword_503B990 = 0x100000000LL;
  qword_503B988 = (__int64)&unk_503B998;
  qword_503B958 = 0;
  qword_503B960 = 0;
  dword_503B948 = v10;
  qword_503B968 = 0;
  qword_503B970 = 0;
  qword_503B978 = 0;
  qword_503B980 = 0;
  qword_503B9A0 = 0;
  qword_503B9A8 = (__int64)&unk_503B9C0;
  qword_503B9B0 = 1;
  dword_503B9B8 = 0;
  byte_503B9BC = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_503B990;
  if ( (unsigned __int64)(unsigned int)qword_503B990 + 1 > HIDWORD(qword_503B990) )
  {
    v41 = v11;
    sub_C8D5F0((char *)&unk_503B998 - 16, &unk_503B998, (unsigned int)qword_503B990 + 1LL, 8);
    v12 = (unsigned int)qword_503B990;
    v11 = v41;
  }
  *(_QWORD *)(qword_503B988 + 8 * v12) = v11;
  LODWORD(qword_503B990) = qword_503B990 + 1;
  qword_503BA00 = (__int64)nullsub_58;
  qword_503B9D0 = (__int64)&unk_49DA090;
  qword_503B940 = (__int64)&unk_49DBF90;
  qword_503B9E0 = (__int64)&unk_49DC230;
  qword_503B9C8 = 0;
  qword_503B9F8 = (__int64)sub_B2B5F0;
  qword_503B9D8 = 0;
  sub_C53080(&qword_503B940, "ifcvt-fn-stop", 13);
  LODWORD(qword_503B9C8) = -1;
  BYTE4(qword_503B9D8) = 1;
  LODWORD(qword_503B9D8) = -1;
  LOBYTE(dword_503B94C) = dword_503B94C & 0x9F | 0x20;
  sub_C53130(&qword_503B940);
  __cxa_atexit(sub_B2B680, &qword_503B940, &qword_4A427C0);
  qword_503B860 = (__int64)&unk_49DC150;
  v15 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_B2B680, &qword_503B940, v13, v14), 1u);
  qword_503B8B0 = 0x100000000LL;
  dword_503B86C &= 0x8000u;
  qword_503B8A8 = (__int64)&unk_503B8B8;
  word_503B870 = 0;
  qword_503B878 = 0;
  dword_503B868 = v15;
  qword_503B880 = 0;
  qword_503B888 = 0;
  qword_503B890 = 0;
  qword_503B898 = 0;
  qword_503B8A0 = 0;
  qword_503B8C0 = 0;
  qword_503B8C8 = (__int64)&unk_503B8E0;
  qword_503B8D0 = 1;
  dword_503B8D8 = 0;
  byte_503B8DC = 1;
  v16 = sub_C57470();
  v17 = (unsigned int)qword_503B8B0;
  if ( (unsigned __int64)(unsigned int)qword_503B8B0 + 1 > HIDWORD(qword_503B8B0) )
  {
    v42 = v16;
    sub_C8D5F0((char *)&unk_503B8B8 - 16, &unk_503B8B8, (unsigned int)qword_503B8B0 + 1LL, 8);
    v17 = (unsigned int)qword_503B8B0;
    v16 = v42;
  }
  *(_QWORD *)(qword_503B8A8 + 8 * v17) = v16;
  LODWORD(qword_503B8B0) = qword_503B8B0 + 1;
  qword_503B920 = (__int64)nullsub_58;
  qword_503B8F0 = (__int64)&unk_49DA090;
  qword_503B860 = (__int64)&unk_49DBF90;
  qword_503B900 = (__int64)&unk_49DC230;
  qword_503B8E8 = 0;
  qword_503B918 = (__int64)sub_B2B5F0;
  qword_503B8F8 = 0;
  sub_C53080(&qword_503B860, "ifcvt-limit", 11);
  LODWORD(qword_503B8E8) = -1;
  BYTE4(qword_503B8F8) = 1;
  LODWORD(qword_503B8F8) = -1;
  LOBYTE(dword_503B86C) = dword_503B86C & 0x9F | 0x20;
  sub_C53130(&qword_503B860);
  __cxa_atexit(sub_B2B680, &qword_503B860, &qword_4A427C0);
  qword_503B780 = (__int64)&unk_49DC150;
  v20 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_B2B680, &qword_503B860, v18, v19), 1u);
  qword_503B7D0 = 0x100000000LL;
  dword_503B78C &= 0x8000u;
  word_503B790 = 0;
  qword_503B798 = 0;
  qword_503B7A0 = 0;
  dword_503B788 = v20;
  qword_503B7A8 = 0;
  qword_503B7B0 = 0;
  qword_503B7B8 = 0;
  qword_503B7C0 = 0;
  qword_503B7C8 = (__int64)&unk_503B7D8;
  qword_503B7E0 = 0;
  qword_503B7E8 = (__int64)&unk_503B800;
  qword_503B7F0 = 1;
  dword_503B7F8 = 0;
  byte_503B7FC = 1;
  v21 = sub_C57470();
  v22 = (unsigned int)qword_503B7D0;
  v23 = (unsigned int)qword_503B7D0 + 1LL;
  if ( v23 > HIDWORD(qword_503B7D0) )
  {
    sub_C8D5F0((char *)&unk_503B7D8 - 16, &unk_503B7D8, v23, 8);
    v22 = (unsigned int)qword_503B7D0;
  }
  *(_QWORD *)(qword_503B7C8 + 8 * v22) = v21;
  LODWORD(qword_503B7D0) = qword_503B7D0 + 1;
  qword_503B808 = 0;
  qword_503B810 = (__int64)&unk_49D9748;
  qword_503B818 = 0;
  qword_503B780 = (__int64)&unk_49DC090;
  qword_503B820 = (__int64)&unk_49DC1D0;
  qword_503B840 = (__int64)nullsub_23;
  qword_503B838 = (__int64)sub_984030;
  sub_C53080(&qword_503B780, "disable-ifcvt-simple", 20);
  LOWORD(qword_503B818) = 256;
  LOBYTE(qword_503B808) = 0;
  LOBYTE(dword_503B78C) = dword_503B78C & 0x9F | 0x20;
  sub_C53130(&qword_503B780);
  __cxa_atexit(sub_984900, &qword_503B780, &qword_4A427C0);
  v45 = 0;
  v46 = 1;
  v47[0] = &v45;
  sub_34EB9E0(&unk_503B6A0, "disable-ifcvt-simple-false", v47, &v46);
  __cxa_atexit(sub_984900, &unk_503B6A0, &qword_4A427C0);
  qword_503B5C0 = (__int64)&unk_49DC150;
  v26 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &unk_503B6A0, v24, v25), 1u);
  qword_503B610 = 0x100000000LL;
  dword_503B5CC &= 0x8000u;
  qword_503B608 = (__int64)&unk_503B618;
  word_503B5D0 = 0;
  qword_503B5D8 = 0;
  dword_503B5C8 = v26;
  qword_503B5E0 = 0;
  qword_503B5E8 = 0;
  qword_503B5F0 = 0;
  qword_503B5F8 = 0;
  qword_503B600 = 0;
  qword_503B620 = 0;
  qword_503B628 = (__int64)&unk_503B640;
  qword_503B630 = 1;
  dword_503B638 = 0;
  byte_503B63C = 1;
  v27 = sub_C57470();
  v28 = (unsigned int)qword_503B610;
  if ( (unsigned __int64)(unsigned int)qword_503B610 + 1 > HIDWORD(qword_503B610) )
  {
    v43 = v27;
    sub_C8D5F0((char *)&unk_503B618 - 16, &unk_503B618, (unsigned int)qword_503B610 + 1LL, 8);
    v28 = (unsigned int)qword_503B610;
    v27 = v43;
  }
  *(_QWORD *)(qword_503B608 + 8 * v28) = v27;
  LODWORD(qword_503B610) = qword_503B610 + 1;
  qword_503B648 = 0;
  qword_503B650 = (__int64)&unk_49D9748;
  qword_503B658 = 0;
  qword_503B5C0 = (__int64)&unk_49DC090;
  qword_503B660 = (__int64)&unk_49DC1D0;
  qword_503B680 = (__int64)nullsub_23;
  qword_503B678 = (__int64)sub_984030;
  sub_C53080(&qword_503B5C0, "disable-ifcvt-triangle", 22);
  LOWORD(qword_503B658) = 256;
  LOBYTE(qword_503B648) = 0;
  LOBYTE(dword_503B5CC) = dword_503B5CC & 0x9F | 0x20;
  sub_C53130(&qword_503B5C0);
  __cxa_atexit(sub_984900, &qword_503B5C0, &qword_4A427C0);
  v45 = 0;
  v46 = 1;
  v47[0] = &v45;
  sub_34EB9E0(&unk_503B4E0, "disable-ifcvt-triangle-rev", v47, &v46);
  __cxa_atexit(sub_984900, &unk_503B4E0, &qword_4A427C0);
  v45 = 0;
  v46 = 1;
  v47[0] = &v45;
  sub_34EBBE0(&unk_503B400, "disable-ifcvt-triangle-false", v47, &v46);
  __cxa_atexit(sub_984900, &unk_503B400, &qword_4A427C0);
  qword_503B320 = (__int64)&unk_49DC150;
  v31 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &unk_503B400, v29, v30), 1u);
  qword_503B370 = 0x100000000LL;
  dword_503B32C &= 0x8000u;
  qword_503B368 = (__int64)&unk_503B378;
  word_503B330 = 0;
  qword_503B338 = 0;
  dword_503B328 = v31;
  qword_503B340 = 0;
  qword_503B348 = 0;
  qword_503B350 = 0;
  qword_503B358 = 0;
  qword_503B360 = 0;
  qword_503B380 = 0;
  qword_503B388 = (__int64)&unk_503B3A0;
  qword_503B390 = 1;
  dword_503B398 = 0;
  byte_503B39C = 1;
  v32 = sub_C57470();
  v33 = (unsigned int)qword_503B370;
  if ( (unsigned __int64)(unsigned int)qword_503B370 + 1 > HIDWORD(qword_503B370) )
  {
    v44 = v32;
    sub_C8D5F0((char *)&unk_503B378 - 16, &unk_503B378, (unsigned int)qword_503B370 + 1LL, 8);
    v33 = (unsigned int)qword_503B370;
    v32 = v44;
  }
  *(_QWORD *)(qword_503B368 + 8 * v33) = v32;
  LODWORD(qword_503B370) = qword_503B370 + 1;
  qword_503B3A8 = 0;
  qword_503B3B0 = (__int64)&unk_49D9748;
  qword_503B3B8 = 0;
  qword_503B320 = (__int64)&unk_49DC090;
  qword_503B3C0 = (__int64)&unk_49DC1D0;
  qword_503B3E0 = (__int64)nullsub_23;
  qword_503B3D8 = (__int64)sub_984030;
  sub_C53080(&qword_503B320, "disable-ifcvt-diamond", 21);
  LOWORD(qword_503B3B8) = 256;
  LOBYTE(qword_503B3A8) = 0;
  LOBYTE(dword_503B32C) = dword_503B32C & 0x9F | 0x20;
  sub_C53130(&qword_503B320);
  __cxa_atexit(sub_984900, &qword_503B320, &qword_4A427C0);
  v45 = 0;
  v46 = 1;
  v47[0] = &v45;
  sub_34EBBE0(&unk_503B240, "disable-ifcvt-forked-diamond", v47, &v46);
  __cxa_atexit(sub_984900, &unk_503B240, &qword_4A427C0);
  qword_503B160 = (__int64)&unk_49DC150;
  v36 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &unk_503B240, v34, v35), 1u);
  qword_503B1B0 = 0x100000000LL;
  word_503B170 = 0;
  dword_503B16C &= 0x8000u;
  qword_503B178 = 0;
  qword_503B180 = 0;
  dword_503B168 = v36;
  qword_503B188 = 0;
  qword_503B190 = 0;
  qword_503B198 = 0;
  qword_503B1A0 = 0;
  qword_503B1A8 = (__int64)&unk_503B1B8;
  qword_503B1C0 = 0;
  qword_503B1C8 = (__int64)&unk_503B1E0;
  qword_503B1D0 = 1;
  dword_503B1D8 = 0;
  byte_503B1DC = 1;
  v37 = sub_C57470();
  v38 = (unsigned int)qword_503B1B0;
  v39 = (unsigned int)qword_503B1B0 + 1LL;
  if ( v39 > HIDWORD(qword_503B1B0) )
  {
    sub_C8D5F0((char *)&unk_503B1B8 - 16, &unk_503B1B8, v39, 8);
    v38 = (unsigned int)qword_503B1B0;
  }
  *(_QWORD *)(qword_503B1A8 + 8 * v38) = v37;
  LODWORD(qword_503B1B0) = qword_503B1B0 + 1;
  qword_503B1E8 = 0;
  qword_503B1F0 = (__int64)&unk_49D9748;
  qword_503B1F8 = 0;
  qword_503B160 = (__int64)&unk_49DC090;
  qword_503B200 = (__int64)&unk_49DC1D0;
  qword_503B220 = (__int64)nullsub_23;
  qword_503B218 = (__int64)sub_984030;
  sub_C53080(&qword_503B160, "ifcvt-branch-fold", 17);
  LOBYTE(qword_503B1E8) = 1;
  LOWORD(qword_503B1F8) = 257;
  LOBYTE(dword_503B16C) = dword_503B16C & 0x9F | 0x20;
  sub_C53130(&qword_503B160);
  return __cxa_atexit(sub_984900, &qword_503B160, &qword_4A427C0);
}
