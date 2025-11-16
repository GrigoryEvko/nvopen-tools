// Function: sub_C60B10
// Address: 0xc60b10
//
__int64 *sub_C60B10()
{
  int v1; // edx
  __int64 *v2; // r12
  __int64 v3; // rax
  unsigned __int64 v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rax
  int v10; // edx
  __int64 *v11; // r12
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rax
  _BYTE *v19; // rax
  int v20; // edx
  __int64 *v21; // rbx
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  const char *v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rax
  char *v30; // rax
  _QWORD v31[4]; // [rsp-68h] [rbp-68h] BYREF
  char v32; // [rsp-48h] [rbp-48h]
  char v33; // [rsp-47h] [rbp-47h]

  if ( byte_4F83D48 )
    return &qword_4F83D60;
  if ( (unsigned int)sub_2207590(&byte_4F83D48) )
  {
    qword_4F83D60 = 0;
    qword_4F83D98 = (__int64)&qword_4F83D88;
    qword_4F83DA0 = (__int64)&qword_4F83D88;
    word_4F83DC8 = 0;
    qword_4F83D68 = 0;
    qword_4F83D70 = 0;
    qword_4F83D78 = 0;
    qword_4F83D88 = 0;
    qword_4F83D90 = 0;
    qword_4F83DA8 = 0;
    qword_4F83DB0 = 0;
    qword_4F83DB8 = 0;
    qword_4F83DC0 = 0;
    byte_4F83DCA = 0;
    qword_4F83DD0 = (__int64)&unk_49DC150;
    v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    word_4F83DE0 = 0;
    qword_4F83DE8 = 0;
    qword_4F83DF0 = 0;
    dword_4F83DDC = dword_4F83DDC & 0x8000 | 1;
    qword_4F83E20 = 0x100000000LL;
    dword_4F83DD8 = v1;
    qword_4F83DF8 = 0;
    qword_4F83E00 = 0;
    qword_4F83E08 = 0;
    qword_4F83E10 = 0;
    qword_4F83E18 = (__int64)&unk_4F83E28;
    qword_4F83E30 = 0;
    qword_4F83E38 = (__int64)&unk_4F83E50;
    qword_4F83E40 = 1;
    dword_4F83E48 = 0;
    byte_4F83E4C = 1;
    v2 = sub_C57470();
    v3 = (unsigned int)qword_4F83E20;
    v4 = (unsigned int)qword_4F83E20 + 1LL;
    if ( v4 > HIDWORD(qword_4F83E20) )
    {
      sub_C8D5F0((char *)&unk_4F83E28 - 16, &unk_4F83E28, v4, 8);
      v3 = (unsigned int)qword_4F83E20;
    }
    *(_QWORD *)(qword_4F83E18 + 8 * v3) = v2;
    LODWORD(qword_4F83E20) = qword_4F83E20 + 1;
    qword_4F83E58 = 0;
    qword_4F83DD0 = (__int64)&unk_49DC600;
    qword_4F83E60 = 0;
    byte_4F83E78 = 0;
    qword_4F83E98 = (__int64)&unk_49DC350;
    qword_4F83E68 = 0;
    qword_4F83EB8 = (__int64)nullsub_149;
    qword_4F83E70 = 0;
    qword_4F83EB0 = (__int64)sub_C5F7C0;
    qword_4F83E80 = 0;
    qword_4F83E88 = 0;
    qword_4F83E90 = 0;
    sub_C53080((__int64)&qword_4F83DD0, (__int64)"debug-counter", 13);
    BYTE1(dword_4F83DDC) |= 2u;
    qword_4F83E00 = 52;
    LOBYTE(dword_4F83DDC) = dword_4F83DDC & 0x9F | 0x20;
    qword_4F83DF8 = (__int64)"Comma separated list of debug counter skip and count";
    if ( qword_4F83E58 )
    {
      v9 = sub_CEADF0(&qword_4F83DD0, "debug-counter", v5, v6, v7, v8);
      v33 = 1;
      v31[0] = "cl::location(x) specified more than once!";
      v32 = 3;
      sub_C53280((__int64)&qword_4F83DD0, (__int64)v31, 0, 0, v9);
    }
    else
    {
      qword_4F83E58 = (__int64)&qword_4F83D60;
    }
    sub_C53130((__int64)&qword_4F83DD0);
    qword_4F83DD0 = (__int64)off_49DC680;
    qword_4F83EC0 = (__int64)&unk_49DC150;
    v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    qword_4F83F10 = 0x100000000LL;
    dword_4F83ECC &= 0x8000u;
    word_4F83ED0 = 0;
    qword_4F83ED8 = 0;
    qword_4F83EE0 = 0;
    dword_4F83EC8 = v10;
    qword_4F83EE8 = 0;
    qword_4F83EF0 = 0;
    qword_4F83EF8 = 0;
    qword_4F83F00 = 0;
    qword_4F83F08 = (__int64)&unk_4F83F18;
    qword_4F83F20 = 0;
    qword_4F83F28 = (__int64)&unk_4F83F40;
    qword_4F83F30 = 1;
    dword_4F83F38 = 0;
    byte_4F83F3C = 1;
    v11 = sub_C57470();
    v12 = (unsigned int)qword_4F83F10;
    v13 = (unsigned int)qword_4F83F10 + 1LL;
    if ( v13 > HIDWORD(qword_4F83F10) )
    {
      sub_C8D5F0((char *)&unk_4F83F18 - 16, &unk_4F83F18, v13, 8);
      v12 = (unsigned int)qword_4F83F10;
    }
    *(_QWORD *)(qword_4F83F08 + 8 * v12) = v11;
    qword_4F83F50 = (__int64)&unk_49D9748;
    qword_4F83EC0 = (__int64)&unk_49D9AD8;
    qword_4F83F60 = (__int64)&unk_49DC1D0;
    LODWORD(qword_4F83F10) = qword_4F83F10 + 1;
    qword_4F83F80 = (__int64)nullsub_39;
    qword_4F83F48 = 0;
    qword_4F83F78 = (__int64)sub_AA4180;
    byte_4F83F59 = 0;
    sub_C53080((__int64)&qword_4F83EC0, (__int64)"print-debug-counter", 19);
    LOBYTE(dword_4F83ECC) = dword_4F83ECC & 0x98 | 0x20;
    if ( qword_4F83F48 )
    {
      v18 = sub_CEADF0(&qword_4F83EC0, "print-debug-counter", v14, v15, v16, v17);
      v33 = 1;
      v31[0] = "cl::location(x) specified more than once!";
      v32 = 3;
      sub_C53280((__int64)&qword_4F83EC0, (__int64)v31, 0, 0, v18);
      v19 = (_BYTE *)qword_4F83F48;
    }
    else
    {
      v19 = (char *)&word_4F83DC8 + 1;
      qword_4F83F48 = (__int64)&word_4F83DC8 + 1;
    }
    *v19 = 0;
    unk_4F83F58 = 256;
    qword_4F83EE8 = (__int64)"Print out debug counter info after all counters accumulated";
    qword_4F83EF0 = 59;
    sub_C53130((__int64)&qword_4F83EC0);
    qword_4F83F88 = (__int64)&unk_49DC150;
    v20 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
    qword_4F83FD8 = 0x100000000LL;
    word_4F83F98 = 0;
    dword_4F83F94 &= 0x8000u;
    qword_4F83FA0 = 0;
    qword_4F83FA8 = 0;
    dword_4F83F90 = v20;
    qword_4F83FB0 = 0;
    qword_4F83FB8 = 0;
    qword_4F83FC0 = 0;
    qword_4F83FC8 = 0;
    qword_4F83FD0 = (__int64)&unk_4F83FE0;
    qword_4F83FE8 = 0;
    qword_4F83FF0 = (__int64)&algn_4F84005[3];
    qword_4F83FF8 = 1;
    dword_4F84000 = 0;
    byte_4F84004 = 1;
    v21 = sub_C57470();
    v22 = (unsigned int)qword_4F83FD8;
    v23 = (unsigned int)qword_4F83FD8 + 1LL;
    if ( v23 > HIDWORD(qword_4F83FD8) )
    {
      sub_C8D5F0((char *)&unk_4F83FE0 - 16, &unk_4F83FE0, v23, 8);
      v22 = (unsigned int)qword_4F83FD8;
    }
    v24 = "debug-counter-break-on-last";
    *(_QWORD *)(qword_4F83FD0 + 8 * v22) = v21;
    qword_4F84018 = (__int64)&unk_49D9748;
    qword_4F83F88 = (__int64)&unk_49D9AD8;
    qword_4F84028 = (__int64)&unk_49DC1D0;
    LODWORD(qword_4F83FD8) = qword_4F83FD8 + 1;
    qword_4F84048 = (__int64)nullsub_39;
    qword_4F84010 = 0;
    qword_4F84040 = (__int64)sub_AA4180;
    byte_4F84021 = 0;
    sub_C53080((__int64)&qword_4F83F88, (__int64)"debug-counter-break-on-last", 27);
    LOBYTE(dword_4F83F94) = dword_4F83F94 & 0x98 | 0x20;
    if ( qword_4F84010 )
    {
      v29 = sub_CEADF0(&qword_4F83F88, "debug-counter-break-on-last", v25, v26, v27, v28);
      v24 = (const char *)v31;
      v33 = 1;
      v31[0] = "cl::location(x) specified more than once!";
      v32 = 3;
      sub_C53280((__int64)&qword_4F83F88, (__int64)v31, 0, 0, v29);
      v30 = (char *)qword_4F84010;
    }
    else
    {
      v30 = &byte_4F83DCA;
      qword_4F84010 = (__int64)&byte_4F83DCA;
    }
    *v30 = 0;
    unk_4F84020 = 256;
    qword_4F83FB0 = (__int64)"Insert a break point on the last enabled count of a chunks list";
    qword_4F83FB8 = 63;
    sub_C53130((__int64)&qword_4F83F88);
    sub_C5F790((__int64)&qword_4F83F88, (__int64)v24);
    __cxa_atexit(sub_C62F20, &qword_4F83D60, &qword_4A427C0);
    sub_2207640(&byte_4F83D48);
  }
  return &qword_4F83D60;
}
