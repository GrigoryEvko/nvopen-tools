// Function: ctor_619_0
// Address: 0x58b300
//
int __fastcall ctor_619_0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v8; // [rsp+0h] [rbp-F0h] BYREF
  __int64 *v9; // [rsp+8h] [rbp-E8h] BYREF
  _QWORD v10[2]; // [rsp+10h] [rbp-E0h] BYREF
  _QWORD v11[2]; // [rsp+20h] [rbp-D0h] BYREF
  _QWORD v12[2]; // [rsp+30h] [rbp-C0h] BYREF
  int v13; // [rsp+40h] [rbp-B0h]
  const char *v14; // [rsp+48h] [rbp-A8h]
  __int64 v15; // [rsp+50h] [rbp-A0h]
  const char *v16; // [rsp+58h] [rbp-98h]
  __int64 v17; // [rsp+60h] [rbp-90h]
  int v18; // [rsp+68h] [rbp-88h]
  const char *v19; // [rsp+70h] [rbp-80h]
  __int64 v20; // [rsp+78h] [rbp-78h]

  qword_502E420 = &unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  *(_DWORD *)&word_502E42C = word_502E42C & 0x8000;
  unk_502E430 = 0;
  qword_502E468[1] = 0x100000000LL;
  unk_502E428 = v4;
  unk_502E438 = 0;
  unk_502E440 = 0;
  unk_502E448 = 0;
  unk_502E450 = 0;
  unk_502E458 = 0;
  unk_502E460 = 0;
  qword_502E468[0] = &qword_502E468[2];
  qword_502E468[3] = 0;
  qword_502E468[4] = &qword_502E468[7];
  qword_502E468[5] = 1;
  LODWORD(qword_502E468[6]) = 0;
  BYTE4(qword_502E468[6]) = 1;
  v5 = sub_C57470();
  v6 = LODWORD(qword_502E468[1]);
  if ( (unsigned __int64)LODWORD(qword_502E468[1]) + 1 > HIDWORD(qword_502E468[1]) )
  {
    sub_C8D5F0(qword_502E468, &qword_502E468[2], LODWORD(qword_502E468[1]) + 1LL, 8);
    v6 = LODWORD(qword_502E468[1]);
  }
  *(_QWORD *)(qword_502E468[0] + 8 * v6) = v5;
  qword_502E468[8] = &qword_502E468[10];
  qword_502E468[13] = &qword_502E468[15];
  ++LODWORD(qword_502E468[1]);
  qword_502E468[9] = 0;
  qword_502E468[12] = &unk_49DC130;
  LOBYTE(qword_502E468[10]) = 0;
  qword_502E468[14] = 0;
  qword_502E420 = &unk_49DC010;
  LOBYTE(qword_502E468[15]) = 0;
  LOBYTE(qword_502E468[17]) = 0;
  qword_502E468[18] = &unk_49DC350;
  qword_502E468[22] = nullsub_92;
  qword_502E468[21] = sub_BC4D70;
  sub_C53080(&qword_502E420, "use-ctx-profile", 15);
  v11[0] = v12;
  sub_30A69B0(v11, byte_3F871B3);
  sub_2240AE0(&qword_502E468[8], v11);
  LOBYTE(qword_502E468[17]) = 1;
  sub_2240AE0(&qword_502E468[13], v11);
  if ( (_QWORD *)v11[0] != v12 )
    j_j___libc_free_0(v11[0], v12[0] + 1LL);
  unk_502E450 = 41;
  LOBYTE(word_502E42C) = word_502E42C & 0x9F | 0x20;
  unk_502E448 = "Use the specified contextual profile file";
  sub_C53130(&qword_502E420);
  __cxa_atexit(sub_BC5A40, &qword_502E420, &qword_4A427C0);
  v10[0] = "Verbosity level of the contextual profile printer pass.";
  v12[0] = "everything";
  v14 = "print everything - most verbose";
  v16 = "yaml";
  v19 = "just the yaml representation of the profile";
  v11[1] = 0x400000002LL;
  v10[1] = 55;
  v11[0] = v12;
  v12[1] = 10;
  v13 = 0;
  v15 = 31;
  v17 = 4;
  v18 = 1;
  v20 = 43;
  v9 = &v8;
  ((void (__fastcall *)(void *, const char *, __int64 **, char *, _QWORD *, _QWORD *, __int64))sub_30AB120)(
    &unk_502E1C0,
    "ctx-profile-printer-level",
    &v9,
    (char *)&v8 + 4,
    v11,
    v10,
    0x100000001LL);
  if ( (_QWORD *)v11[0] != v12 )
    _libc_free(v11[0], "ctx-profile-printer-level");
  return __cxa_atexit(sub_30A7060, &unk_502E1C0, &qword_4A427C0);
}
