// Function: ctor_584_0
// Address: 0x579f60
//
int __fastcall ctor_584_0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  int v9; // [rsp+0h] [rbp-F0h] BYREF
  int v10; // [rsp+4h] [rbp-ECh] BYREF
  int *v11; // [rsp+8h] [rbp-E8h] BYREF
  _QWORD v12[2]; // [rsp+10h] [rbp-E0h] BYREF
  _QWORD v13[2]; // [rsp+20h] [rbp-D0h] BYREF
  _QWORD v14[2]; // [rsp+30h] [rbp-C0h] BYREF
  int v15; // [rsp+40h] [rbp-B0h]
  char *v16; // [rsp+48h] [rbp-A8h]
  __int64 v17; // [rsp+50h] [rbp-A0h]
  char *v18; // [rsp+58h] [rbp-98h]
  __int64 v19; // [rsp+60h] [rbp-90h]
  int v20; // [rsp+68h] [rbp-88h]
  const char *v21; // [rsp+70h] [rbp-80h]
  __int64 v22; // [rsp+78h] [rbp-78h]
  const char *v23; // [rsp+80h] [rbp-70h]
  __int64 v24; // [rsp+88h] [rbp-68h]
  int v25; // [rsp+90h] [rbp-60h]
  const char *v26; // [rsp+98h] [rbp-58h]
  __int64 v27; // [rsp+A0h] [rbp-50h]
  char *v28; // [rsp+A8h] [rbp-48h]
  __int64 v29; // [rsp+B0h] [rbp-40h]
  int v30; // [rsp+B8h] [rbp-38h]
  const char *v31; // [rsp+C0h] [rbp-30h]
  __int64 v32; // [rsp+C8h] [rbp-28h]

  v14[0] = "default";
  v16 = "Default";
  v18 = "release";
  v21 = "precompiled";
  v23 = "development";
  v26 = "for training";
  v28 = "dummy";
  v31 = "prioritize low virtual register numbers for test and debug";
  v13[1] = 0x400000004LL;
  v12[0] = "Enable regalloc advisor mode";
  v13[0] = v14;
  v14[1] = 7;
  v15 = 0;
  v17 = 7;
  v19 = 7;
  v20 = 1;
  v22 = 11;
  v24 = 11;
  v25 = 2;
  v27 = 12;
  v29 = 5;
  v30 = 3;
  v32 = 58;
  v12[1] = 28;
  v10 = 0;
  v11 = &v10;
  v9 = 1;
  qword_5024440 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  dword_502444C &= 0x8000u;
  word_5024450 = 0;
  qword_5024490 = 0x100000000LL;
  qword_5024458 = 0;
  qword_5024460 = 0;
  qword_5024468 = 0;
  dword_5024448 = v4;
  qword_5024470 = 0;
  qword_5024478 = 0;
  qword_5024480 = 0;
  qword_5024488 = (__int64)&unk_5024498;
  qword_50244A0 = 0;
  qword_50244A8 = (__int64)&unk_50244C0;
  qword_50244B0 = 1;
  dword_50244B8 = 0;
  byte_50244BC = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_5024490;
  v7 = (unsigned int)qword_5024490 + 1LL;
  if ( v7 > HIDWORD(qword_5024490) )
  {
    sub_C8D5F0((char *)&unk_5024498 - 16, &unk_5024498, v7, 8);
    v6 = (unsigned int)qword_5024490;
  }
  *(_QWORD *)(qword_5024488 + 8 * v6) = v5;
  LODWORD(qword_5024490) = qword_5024490 + 1;
  qword_50244E8 = (__int64)&qword_5024440;
  qword_50244D0 = (__int64)&unk_4A2B2D8;
  qword_50244C8 = 0;
  qword_50244D8 = 0;
  qword_5024440 = (__int64)&unk_4A2B348;
  qword_50244E0 = (__int64)&unk_4A2B2F8;
  qword_50244F0 = (__int64)&unk_5024500;
  qword_50244F8 = 0x800000000LL;
  qword_5024698 = (__int64)nullsub_1655;
  qword_5024690 = (__int64)sub_2F5E720;
  sub_2F5F6A0(&qword_5024440, "regalloc-enable-priority-advisor", &v9, &v11, v12, v13);
  sub_C53130(&qword_5024440);
  if ( (_QWORD *)v13[0] != v14 )
    _libc_free(v13[0], "regalloc-enable-priority-advisor");
  return __cxa_atexit(sub_2F5F0F0, &qword_5024440, &qword_4A427C0);
}
