// Function: ctor_417_0
// Address: 0x530e50
//
int ctor_417_0()
{
  int v0; // edx
  __int64 v1; // rbx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  int v5; // [rsp+4h] [rbp-ECh] BYREF
  int *v6; // [rsp+8h] [rbp-E8h] BYREF
  _QWORD v7[2]; // [rsp+10h] [rbp-E0h] BYREF
  _QWORD v8[2]; // [rsp+20h] [rbp-D0h] BYREF
  _QWORD v9[2]; // [rsp+30h] [rbp-C0h] BYREF
  int v10; // [rsp+40h] [rbp-B0h]
  const char *v11; // [rsp+48h] [rbp-A8h]
  __int64 v12; // [rsp+50h] [rbp-A0h]
  char *v13; // [rsp+58h] [rbp-98h]
  __int64 v14; // [rsp+60h] [rbp-90h]
  int v15; // [rsp+68h] [rbp-88h]
  const char *v16; // [rsp+70h] [rbp-80h]
  __int64 v17; // [rsp+78h] [rbp-78h]
  char *v18; // [rsp+80h] [rbp-70h]
  __int64 v19; // [rsp+88h] [rbp-68h]
  int v20; // [rsp+90h] [rbp-60h]
  const char *v21; // [rsp+98h] [rbp-58h]
  __int64 v22; // [rsp+A0h] [rbp-50h]
  char *v23; // [rsp+A8h] [rbp-48h]
  __int64 v24; // [rsp+B0h] [rbp-40h]
  int v25; // [rsp+B8h] [rbp-38h]
  const char *v26; // [rsp+C0h] [rbp-30h]
  __int64 v27; // [rsp+C8h] [rbp-28h]

  v9[0] = "unspecified";
  v11 = "Use the implementation defaults";
  v13 = "disable";
  v16 = "Disable the pass entirely";
  v18 = "optimize";
  v21 = "Optimise without changing ABI";
  v23 = "lowering";
  v26 = "Change variadic calling convention";
  v8[1] = 0x400000004LL;
  v6 = &v5;
  v7[0] = "Override the behaviour of expand-variadics";
  v8[0] = v9;
  v9[1] = 11;
  v10 = 0;
  v12 = 31;
  v14 = 7;
  v15 = 1;
  v17 = 25;
  v19 = 8;
  v20 = 2;
  v22 = 29;
  v24 = 8;
  v25 = 3;
  v27 = 34;
  v5 = 0;
  v7[1] = 42;
  qword_4FEFDE0 = (__int64)&unk_49DC150;
  v0 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  dword_4FEFDEC &= 0x8000u;
  word_4FEFDF0 = 0;
  qword_4FEFE30 = 0x100000000LL;
  qword_4FEFDF8 = 0;
  qword_4FEFE00 = 0;
  qword_4FEFE08 = 0;
  dword_4FEFDE8 = v0;
  qword_4FEFE10 = 0;
  qword_4FEFE18 = 0;
  qword_4FEFE20 = 0;
  qword_4FEFE28 = (__int64)&unk_4FEFE38;
  qword_4FEFE40 = 0;
  qword_4FEFE48 = (__int64)&unk_4FEFE60;
  qword_4FEFE50 = 1;
  dword_4FEFE58 = 0;
  byte_4FEFE5C = 1;
  v1 = sub_C57470();
  v2 = (unsigned int)qword_4FEFE30;
  v3 = (unsigned int)qword_4FEFE30 + 1LL;
  if ( v3 > HIDWORD(qword_4FEFE30) )
  {
    sub_C8D5F0((char *)&unk_4FEFE38 - 16, &unk_4FEFE38, v3, 8);
    v2 = (unsigned int)qword_4FEFE30;
  }
  *(_QWORD *)(qword_4FEFE28 + 8 * v2) = v1;
  LODWORD(qword_4FEFE30) = qword_4FEFE30 + 1;
  qword_4FEFE88 = (__int64)&qword_4FEFDE0;
  qword_4FEFE70 = (__int64)&unk_4A1F110;
  qword_4FEFE68 = 0;
  qword_4FEFE78 = 0;
  qword_4FEFDE0 = (__int64)&unk_4A1F180;
  qword_4FEFE80 = (__int64)&unk_4A1F130;
  qword_4FEFE90 = (__int64)&unk_4FEFEA0;
  qword_4FEFE98 = 0x800000000LL;
  qword_4FF0038 = (__int64)nullsub_1530;
  qword_4FF0030 = (__int64)sub_25B5E40;
  sub_25BB1C0(&qword_4FEFDE0, "expand-variadics-override", v7, &v6, v8);
  sub_C53130(&qword_4FEFDE0);
  if ( (_QWORD *)v8[0] != v9 )
    _libc_free(v8[0], "expand-variadics-override");
  return __cxa_atexit(sub_25B6440, &qword_4FEFDE0, &qword_4A427C0);
}
