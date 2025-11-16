// Function: sub_C91D20
// Address: 0xc91d20
//
__int64 sub_C91D20()
{
  __int64 result; // rax
  int v1; // edx
  __int64 *v2; // rbx
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rax
  unsigned __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  int v12; // edx
  __int64 *v13; // rbx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rax
  _QWORD v23[4]; // [rsp+0h] [rbp-40h] BYREF
  char v24; // [rsp+20h] [rbp-20h]
  char v25; // [rsp+21h] [rbp-1Fh]

  if ( byte_4F84E08 || !(unsigned int)sub_2207590(&byte_4F84E08) )
  {
    result = (unsigned __int8)byte_4F84D20;
    if ( byte_4F84D20 )
      return result;
    goto LABEL_10;
  }
  qword_4F84E20 = (__int64)&unk_49DC150;
  v1 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  byte_4F84E9C = 1;
  qword_4F84E70 = 0x100000000LL;
  dword_4F84E2C &= 0x8000u;
  qword_4F84E38 = 0;
  qword_4F84E40 = 0;
  qword_4F84E48 = 0;
  dword_4F84E28 = v1;
  word_4F84E30 = 0;
  qword_4F84E50 = 0;
  qword_4F84E58 = 0;
  qword_4F84E60 = 0;
  qword_4F84E68 = (__int64)&unk_4F84E78;
  qword_4F84E80 = 0;
  qword_4F84E88 = (__int64)&unk_4F84EA0;
  qword_4F84E90 = 1;
  dword_4F84E98 = 0;
  v2 = sub_C57470();
  v5 = (unsigned int)qword_4F84E70;
  v6 = (unsigned int)qword_4F84E70 + 1LL;
  if ( v6 > HIDWORD(qword_4F84E70) )
  {
    sub_C8D5F0((__int64)&unk_4F84E78 - 16, &unk_4F84E78, v6, 8u, v3, v4);
    v5 = (unsigned int)qword_4F84E70;
  }
  *(_QWORD *)(qword_4F84E68 + 8 * v5) = v2;
  LODWORD(qword_4F84E70) = qword_4F84E70 + 1;
  byte_4F84EB9 = 0;
  qword_4F84EB0 = (__int64)&unk_49D9748;
  qword_4F84EA8 = 0;
  qword_4F84E20 = (__int64)&unk_49D9AD8;
  qword_4F84EC0 = (__int64)&unk_49DC1D0;
  qword_4F84EE0 = (__int64)nullsub_39;
  qword_4F84ED8 = (__int64)sub_AA4180;
  sub_C53080((__int64)&qword_4F84E20, (__int64)"stats", 5);
  qword_4F84E50 = 62;
  qword_4F84E48 = (__int64)"Enable statistics output from program (available with Asserts)";
  if ( qword_4F84EA8 )
  {
    v11 = sub_CEADF0(&qword_4F84E20, "stats", v7, v8, v9, v10);
    v25 = 1;
    v23[0] = "cl::location(x) specified more than once!";
    v24 = 3;
    sub_C53280((__int64)&qword_4F84E20, (__int64)v23, 0, 0, v11);
  }
  else
  {
    byte_4F84EB9 = 1;
    qword_4F84EA8 = (__int64)&byte_4F84EEA;
    byte_4F84EB8 = byte_4F84EEA;
  }
  LOBYTE(dword_4F84E2C) = dword_4F84E2C & 0x9F | 0x20;
  sub_C53130((__int64)&qword_4F84E20);
  __cxa_atexit((void (*)(void *))sub_AA4490, &qword_4F84E20, &qword_4A427C0);
  sub_2207640(&byte_4F84E08);
  result = (unsigned __int8)byte_4F84D20;
  if ( !byte_4F84D20 )
  {
LABEL_10:
    result = sub_2207590(&byte_4F84D20);
    if ( (_DWORD)result )
    {
      qword_4F84D40 = (__int64)&unk_49DC150;
      v12 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
      dword_4F84D4C &= 0x8000u;
      word_4F84D50 = 0;
      qword_4F84D90 = 0x100000000LL;
      qword_4F84D58 = 0;
      qword_4F84D60 = 0;
      qword_4F84D68 = 0;
      dword_4F84D48 = v12;
      qword_4F84D70 = 0;
      qword_4F84D78 = 0;
      qword_4F84D80 = 0;
      qword_4F84D88 = (__int64)&unk_4F84D98;
      qword_4F84DA0 = 0;
      qword_4F84DA8 = (__int64)&unk_4F84DC0;
      qword_4F84DB0 = 1;
      dword_4F84DB8 = 0;
      byte_4F84DBC = 1;
      v13 = sub_C57470();
      v16 = (unsigned int)qword_4F84D90;
      v17 = (unsigned int)qword_4F84D90 + 1LL;
      if ( v17 > HIDWORD(qword_4F84D90) )
      {
        sub_C8D5F0((__int64)&unk_4F84D98 - 16, &unk_4F84D98, v17, 8u, v14, v15);
        v16 = (unsigned int)qword_4F84D90;
      }
      *(_QWORD *)(qword_4F84D88 + 8 * v16) = v13;
      LODWORD(qword_4F84D90) = qword_4F84D90 + 1;
      byte_4F84DD9 = 0;
      qword_4F84DD0 = (__int64)&unk_49D9748;
      qword_4F84DC8 = 0;
      qword_4F84D40 = (__int64)&unk_49D9AD8;
      qword_4F84DE0 = (__int64)&unk_49DC1D0;
      qword_4F84E00 = (__int64)nullsub_39;
      qword_4F84DF8 = (__int64)sub_AA4180;
      sub_C53080((__int64)&qword_4F84D40, (__int64)"stats-json", 10);
      qword_4F84D70 = 31;
      qword_4F84D68 = (__int64)"Display statistics as json data";
      if ( qword_4F84DC8 )
      {
        v22 = sub_CEADF0(&qword_4F84D40, "stats-json", v18, v19, v20, v21);
        v25 = 1;
        v23[0] = "cl::location(x) specified more than once!";
        v24 = 3;
        sub_C53280((__int64)&qword_4F84D40, (__int64)v23, 0, 0, v22);
      }
      else
      {
        byte_4F84DD9 = 1;
        qword_4F84DC8 = (__int64)&byte_4F84EE9;
        byte_4F84DD8 = byte_4F84EE9;
      }
      LOBYTE(dword_4F84D4C) = dword_4F84D4C & 0x9F | 0x20;
      sub_C53130((__int64)&qword_4F84D40);
      __cxa_atexit((void (*)(void *))sub_AA4490, &qword_4F84D40, &qword_4A427C0);
      return sub_2207640(&byte_4F84D20);
    }
  }
  return result;
}
